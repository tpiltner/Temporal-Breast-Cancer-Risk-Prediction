# modelArchitecture.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from imageEncoder import get_image_encoder

# Time encoding
class ContinuousPosEncoding(nn.Module):
    """
    Continuous sinusoidal time encoding with linear interpolation.

    Used for the multi-prior temporal models to encode how far each prior
    visit is from the current exam.
    """
    def __init__(self, dim: int, drop: float = 0.1, maxtime: float = 10.0, num_steps: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(drop)
        self.maxtime = float(maxtime)
        self.num_steps = int(num_steps)

        position = torch.linspace(0, self.maxtime, steps=self.num_steps).unsqueeze(1)  # [S,1]
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe = torch.zeros(self.num_steps, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def interp(self, times: torch.Tensor) -> torch.Tensor:
        """
        times: [B,S] or [B,K] in years
        returns interpolated positional encoding of shape [B,S,dim] or [B,K,dim]
        """
        times = torch.clamp(times, 0.0, self.maxtime) * (self.num_steps - 1) / self.maxtime
        t_floor = torch.floor(times).long().clamp(0, self.num_steps - 1)
        t_ceil = torch.ceil(times).long().clamp(0, self.num_steps - 1)
        alpha = (times - t_floor.float()).unsqueeze(-1)

        pe_floor = self.pe[t_floor]
        pe_ceil = self.pe[t_ceil]
        return (1.0 - alpha) * pe_floor + alpha * pe_ceil


# Transformer block
class TemporalAttentionLayer(nn.Module):
    """
    Standard transformer encoder-style block.
    Input expected as [N,B,C] with batch_first=False.
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + self.drop1(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.drop2(ff_out))
        return x


# Monotone cumulative hazard head
class CumulativeProbabilityLayer(nn.Module):
    """
    Monotone cumulative logits:
      z_t = base + cumsum(softplus(inc_t))

    Produces nondecreasing logits across prediction horizons.
    """
    def __init__(self, in_dim: int, horizons: int = 5, dropout: float = 0.0):
        super().__init__()
        self.base = nn.Linear(in_dim, 1)
        self.inc = nn.Linear(in_dim, horizons)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        base = self.base(x)                 # [B,1]
        inc = F.softplus(self.inc(x))       # [B,H]
        z = base + torch.cumsum(inc, dim=1)
        return z


# View-attention pooling
class ViewAttentionPooling(nn.Module):
    """
    Learn attention weights over the 4 mammography views and produce a
    pooled patient-level vector.
    """
    def __init__(self, dim: int = 512, hidden: int = 128, temperature: float = 1.0):
        super().__init__()
        self.temperature = float(temperature)
        self.attn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        # Initialize to approximately uniform attention
        nn.init.zeros_(self.attn[-1].weight)
        nn.init.zeros_(self.attn[-1].bias)

    @staticmethod
    def masked_softmax(logits: torch.Tensor, mask_keep: torch.Tensor) -> torch.Tensor:
        """
        logits: [B,V]
        mask_keep: [B,V] bool True=keep
        """
        mask_keep = mask_keep.bool()
        none_valid = (mask_keep.sum(dim=1) == 0)

        logits_f = logits.float().masked_fill(~mask_keep, float("-inf"))
        if none_valid.any():
            logits_f = torch.where(none_valid.unsqueeze(1), torch.zeros_like(logits_f), logits_f)

        weights = torch.softmax(logits_f, dim=1)
        if none_valid.any():
            weights = torch.where(none_valid.unsqueeze(1), torch.zeros_like(weights), weights)

        return weights.to(dtype=logits.dtype)

    def forward(self, v: torch.Tensor, mask_keep: torch.Tensor = None):
        """
        v: [B,4,C]
        mask_keep: [B,4] bool, optional

        returns:
          pooled: [B,C]
          weights: [B,4]
        """
        scores = self.attn(v).squeeze(-1)  # [B,4]
        if self.temperature != 1.0:
            scores = scores / self.temperature

        if mask_keep is None:
            weights = torch.softmax(scores.float(), dim=1).to(dtype=v.dtype)
        else:
            weights = self.masked_softmax(scores, mask_keep).to(dtype=v.dtype)

        pooled = (v * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights


# Encoder wrapper
class ImageBackbone(nn.Module):
    """
    Shared ResNet-18 
    Encodes 4-view images into 4 view-level 512-d vectors.
    """
    def __init__(self, dim: int = 512, pretrained: bool = True, freeze_encoder: bool = False):
        super().__init__()
        self.dim = int(dim)
        self.freeze_encoder_flag = bool(freeze_encoder)

        self.encoder = get_image_encoder(pretrained=pretrained)

        if self.freeze_encoder_flag:
            for p in self.encoder.parameters():
                p.requires_grad = False
            if hasattr(self.encoder, "freeze"):
                self.encoder.freeze()

    def _set_bn_eval_if_frozen(self):
        if not self.freeze_encoder_flag:
            return
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def encode_views(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: [B,4,1,H,W]
        returns: [B,4,dim]
        """
        assert imgs.dim() == 5, f"Expected [B,4,1,H,W], got {tuple(imgs.shape)}"
        B, V, C, H, W = imgs.shape
        assert V == 4, f"Expected 4 views, got {V}"

        self._set_bn_eval_if_frozen()

        x = imgs.reshape(B * V, C, H, W)
        fmap = self.encoder(x, return_map=True)   # [B*4,dim,h,w]
        fmap = F.adaptive_avg_pool2d(fmap, (1, 1)).flatten(1)  # [B*4,dim]

        if fmap.shape[1] != self.dim:
            raise RuntimeError(f"Encoder returned dim {fmap.shape[1]}, expected {self.dim}")

        return fmap.view(B, V, self.dim)  # [B,4,dim]


# Baseline current-only model
class BaselineCurrentOnlyModel(nn.Module):
    """
    Baseline model architecture:
      4 current views
      -> encoder
      -> 512-d feature per image
      -> view attention pooling
      -> MLP head
      -> cumulative hazard prediction
    """
    def __init__(
        self,
        pretrained_encoder: bool = True,
        num_years: int = 5,
        dim: int = 512,
        mlp_hidden: int = 512,
        mlp_layers: int = 1,
        dropout: float = 0.2,
        freeze_encoder: bool = False,
        attn_hidden: int = 128,
        attn_temperature: float = 1.0,
        cum_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_years = int(num_years)
        self.dim = int(dim)

        self.backbone = ImageBackbone(
            dim=self.dim,
            pretrained=pretrained_encoder,
            freeze_encoder=freeze_encoder,
        )
        self.view_pool = ViewAttentionPooling(
            dim=self.dim,
            hidden=attn_hidden,
            temperature=attn_temperature,
        )

        layers = []
        in_dim = self.dim
        L = max(1, int(mlp_layers))
        for i in range(L):
            out_dim = mlp_hidden if i < L - 1 else self.dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < L - 1:
                layers.append(nn.ReLU(inplace=True))
                if dropout and float(dropout) > 0:
                    layers.append(nn.Dropout(float(dropout)))
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        self.cum = CumulativeProbabilityLayer(self.dim, horizons=self.num_years, dropout=float(cum_dropout))

    def forward(self, imgs: torch.Tensor, delta_feat: torch.Tensor = None, has_prior_views: torch.Tensor = None):
        """
        imgs: [B,4,1,H,W] or [B,12,1,H,W] (uses first 4 as current)
        """
        assert imgs.dim() == 5, f"Expected [B,V,1,H,W], got {tuple(imgs.shape)}"
        V = imgs.size(1)

        if V == 4:
            cur_imgs = imgs
        elif V == 12:
            cur_imgs = imgs[:, 0:4]
        else:
            raise AssertionError(f"Expected V=4 or V=12, got {V}")

        cur_vecs = self.backbone.encode_views(cur_imgs)     # [B,4,dim]
        fused_vec, weights = self.view_pool(cur_vecs)       # [B,dim]
        fused_vec = self.mlp(fused_vec)                     # [B,dim]
        logits = self.cum(fused_vec)                        # [B,num_years]

        return {
            "risk_prediction": {"pred_fused": logits},
            "attention_weights": weights,
        }


# Single-prior temporal core
class SinglePriorTemporalCore(nn.Module):
    """
    Single-prior temporal architecture:
      prior + current
      -> encoder
      -> view attention pooling for each exam
      -> diff = current - prior
      -> tokens [prior, diff, current]
      -> transformer
      -> mean fusion
      -> cumulative hazard prediction

    This same core is used for both aligned and non-aligned versions.
    The only difference should be which dataset provides imgs[:,4:8].
    """
    def __init__(
        self,
        pretrained_encoder: bool = True,
        num_years: int = 5,
        dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        attn_hidden: int = 128,
        attn_temperature: float = 1.0,
        cum_dropout: float = 0.0,
        return_attention: bool = False,
    ):
        super().__init__()
        self.num_years = int(num_years)
        self.dim = int(dim)
        self.return_attention = bool(return_attention)

        self.backbone = ImageBackbone(
            dim=self.dim,
            pretrained=pretrained_encoder,
            freeze_encoder=freeze_encoder,
        )
        self.view_pool = ViewAttentionPooling(
            dim=self.dim,
            hidden=attn_hidden,
            temperature=attn_temperature,
        )
        self.token_attn = TemporalAttentionLayer(
            dim=self.dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.feature_projection = nn.Linear(self.dim, self.dim)
        self.head_dropout = nn.Dropout(float(dropout)) if dropout and float(dropout) > 0 else nn.Identity()

        self.cum_fused = CumulativeProbabilityLayer(self.dim, horizons=self.num_years, dropout=float(cum_dropout))
        self.cum_cur = CumulativeProbabilityLayer(self.dim, horizons=self.num_years, dropout=float(cum_dropout))
        self.cum_pri = CumulativeProbabilityLayer(self.dim, horizons=self.num_years, dropout=float(cum_dropout))

    def forward(self, imgs: torch.Tensor, delta_feat: torch.Tensor, has_prior_views: torch.Tensor):
        """
        imgs: [B,12,1,H,W]
              uses imgs[:,0:4] as current
              uses imgs[:,4:8] as prior
              ignores imgs[:,8:12]
        has_prior_views: [B,4]
        """
        assert imgs.dim() == 5, f"Expected [B,12,1,H,W], got {tuple(imgs.shape)}"
        B, V, C, H, W = imgs.shape
        assert V == 12, f"Expected V=12, got V={V}"
        assert has_prior_views.shape == (B, 4), f"has_prior_views must be [B,4], got {tuple(has_prior_views.shape)}"

        cur_imgs = imgs[:, 0:4]
        pri_imgs = imgs[:, 4:8]

        cur_vecs = self.backbone.encode_views(cur_imgs)    # [B,4,dim]
        pri_vecs = self.backbone.encode_views(pri_imgs)    # [B,4,dim]

        prior_keep = (has_prior_views > 0.5)               # [B,4]
        cur_tok, w_cur = self.view_pool(cur_vecs, mask_keep=None)
        pri_tok, w_pri = self.view_pool(pri_vecs, mask_keep=prior_keep)

        diff_tok = cur_tok - pri_tok

        seq = torch.stack([pri_tok, diff_tok, cur_tok], dim=0)  # [3,B,dim]

        has_any_prior = (prior_keep.sum(dim=1) > 0)
        token_mask = torch.stack(
            [
                ~has_any_prior,
                ~has_any_prior,
                torch.zeros_like(has_any_prior, dtype=torch.bool),
            ],
            dim=1,
        )  # [B,3]

        seq_att = self.token_attn(seq, key_padding_mask=token_mask)

        keep = (~token_mask).to(dtype=seq_att.dtype).T.unsqueeze(-1)  # [3,B,1]
        fused = (seq_att * keep).sum(dim=0) / keep.sum(dim=0).clamp(min=1.0)
        fused = self.feature_projection(fused)

        fused_h = self.head_dropout(fused)
        cur_h = self.head_dropout(cur_tok)
        pri_h = self.head_dropout(pri_tok)

        out = {
            "pred_fused": self.cum_fused(fused_h),
            "pred_cur": self.cum_cur(cur_h),
            "pred_pri": self.cum_pri(pri_h),
        }

        if self.return_attention:
            out["attention"] = {
                "view_cur": w_cur,
                "view_pri": w_pri,
                "token_mask": token_mask,
            }

        return out

class RiskModel_no_alignment(nn.Module):
    def __init__(self, pretrained_encoder=True, num_years=5, freeze_encoder=False, **kwargs):
        super().__init__()
        self.risk_prediction_model = SinglePriorTemporalCore(
            pretrained_encoder=pretrained_encoder,
            num_years=num_years,
            freeze_encoder=freeze_encoder,
            **kwargs,
        )

    def forward(self, imgs: torch.Tensor, delta_feat: torch.Tensor, has_prior_views: torch.Tensor):
        pred = self.risk_prediction_model(imgs, delta_feat, has_prior_views)
        return {"risk_prediction": pred}


class RiskModel_alignedprior(nn.Module):
    def __init__(self, pretrained_encoder=True, num_years=5, freeze_encoder=False, **kwargs):
        super().__init__()
        self.risk_prediction_model = SinglePriorTemporalCore(
            pretrained_encoder=pretrained_encoder,
            num_years=num_years,
            freeze_encoder=freeze_encoder,
            **kwargs,
        )

    def forward(self, imgs: torch.Tensor, delta_feat: torch.Tensor, has_prior_views: torch.Tensor):
        pred = self.risk_prediction_model(imgs, delta_feat, has_prior_views)
        return {"risk_prediction": pred}

# Multi-prior temporal core
class MultiPriorTemporalCore(nn.Module):
    """
    Multi-prior temporal architecture:
      current + K priors
      -> encoder
      -> view attention pooling for each visit
      -> diff_k = current - prior_k
      -> token sequence [pri_K, diff_K, ..., pri_1, diff_1, current]
      -> time encoding
      -> transformer
      -> mean fusion
      -> cumulative hazard prediction

    This same core is used for both aligned and non-aligned versions.
    The only difference should be whether input priors are raw or aligned.
    """
    def __init__(
        self,
        num_years: int = 5,
        dim: int = 512,
        heads: int = 8,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        attn_hidden: int = 128,
        attn_temperature: float = 1.0,
        cum_dropout: float = 0.0,
        return_attention: bool = False,
        pretrained_encoder: bool = True,
        maxtime: float = 10.0,
        time_num_steps: int = 200,
    ):
        super().__init__()
        self.num_years = int(num_years)
        self.dim = int(dim)
        self.return_attention = bool(return_attention)

        self.backbone = ImageBackbone(
            dim=self.dim,
            pretrained=pretrained_encoder,
            freeze_encoder=freeze_encoder,
        )
        self.view_pool = ViewAttentionPooling(
            dim=self.dim,
            hidden=attn_hidden,
            temperature=attn_temperature,
        )
        self.time_enc = ContinuousPosEncoding(
            dim=self.dim,
            drop=float(dropout),
            maxtime=float(maxtime),
            num_steps=int(time_num_steps),
        )
        self.token_attn = TemporalAttentionLayer(
            dim=self.dim,
            num_heads=heads,
            dropout=dropout,
        )
        self.feature_projection = nn.Linear(self.dim, self.dim)
        self.head_dropout = nn.Dropout(float(dropout)) if dropout and float(dropout) > 0 else nn.Identity()

        self.cum_fused = CumulativeProbabilityLayer(self.dim, horizons=self.num_years, dropout=float(cum_dropout))
        self.cum_cur = CumulativeProbabilityLayer(self.dim, horizons=self.num_years, dropout=float(cum_dropout))
        self.cum_pri = CumulativeProbabilityLayer(self.dim, horizons=self.num_years, dropout=float(cum_dropout))

    def forward(self, cur_imgs, pri_imgs, pri_years=None, pri_pad_mask=None, has_prior_views=None):
        """
        cur_imgs: [B,4,1,H,W]
        pri_imgs: [B,K,4,1,H,W]
        pri_years: [B,K]
        pri_pad_mask: [B,K] bool True=pad
        has_prior_views: [B,K,4] float/bool
        """
        assert cur_imgs.dim() == 5 and cur_imgs.size(1) == 4, f"cur_imgs must be [B,4,1,H,W], got {tuple(cur_imgs.shape)}"
        assert pri_imgs.dim() == 6 and pri_imgs.size(2) == 4, f"pri_imgs must be [B,K,4,1,H,W], got {tuple(pri_imgs.shape)}"

        B, _, _, H, W = cur_imgs.shape
        K = int(pri_imgs.size(1))
        device = cur_imgs.device

        if pri_pad_mask is None:
            pri_pad_mask = torch.zeros(B, K, dtype=torch.bool, device=device)
        else:
            pri_pad_mask = pri_pad_mask.to(device=device, dtype=torch.bool)
            assert pri_pad_mask.shape == (B, K), f"pri_pad_mask must be [B,K], got {tuple(pri_pad_mask.shape)}"

        if pri_years is None:
            pri_years = torch.zeros(B, K, dtype=torch.float32, device=device)
        else:
            pri_years = pri_years.to(device=device, dtype=torch.float32)
            assert pri_years.shape == (B, K), f"pri_years must be [B,K], got {tuple(pri_years.shape)}"

        if has_prior_views is None:
            has_prior_views = (~pri_pad_mask).unsqueeze(-1).expand(B, K, 4).float()
        else:
            has_prior_views = has_prior_views.to(device=device, dtype=torch.float32)
            assert has_prior_views.shape == (B, K, 4), f"has_prior_views must be [B,K,4], got {tuple(has_prior_views.shape)}"

        # Current token
        cur_vecs = self.backbone.encode_views(cur_imgs)          # [B,4,dim]
        cur_tok, w_cur = self.view_pool(cur_vecs, mask_keep=None)

        if K == 0:
            fused = self.feature_projection(cur_tok)
            fused_h = self.head_dropout(fused)
            zeros = torch.zeros_like(fused_h)
            return {
                "risk_prediction": {
                    "pred_fused": self.cum_fused(fused_h),
                    "pred_cur": self.cum_cur(fused_h),
                    "pred_pri": self.cum_pri(zeros),
                }
            }

        # Encode all prior visits
        x = pri_imgs.reshape(B * K, 4, 1, H, W)                   # [B*K,4,1,H,W]
        pri_vecs = self.backbone.encode_views(x)                  # [B*K,4,dim]
        pri_vecs = pri_vecs.reshape(B, K, 4, self.dim)            # [B,K,4,dim]

        view_keep = (has_prior_views > 0.5) & (~pri_pad_mask.unsqueeze(-1))   # [B,K,4]
        has_visit = (~pri_pad_mask) & (view_keep.sum(dim=2) > 0)               # [B,K]

        pri_flat = pri_vecs.reshape(B * K, 4, self.dim)           # [B*K,4,dim]
        keep_flat = view_keep.reshape(B * K, 4)                   # [B*K,4]

        pri_tok_flat, w_pri_flat = self.view_pool(pri_flat, mask_keep=keep_flat)
        pri_tok = pri_tok_flat.reshape(B, K, self.dim)            # [B,K,dim]

        cur_expand = cur_tok.unsqueeze(1).expand(B, K, self.dim)  # [B,K,dim]
        diff_tok = cur_expand - pri_tok                           # [B,K,dim]

        # Reverse visits so sequence matches:
        # [pri_K, diff_K, ..., pri_1, diff_1, cur]
        pri_tok = torch.flip(pri_tok, dims=[1])
        diff_tok = torch.flip(diff_tok, dims=[1])
        has_visit = torch.flip(has_visit, dims=[1])
        pri_years = torch.flip(pri_years, dims=[1])
        w_pri_all = torch.flip(w_pri_flat.reshape(B, K, 4), dims=[1])

        seq_2k = torch.stack([pri_tok, diff_tok], dim=2).reshape(B, 2 * K, self.dim)  # [B,2K,dim]
        seq = torch.cat([seq_2k, cur_tok.unsqueeze(1)], dim=1)                          # [B,2K+1,dim]

        # Time encoding
        t = pri_years.clamp(min=0.0)
        times_2k = torch.stack([t, t], dim=2).reshape(B, 2 * K)
        times = torch.cat([times_2k, torch.zeros(B, 1, device=device, dtype=torch.float32)], dim=1)  # [B,S]

        pe = self.time_enc.interp(times)         # [B,S,dim]
        seq = self.time_enc.dropout(seq + pe)    # [B,S,dim]

        # Token mask
        mk = (~has_visit)
        mask_2k = torch.stack([mk, mk], dim=2).reshape(B, 2 * K)
        token_mask = torch.cat(
            [mask_2k, torch.zeros(B, 1, device=device, dtype=torch.bool)],
            dim=1,
        )  # [B,S]

        # Transformer expects [S,B,dim]
        seq = seq.permute(1, 0, 2)  # [S,B,dim]

        if token_mask.any():
            seq = seq.masked_fill(token_mask.permute(1, 0).unsqueeze(-1), 0.0)

        seq_att = self.token_attn(seq, key_padding_mask=token_mask)

        keep = (~token_mask).to(dtype=seq_att.dtype).permute(1, 0).unsqueeze(-1)  # [S,B,1]
        fused = (seq_att * keep).sum(dim=0) / keep.sum(dim=0).clamp(min=1.0)      # [B,dim]
        fused = self.feature_projection(fused)

        pri_keep = has_visit.to(dtype=pri_tok.dtype)
        den = pri_keep.sum(dim=1, keepdim=True).clamp(min=1.0)
        pri_agg = (pri_tok * pri_keep.unsqueeze(-1)).sum(dim=1) / den

        fused_h = self.head_dropout(fused)
        cur_h = self.head_dropout(cur_tok)
        pri_h = self.head_dropout(pri_agg)

        out = {
            "risk_prediction": {
                "pred_fused": self.cum_fused(fused_h),
                "pred_cur": self.cum_cur(cur_h),
                "pred_pri": self.cum_pri(pri_h),
            }
        }

        if self.return_attention:
            out["attention"] = {
                "view_cur": w_cur,           # [B,4]
                "view_pri": w_pri_all,       # [B,K,4]
                "token_mask": token_mask,    # [B,S]
                "times": times,              # [B,S]
            }

        return out


class MultiPriorRisk(nn.Module):
    """
    Multi-prior temporal model with raw priors.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.risk_prediction_model = MultiPriorTemporalCore(**kwargs)

    def forward(self, cur_imgs, pri_imgs, pri_years=None, pri_pad_mask=None, has_prior_views=None):
        return self.risk_prediction_model(
            cur_imgs,
            pri_imgs,
            pri_years=pri_years,
            pri_pad_mask=pri_pad_mask,
            has_prior_views=has_prior_views,
        )


class MultiPriorRiskAligned(nn.Module):
    """
    Multi-prior temporal model with aligned priors.
    Architecture is identical to MultiPriorRisk.
    The only difference is that ali_imgs comes from the aligned dataset.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.risk_prediction_model = MultiPriorTemporalCore(**kwargs)

    def forward(self, cur_imgs, ali_imgs, pri_years=None, pri_pad_mask=None, has_prior_views=None):
        return self.risk_prediction_model(
            cur_imgs,
            ali_imgs,
            pri_years=pri_years,
            pri_pad_mask=pri_pad_mask,
            has_prior_views=has_prior_views,
        )


class MultiPriorRiskAlignedWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.risk_prediction_model = MultiPriorRiskAligned(**kwargs)

    def forward(self, cur_imgs, ali_imgs, pri_years=None, pri_pad_mask=None, has_prior_views=None):
        return self.risk_prediction_model(
            cur_imgs,
            ali_imgs,
            pri_years=pri_years,
            pri_pad_mask=pri_pad_mask,
            has_prior_views=has_prior_views,
        )

# Collate function for aligned multi-prior dataset
def collate_multiprior_aligned(batch):
    """
    Batch item format:
      (
        cur_imgs [4,1,H,W],
        ali_imgs [K,4,1,H,W],
        pri_years [K],
        has_prior_views [K,4],
        y [5],
        mask [5]
      )

    Returns:
      cur   [B,4,1,H,W]
      ali   [B,Kmax,4,1,H,W]
      years [B,Kmax]
      pad   [B,Kmax] bool
      hpv   [B,Kmax,4]
      y     [B,5]
      m     [B,5]
    """
    cur_list, ali_list, years_list, hpv_list, y_list, mask_list = zip(*batch)

    B = len(batch)
    cur = torch.stack(cur_list, dim=0)  # [B,4,1,H,W]

    Ks = [a.shape[0] for a in ali_list]
    Kmax = max(Ks) if Ks else 0

    if Kmax == 0:
        ali = torch.zeros((B, 0, 4, 1, cur.shape[-2], cur.shape[-1]), dtype=cur.dtype, device=cur.device)
        years = torch.zeros((B, 0), dtype=torch.float32, device=cur.device)
        pad = torch.zeros((B, 0), dtype=torch.bool, device=cur.device)
        hpv = torch.zeros((B, 0, 4), dtype=torch.float32, device=cur.device)
    else:
        ali = torch.zeros((B, Kmax, 4, 1, cur.shape[-2], cur.shape[-1]), dtype=cur.dtype, device=cur.device)
        years = torch.zeros((B, Kmax), dtype=torch.float32, device=cur.device)
        pad = torch.ones((B, Kmax), dtype=torch.bool, device=cur.device)
        hpv = torch.zeros((B, Kmax, 4), dtype=torch.float32, device=cur.device)

        for i in range(B):
            K = Ks[i]
            if K == 0:
                continue
            ali[i, :K] = ali_list[i]
            years[i, :K] = years_list[i].float()
            hpv[i, :K] = hpv_list[i].float()
            pad[i, :K] = False

    y = torch.stack(y_list, dim=0).float()
    m = torch.stack(mask_list, dim=0).float()

    return cur, ali, years, pad, hpv, y, m