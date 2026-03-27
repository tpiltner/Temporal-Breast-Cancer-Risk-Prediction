import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score

from datasetModel import TemporalMultiPriorAllAlignedDataset, RISK_COLS
from modelArchitecture import MultiPriorRiskAlignedWrapper  


EXP_ROOT = Path("/local/scratch/tpiltne/models/multiPriorAligned")

GRID_EPOCHS = 5
FINAL_EPOCHS = 15

BATCH_SIZE = 1
NUM_WORKERS = 2
PIN_MEMORY = True

PERSISTENT_WORKERS = False
PREFETCH_FACTOR = 2

SEED = 1337

USE_AUG_GRID = False
FINAL_USE_AUG = True

USE_AMP = True
CLIP_GRAD_NORM = 1.0
GRAD_ACCUM_STEPS = 4

EARLY_STOP_PATIENCE = 3

AUC3TO5_IDXS = [2, 3, 4]

POS_WEIGHT_SMOOTH = 5.0
POS_WEIGHT_CLAMP_MAX = 50.0

FREEZE_ENCODER_GRID = True
UNFREEZE_ENCODER_FINAL = True
ENCODER_LR_MULT_FINAL = 0.05

GRID_NUM_HEADS = [2, 4]
GRID_DROPOUT = [0.1, 0.2]
GRID_LR = [5e-5, 1e-4]
GRID_WD = [1e-4, 3e-4]


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Augmentations 
def _rand_uniform(a: float, b: float) -> float:
    return float(torch.empty(1).uniform_(a, b).item())


def random_gamma(x: torch.Tensor, gamma_range=(0.8, 1.2), p=0.5) -> torch.Tensor:
    if torch.rand(1).item() > p:
        return x
    g = _rand_uniform(*gamma_range)
    x = x.clamp(0, 1)
    return x.pow(g)


def random_brightness_contrast(
    x: torch.Tensor, b_range=(0.9, 1.1), c_range=(0.9, 1.1), p=0.5
) -> torch.Tensor:
    if torch.rand(1).item() > p:
        return x
    b = _rand_uniform(*b_range)
    c = _rand_uniform(*c_range)
    mean = x.mean(dim=(-2, -1), keepdim=True)
    x = (x - mean) * c + mean
    x = x * b
    return x.clamp(0, 1)


def random_crop_resize(x: torch.Tensor, scale=(0.85, 1.0), p=0.5) -> torch.Tensor:
    if torch.rand(1).item() > p:
        return x
    N, C, H, W = x.shape
    s = _rand_uniform(*scale)
    new_h = max(8, int(H * s))
    new_w = max(8, int(W * s))
    top = 0 if H == new_h else int(torch.randint(0, H - new_h + 1, (1,)).item())
    left = 0 if W == new_w else int(torch.randint(0, W - new_w + 1, (1,)).item())
    crop = x[:, :, top : top + new_h, left : left + new_w]
    return F.interpolate(crop, size=(H, W), mode="bilinear", align_corners=False)


def random_small_translate(x: torch.Tensor, max_translate=0.02, p=0.5) -> torch.Tensor:
    if torch.rand(1).item() > p:
        return x
    N, C, H, W = x.shape
    tx = _rand_uniform(-max_translate, max_translate) * 2.0
    ty = _rand_uniform(-max_translate, max_translate) * 2.0
    theta = (
        torch.tensor([[1.0, 0.0, tx], [0.0, 1.0, ty]], device=x.device, dtype=x.dtype)
        .unsqueeze(0)
        .repeat(N, 1, 1)
    )
    grid = F.affine_grid(theta, size=x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def apply_train_augs(cur_views: torch.Tensor) -> torch.Tensor:
    """
    cur_views: [B,4,1,H,W]
    """
    B, V, C, H, W = cur_views.shape
    x = cur_views.reshape(B * V, C, H, W)
    x = random_crop_resize(x, p=0.5)
    x = random_small_translate(x, p=0.5)
    x = random_brightness_contrast(x, p=0.5)
    x = random_gamma(x, p=0.5)
    return x.view(B, V, C, H, W)


# Collate: pad variable K aligned priors + has_prior_views
def collate_multiprior_aligned(batch):
    # item:
    # (cur [4,1,H,W], ali [K,4,1,H,W], years [K], hpv [K,4], y [5], m [5])
    cur_list, ali_list, years_list, hpv_list, y_list, m_list = zip(*batch)
    B = len(batch)

    cur = torch.stack(cur_list, dim=0)  # [B,4,1,H,W]
    y = torch.stack(y_list, dim=0).float()
    m = torch.stack(m_list, dim=0).float()

    Ks = [int(a.shape[0]) for a in ali_list]
    Kmax = max(Ks) if Ks else 0

    if Kmax == 0:
        ali = torch.zeros((B, 0, 4, 1, cur.shape[-2], cur.shape[-1]), dtype=cur.dtype)
        years = torch.zeros((B, 0), dtype=torch.float32)
        pad = torch.zeros((B, 0), dtype=torch.bool)
        hpv = torch.zeros((B, 0, 4), dtype=torch.float32)
        return cur, ali, years, pad, hpv, y, m

    ali = torch.zeros((B, Kmax, 4, 1, cur.shape[-2], cur.shape[-1]), dtype=cur.dtype)
    years = torch.zeros((B, Kmax), dtype=torch.float32)
    pad = torch.ones((B, Kmax), dtype=torch.bool)
    hpv = torch.zeros((B, Kmax, 4), dtype=torch.float32)

    for i in range(B):
        K = Ks[i]
        if K == 0:
            continue
        ali[i, :K] = ali_list[i]
        years[i, :K] = years_list[i].float()
        hpv[i, :K] = hpv_list[i].float()
        pad[i, :K] = False

    return cur, ali, years, pad, hpv, y, m

# Pos-weight computation (observed-only)
@torch.no_grad()
def compute_pos_weight_from_train(
    train_ds,
    smooth: float = POS_WEIGHT_SMOOTH,
    clamp_max: float = POS_WEIGHT_CLAMP_MAX,
) -> torch.Tensor:
    T = len(RISK_COLS)
    tot_pos = torch.zeros(T, dtype=torch.float64)
    tot_neg = torch.zeros(T, dtype=torch.float64)

    if hasattr(train_ds, "exam_groups") and isinstance(train_ds.exam_groups, list) and len(train_ds.exam_groups) > 0:
        for exam_id, df_exam, y_event, mask, hw in train_ds.exam_groups:
            y = torch.as_tensor(np.asarray(y_event), dtype=torch.float64)
            mm = torch.as_tensor(np.asarray(mask), dtype=torch.float64)
            tot_pos += (y * mm)
            tot_neg += ((1.0 - y) * mm)
    else:
        tmp_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_multiprior_aligned)
        for cur, ali, years, pad, hpv, y, mm in tmp_loader:
            y = y.to(torch.float64)
            mm = mm.to(torch.float64)
            tot_pos += (y * mm).sum(dim=0)
            tot_neg += ((1.0 - y) * mm).sum(dim=0)

    pw = (tot_neg + float(smooth)) / (tot_pos + float(smooth))
    pw = torch.clamp(pw, max=float(clamp_max)).to(torch.float32)
    return pw

# Loss: masked BCE + pos_weight
def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    loss = loss * mask
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom

# Metrics
@torch.no_grad()
def compute_auc_auprc(probs: np.ndarray, y: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    T = probs.shape[1]
    out: Dict[str, float] = {}
    for t in range(T):
        idx = mask[:, t] > 0.5
        if int(idx.sum()) < 5:
            out[f"auc_{t}"] = float("nan")
            out[f"auprc_{t}"] = float("nan")
            continue

        yt = y[idx, t]
        pt = probs[idx, t]

        if float(yt.max()) == float(yt.min()):
            out[f"auc_{t}"] = float("nan")
        else:
            out[f"auc_{t}"] = float(roc_auc_score(yt, pt))

        out[f"auprc_{t}"] = float(average_precision_score(yt, pt))

    out["mean_auc"] = float(np.nanmean([out[f"auc_{t}"] for t in range(T)]))
    out["mean_auprc"] = float(np.nanmean([out[f"auprc_{t}"] for t in range(T)]))
    out["mean_auc_3to5"] = float(np.nanmean([out.get(f"auc_{t}", float("nan")) for t in AUC3TO5_IDXS]))
    out["mean_auprc_3to5"] = float(np.nanmean([out.get(f"auprc_{t}", float("nan")) for t in AUC3TO5_IDXS]))
    return {k: float(v) for k, v in out.items()}


def set_encoder_trainable(model: nn.Module, trainable: bool):
    if not hasattr(model, "encoder"):
        return
    for p in model.encoder.parameters():
        p.requires_grad = bool(trainable)


def enforce_encoder_bn_eval(model: nn.Module):
    enc = getattr(model, "encoder", None)
    if enc is None:
        for name, m in model.named_modules():
            if name.endswith("encoder"):
                enc = m
                break
    if enc is None:
        return
    if hasattr(enc, "freeze_bn"):
        enc.freeze_bn()
    else:
        for m in enc.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

# Evaluation
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_weight: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    model.eval()
    all_probs, all_y, all_m = [], [], []
    total_loss = 0.0
    n_batches = 0

    for cur, ali, years, pad, hpv, y, mm in loader:
        cur = cur.to(device, dtype=torch.float32)
        ali = ali.to(device, dtype=torch.float32)
        years = years.to(device, dtype=torch.float32)
        pad = pad.to(device)
        hpv = hpv.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        mm = mm.to(device, dtype=torch.float32)

        out = model(cur, ali, pri_years=years, pri_pad_mask=pad, has_prior_views=hpv)["risk_prediction"]
        logits = out["pred_fused"]

        loss = masked_bce_with_logits(logits, y, mm, pos_weight=pos_weight)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_y.append(y.detach().cpu().numpy())
        all_m.append(mm.detach().cpu().numpy())

        total_loss += float(loss.item())
        n_batches += 1

    probs = np.concatenate(all_probs, axis=0) if len(all_probs) else np.zeros((0, len(RISK_COLS)))
    y = np.concatenate(all_y, axis=0) if len(all_y) else np.zeros((0, len(RISK_COLS)))
    mm = np.concatenate(all_m, axis=0) if len(all_m) else np.zeros((0, len(RISK_COLS)))

    metrics = compute_auc_auprc(probs, y, mm)
    metrics["val_loss"] = float(total_loss / max(1, n_batches))
    return {k: float(v) for k, v in metrics.items()}

# Optimizer builder
def build_optimizer(
    model: nn.Module,
    head_lr: float,
    wd: float,
    unfreeze_encoder: bool,
    encoder_lr_mult: float,
) -> Tuple[torch.optim.Optimizer, List[nn.Parameter]]:
    enc_params: List[nn.Parameter] = []
    head_params: List[nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "encoder" in name:
            enc_params.append(p)
        else:
            head_params.append(p)

    if (not unfreeze_encoder) or (len(enc_params) == 0):
        opt = torch.optim.AdamW(
            [{"params": head_params, "lr": float(head_lr)}],
            weight_decay=float(wd),
        )
        return opt, head_params

    opt = torch.optim.AdamW(
        [
            {"params": head_params, "lr": float(head_lr)},
            {"params": enc_params, "lr": float(head_lr) * float(encoder_lr_mult)},
        ],
        weight_decay=float(wd),
    )
    return opt, (head_params + enc_params)

# Grid 
def build_grid() -> List[Dict[str, Any]]:
    grid: List[Dict[str, Any]] = []
    for nh in GRID_NUM_HEADS:
        for do in GRID_DROPOUT:
            for lr in GRID_LR:
                for wd in GRID_WD:
                    grid.append({"num_heads": int(nh), "dropout": float(do), "lr": float(lr), "wd": float(wd)})
    return grid


def _dl_kwargs() -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = bool(PERSISTENT_WORKERS)
        kwargs["prefetch_factor"] = int(PREFETCH_FACTOR)
    return kwargs


def write_table_csv(rows: List[Dict[str, Any]], path: Path):
    cols = [
        "#", "heads", "dropout", "lr", "wd",
        "val_mean_auc_3to5", "val_mean_auprc_3to5",
        "val_mean_auc_1to5", "val_mean_auprc_1to5",
        "best_epoch", "run_dir",
    ]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(
            ",".join(
                [
                    str(r["idx"]),
                    str(r["num_heads"]),
                    f"{r['dropout']:.6g}",
                    f"{r['lr']:.6g}",
                    f"{r['wd']:.6g}",
                    f"{r.get('best_val_mean_auc_3to5', float('nan')):.6g}",
                    f"{r.get('best_val_mean_auprc_3to5', float('nan')):.6g}",
                    f"{r.get('best_val_mean_auc', float('nan')):.6g}",
                    f"{r.get('best_val_mean_auprc', float('nan')):.6g}",
                    str(r.get("best_epoch", "")),
                    str(r.get("run_dir", "")),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n")

# Resumability helpers
def _safe_load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def run_is_complete(run_dir: Path, required_epochs: int) -> bool:
    hist_path = run_dir / "history.json"
    if not hist_path.exists():
        return False
    hist = _safe_load_json(hist_path, default=None)
    return isinstance(hist, list) and len(hist) >= required_epochs


def _get_selection_scores(row: Dict[str, Any]) -> Tuple[float, float]:
    return float(row.get("mean_auc_3to5", float("-inf"))), float(row.get("mean_auprc_3to5", float("-inf")))


def summarize_completed_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    cfg_path = run_dir / "run_config.json"
    hist_path = run_dir / "history.json"
    if not (cfg_path.exists() and hist_path.exists()):
        return None

    cfg = _safe_load_json(cfg_path, default=None)
    hist = _safe_load_json(hist_path, default=None)
    if not isinstance(cfg, dict) or not isinstance(hist, list) or len(hist) == 0:
        return None

    best_epoch = -1
    best_auc = -1e9
    best_auprc = -1e9

    for i, row in enumerate(hist):
        auc, auprc = _get_selection_scores(row)
        if auc > best_auc + 1e-9 or (abs(auc - best_auc) <= 1e-9 and auprc > best_auprc + 1e-12):
            best_auc = auc
            best_auprc = auprc
            best_epoch = i

    out = {
        "num_heads": int(cfg.get("num_heads")),
        "dropout": float(cfg.get("dropout")),
        "lr": float(cfg.get("lr")),
        "wd": float(cfg.get("wd")),
        "best_epoch": int(best_epoch),
        "best_val_mean_auc_3to5": float(best_auc),
        "best_val_mean_auprc_3to5": float(best_auprc),
        "best_val_mean_auc": float("nan"),
        "best_val_mean_auprc": float("nan"),
        "best_val_loss": float("nan"),
        "run_dir": str(run_dir),
    }

    if best_epoch >= 0:
        best_row = hist[best_epoch]
        out["best_val_mean_auc"] = float(best_row.get("mean_auc", float("nan")))
        out["best_val_mean_auprc"] = float(best_row.get("mean_auprc", float("nan")))
        out["best_val_loss"] = float(best_row.get("val_loss", float("nan")))

    return out


def rebuild_table_and_best(exp_dir: Path, grid: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    runs_dir = exp_dir / "grid_runs"
    table_rows: List[Dict[str, Any]] = []
    best_overall: Optional[Dict[str, Any]] = None

    for idx, cfg in enumerate(grid, start=1):
        run_name = f"run_{idx:03d}_H{cfg['num_heads']}_DO{cfg['dropout']}_LR{cfg['lr']}_WD{cfg['wd']}"
        run_dir = runs_dir / run_name

        if run_is_complete(run_dir, GRID_EPOCHS):
            summary = summarize_completed_run(run_dir)
            if summary is None:
                continue

            row = dict(cfg)
            row["idx"] = idx
            row["run_dir"] = str(run_dir)
            row["best_epoch"] = summary["best_epoch"]
            row["best_val_mean_auc_3to5"] = summary["best_val_mean_auc_3to5"]
            row["best_val_mean_auprc_3to5"] = summary["best_val_mean_auprc_3to5"]
            row["best_val_mean_auc"] = summary["best_val_mean_auc"]
            row["best_val_mean_auprc"] = summary["best_val_mean_auprc"]
            table_rows.append(row)

            if best_overall is None:
                best_overall = summary
            else:
                if summary["best_val_mean_auc_3to5"] > best_overall["best_val_mean_auc_3to5"] + 1e-9:
                    best_overall = summary
                elif abs(summary["best_val_mean_auc_3to5"] - best_overall["best_val_mean_auc_3to5"]) <= 1e-9:
                    if summary["best_val_mean_auprc_3to5"] > best_overall["best_val_mean_auprc_3to5"] + 1e-12:
                        best_overall = summary

    table_rows.sort(key=lambda r: r["idx"])
    return table_rows, best_overall


# =========================
# One grid run
# =========================
def train_one_config_grid(cfg: Dict[str, Any], run_dir: Path, device: torch.device) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = dict(cfg)
    run_cfg.update(
        {
            "run_dir": str(run_dir),
            "mode": "grid",
            "grid_epochs": GRID_EPOCHS,
            "use_aug": USE_AUG_GRID,
            "batch_size": BATCH_SIZE,
            "use_amp": USE_AMP,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "clip_grad_norm": CLIP_GRAD_NORM,
            "selection_metric": "mean_auc_3to5 (tie: mean_auprc_3to5)",
            "selection_horizon_indices": AUC3TO5_IDXS,
            "pos_weight_formula": "(tot_neg[h]+5)/(tot_pos[h]+5), clamp<=50 (observed-only)",
            "freeze_encoder_grid": bool(FREEZE_ENCODER_GRID),
            "model": "MultiPriorRiskAlignedWrapper",
            "dataset": "TemporalMultiPriorAllAlignedDataset",
            "diff_mode": "abs",
        }
    )
    (run_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))

    train_ds = TemporalMultiPriorAllAlignedDataset(split="train")
    val_ds = TemporalMultiPriorAllAlignedDataset(split="val")

    pos_weight = compute_pos_weight_from_train(train_ds).to(device)
    print(f"[INFO] pos_weight = {[round(x,4) for x in pos_weight.detach().cpu().tolist()]}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        collate_fn=collate_multiprior_aligned,
        **_dl_kwargs(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        collate_fn=collate_multiprior_aligned,
        **_dl_kwargs(),
    )

    model = MultiPriorRiskAlignedWrapper(
        num_years=len(RISK_COLS),
        dim=512,
        heads=int(cfg["num_heads"]),
        dropout=float(cfg["dropout"]),
        freeze_encoder=bool(FREEZE_ENCODER_GRID),
        diff_mode="abs",
    ).to(device)

    set_encoder_trainable(model, trainable=not bool(FREEZE_ENCODER_GRID))

    optimizer, params_for_clip = build_optimizer(
        model,
        head_lr=float(cfg["lr"]),
        wd=float(cfg["wd"]),
        unfreeze_encoder=False,
        encoder_lr_mult=ENCODER_LR_MULT_FINAL,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_score_auc = -1e9
    best_score_auprc = -1e9
    best_epoch = -1
    history: List[Dict[str, Any]] = []

    for epoch in range(GRID_EPOCHS):
        model.train()
        enforce_encoder_bn_eval(model)

        running_loss = 0.0
        n_batches = 0

        optimizer.zero_grad(set_to_none=True)
        accum = 0

        for step, (cur, ali, years, pad, hpv, y, mm) in enumerate(train_loader):
            cur = cur.to(device, dtype=torch.float32)
            ali = ali.to(device, dtype=torch.float32)
            years = years.to(device, dtype=torch.float32)
            pad = pad.to(device)
            hpv = hpv.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            mm = mm.to(device, dtype=torch.float32)

            if USE_AUG_GRID:
                cur = apply_train_augs(cur)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                out = model(cur, ali, pri_years=years, pri_pad_mask=pad, has_prior_views=hpv)["risk_prediction"]
                logits = out["pred_fused"]
                loss = masked_bce_with_logits(logits, y, mm, pos_weight=pos_weight)
                loss = loss / float(GRAD_ACCUM_STEPS)

            scaler.scale(loss).backward()
            accum += 1

            if accum == GRAD_ACCUM_STEPS:
                if CLIP_GRAD_NORM and CLIP_GRAD_NORM > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=float(CLIP_GRAD_NORM))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum = 0

            running_loss += float(loss.item()) * float(GRAD_ACCUM_STEPS)
            n_batches += 1

        if accum > 0:
            if CLIP_GRAD_NORM and CLIP_GRAD_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=float(CLIP_GRAD_NORM))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss = float(running_loss / max(1, n_batches))

        val_metrics = evaluate(model, val_loader, device, pos_weight=pos_weight)
        sel_auc = float(val_metrics.get("mean_auc_3to5", float("nan")))
        sel_auprc = float(val_metrics.get("mean_auprc_3to5", float("nan")))

        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "mode": "grid"}, run_dir / "last.pt")

        improved = (sel_auc > best_score_auc + 1e-9) or (
            abs(sel_auc - best_score_auc) <= 1e-9 and sel_auprc > best_score_auprc + 1e-12
        )
        if improved:
            best_score_auc = sel_auc
            best_score_auprc = sel_auprc
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "mode": "grid"}, run_dir / "best.pt")

        row = {"epoch": int(epoch), "train_loss": float(train_loss), **{k: float(v) for k, v in val_metrics.items()}}
        history.append(row)

        print(
            f"[GRID {run_dir.name}] epoch {epoch:02d} | "
            f"train_loss {train_loss:.4f} | sel_auc_3to5 {sel_auc:.4f} | sel_auprc_3to5 {sel_auprc:.4f} | "
            f"val_mean_auc {float(val_metrics['mean_auc']):.4f} | val_mean_auprc {float(val_metrics['mean_auprc']):.4f}"
        )

    (run_dir / "history.json").write_text(json.dumps(history))

    summary = dict(cfg)
    summary.update(
        {
            "best_epoch": int(best_epoch),
            "best_val_mean_auc_3to5": float(best_score_auc),
            "best_val_mean_auprc_3to5": float(best_score_auprc),
            "best_val_mean_auc": float("nan"),
            "best_val_mean_auprc": float("nan"),
            "best_val_loss": float("nan"),
            "run_dir": str(run_dir),
        }
    )
    if best_epoch >= 0:
        best_row = history[best_epoch]
        summary["best_val_mean_auc"] = float(best_row.get("mean_auc", float("nan")))
        summary["best_val_mean_auprc"] = float(best_row.get("mean_auprc", float("nan")))
        summary["best_val_loss"] = float(best_row.get("val_loss", float("nan")))

    return summary

# Final training
def train_final(best_cfg: Dict[str, Any], final_dir: Path, device: torch.device) -> Dict[str, Any]:
    final_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = dict(best_cfg)
    run_cfg.update(
        {
            "run_dir": str(final_dir),
            "mode": "final",
            "final_epochs": FINAL_EPOCHS,
            "use_aug": FINAL_USE_AUG,
            "batch_size": BATCH_SIZE,
            "use_amp": USE_AMP,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "clip_grad_norm": CLIP_GRAD_NORM,
            "selection_metric": "mean_auc_3to5 (tie: mean_auprc_3to5)",
            "selection_horizon_indices": AUC3TO5_IDXS,
            "pos_weight_formula": "(tot_neg[h]+5)/(tot_pos[h]+5), clamp<=50 (observed-only)",
            "unfreeze_encoder_final": bool(UNFREEZE_ENCODER_FINAL),
            "encoder_lr_mult_final": float(ENCODER_LR_MULT_FINAL),
            "early_stop_patience": int(EARLY_STOP_PATIENCE),
            "model": "MultiPriorRiskAlignedWrapper",
            "dataset": "TemporalMultiPriorAllAlignedDataset",
            "diff_mode": "abs",
        }
    )
    (final_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))

    train_ds = TemporalMultiPriorAllAlignedDataset(split="train")
    val_ds = TemporalMultiPriorAllAlignedDataset(split="val")

    pos_weight = compute_pos_weight_from_train(train_ds).to(device)
    print(f"[INFO] pos_weight = {[round(x,4) for x in pos_weight.detach().cpu().tolist()]}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        collate_fn=collate_multiprior_aligned,
        **_dl_kwargs(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        collate_fn=collate_multiprior_aligned,
        **_dl_kwargs(),
    )

    model = MultiPriorRiskAlignedWrapper(
        num_years=len(RISK_COLS),
        dim=512,
        heads=int(best_cfg["num_heads"]),
        dropout=float(best_cfg["dropout"]),
        freeze_encoder=not bool(UNFREEZE_ENCODER_FINAL),
        diff_mode="abs",
    ).to(device)

    set_encoder_trainable(model, trainable=bool(UNFREEZE_ENCODER_FINAL))

    n_total = sum(1 for _ in model.parameters())
    n_train = sum(1 for p in model.parameters() if p.requires_grad)
    n_enc_train = sum(1 for n, p in model.named_parameters() if ("encoder" in n and p.requires_grad))
    print(f"[DEBUG] params: trainable {n_train}/{n_total} | encoder_trainable={n_enc_train}")

    optimizer, params_for_clip = build_optimizer(
        model,
        head_lr=float(best_cfg["lr"]),
        wd=float(best_cfg["wd"]),
        unfreeze_encoder=bool(UNFREEZE_ENCODER_FINAL),
        encoder_lr_mult=float(ENCODER_LR_MULT_FINAL),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_score_auc = -1e9
    best_score_auprc = -1e9
    best_epoch = -1
    history: List[Dict[str, Any]] = []
    no_improve = 0

    for epoch in range(FINAL_EPOCHS):
        model.train()
        enforce_encoder_bn_eval(model)

        running_loss = 0.0
        n_batches = 0

        optimizer.zero_grad(set_to_none=True)
        accum = 0

        for step, (cur, ali, years, pad, hpv, y, mm) in enumerate(train_loader):
            cur = cur.to(device, dtype=torch.float32)
            ali = ali.to(device, dtype=torch.float32)
            years = years.to(device, dtype=torch.float32)
            pad = pad.to(device)
            hpv = hpv.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            mm = mm.to(device, dtype=torch.float32)

            if FINAL_USE_AUG:
                cur = apply_train_augs(cur)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                out = model(cur, ali, pri_years=years, pri_pad_mask=pad, has_prior_views=hpv)["risk_prediction"]
                logits = out["pred_fused"]
                loss = masked_bce_with_logits(logits, y, mm, pos_weight=pos_weight)
                loss = loss / float(GRAD_ACCUM_STEPS)

            scaler.scale(loss).backward()
            accum += 1

            if accum == GRAD_ACCUM_STEPS:
                if CLIP_GRAD_NORM and CLIP_GRAD_NORM > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=float(CLIP_GRAD_NORM))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum = 0

            running_loss += float(loss.item()) * float(GRAD_ACCUM_STEPS)
            n_batches += 1

        if accum > 0:
            if CLIP_GRAD_NORM and CLIP_GRAD_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=float(CLIP_GRAD_NORM))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss = float(running_loss / max(1, n_batches))

        val_metrics = evaluate(model, val_loader, device, pos_weight=pos_weight)
        sel_auc = float(val_metrics.get("mean_auc_3to5", float("nan")))
        sel_auprc = float(val_metrics.get("mean_auprc_3to5", float("nan")))

        torch.save({"model": model.state_dict(), "cfg": best_cfg, "epoch": epoch, "mode": "final"}, final_dir / "last_final.pt")

        improved = (sel_auc > best_score_auc + 1e-9) or (
            abs(sel_auc - best_score_auc) <= 1e-9 and sel_auprc > best_score_auprc + 1e-12
        )
        if improved:
            best_score_auc = sel_auc
            best_score_auprc = sel_auprc
            best_epoch = epoch
            no_improve = 0
            torch.save({"model": model.state_dict(), "cfg": best_cfg, "epoch": epoch, "mode": "final"}, final_dir / "best_final.pt")
        else:
            no_improve += 1

        row = {"epoch": int(epoch), "train_loss": float(train_loss), **{k: float(v) for k, v in val_metrics.items()}}
        history.append(row)

        print(
            f"[FINAL] epoch {epoch:02d} | train_loss {train_loss:.4f} | sel_auc_3to5 {sel_auc:.4f} | sel_auprc_3to5 {sel_auprc:.4f} | "
            f"val_mean_auc {float(val_metrics['mean_auc']):.4f} | val_mean_auprc {float(val_metrics['mean_auprc']):.4f} | "
            f"no_improve {no_improve}/{EARLY_STOP_PATIENCE}"
        )

        if no_improve >= EARLY_STOP_PATIENCE:
            print("[EARLY STOP] No improvement on selection metric.")
            break

    (final_dir / "history.json").write_text(json.dumps(history))

    summary = dict(best_cfg)
    summary.update(
        {
            "best_epoch": int(best_epoch),
            "best_val_mean_auc_3to5": float(best_score_auc),
            "best_val_mean_auprc_3to5": float(best_score_auprc),
            "best_val_mean_auc": float("nan"),
            "best_val_mean_auprc": float("nan"),
            "best_val_loss": float("nan"),
            "run_dir": str(final_dir),
        }
    )

    if best_epoch >= 0:
        best_row = history[best_epoch]
        summary["best_val_mean_auc"] = float(best_row.get("mean_auc", float("nan")))
        summary["best_val_mean_auprc"] = float(best_row.get("mean_auprc", float("nan")))
        summary["best_val_loss"] = float(best_row.get("val_loss", float("nan")))

        for i, name in enumerate(RISK_COLS):
            summary[f"auc_{i}"] = float(best_row.get(f"auc_{i}", float("nan")))
            summary[f"auprc_{i}"] = float(best_row.get(f"auprc_{i}", float("nan")))
            summary[f"auc_{name}"] = float(best_row.get(f"auc_{i}", float("nan")))
            summary[f"auprc_{name}"] = float(best_row.get(f"auprc_{i}", float("nan")))

    (final_dir / "final_summary.json").write_text(json.dumps(summary, indent=2))
    return summary

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default=None, help="Resume in this dir if provided.")
    parser.add_argument("--diff_mode", type=str, default="abs", choices=["abs", "signed"], help="Diff mode for aligned model.")
    args = parser.parse_args()

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.exp_dir is not None:
        exp_dir = Path(args.exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        stamp = (
            exp_dir.name.split("multiPriorAligned_grid_then_final_")[-1]
            if "multiPriorAligned_grid_then_final_" in exp_dir.name
            else datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = EXP_ROOT / f"multiPriorAligned_grid_then_final_{stamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = exp_dir / "grid_runs"
    final_dir = exp_dir / "final_best"
    runs_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    grid = build_grid()

    exp_cfg_path = exp_dir / "exp_config.json"
    if not exp_cfg_path.exists():
        exp_cfg = {
            "EXP_ROOT": str(EXP_ROOT),
            "exp_dir": str(exp_dir),
            "timestamp": stamp,
            "grid_epochs": GRID_EPOCHS,
            "final_epochs": FINAL_EPOCHS,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "pin_memory": PIN_MEMORY,
            "persistent_workers": PERSISTENT_WORKERS,
            "prefetch_factor": PREFETCH_FACTOR,
            "use_aug_grid": USE_AUG_GRID,
            "final_use_aug": FINAL_USE_AUG,
            "use_amp": USE_AMP,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "clip_grad_norm": CLIP_GRAD_NORM,
            "selection_metric": "mean_auc_3to5 (tie: mean_auprc_3to5)",
            "selection_horizon_indices": AUC3TO5_IDXS,
            "pos_weight_formula": "(tot_neg[h]+5)/(tot_pos[h]+5), clamp<=50 (observed-only)",
            "encoder_policy": {
                "freeze_encoder_grid": FREEZE_ENCODER_GRID,
                "unfreeze_encoder_final": UNFREEZE_ENCODER_FINAL,
                "encoder_lr_mult_final": ENCODER_LR_MULT_FINAL,
            },
            "final_early_stop_patience": EARLY_STOP_PATIENCE,
            "grid": {"num_heads": GRID_NUM_HEADS, "dropout": GRID_DROPOUT, "lr": GRID_LR, "wd": GRID_WD},
            "model": "MultiPriorRiskAlignedWrapper",
            "dataset": "TemporalMultiPriorAllAlignedDataset",
            "diff_mode": args.diff_mode,
        }
        exp_cfg_path.write_text(json.dumps(exp_cfg, indent=2))

    print(f"[INFO] Saving to: {exp_dir}")
    print(f"[INFO] Total grid configs: {len(grid)}")
    print(f"[INFO] Grid epochs={GRID_EPOCHS} | Final epochs={FINAL_EPOCHS}")

    table_rows, best_overall = rebuild_table_and_best(exp_dir, grid)
    write_table_csv(table_rows, exp_dir / "grid_table.csv")

    if best_overall is not None:
        (exp_dir / "best_config.json").write_text(json.dumps(best_overall, indent=2))
        print(f"[INFO] Rebuilt from disk: completed_runs={len(table_rows)}/{len(grid)}")
        print(
            f"[INFO] Current best sel_auc_3to5={best_overall['best_val_mean_auc_3to5']:.4f} "
            f"sel_auprc_3to5={best_overall['best_val_mean_auprc_3to5']:.4f} in {best_overall['run_dir']}"
        )
    else:
        print("[INFO] No completed runs found yet (starting fresh).")

    # GRID loop
    for idx, cfg in enumerate(grid, start=1):
        run_name = f"run_{idx:03d}_H{cfg['num_heads']}_DO{cfg['dropout']}_LR{cfg['lr']}_WD{cfg['wd']}"
        run_dir = runs_dir / run_name

        if run_is_complete(run_dir, GRID_EPOCHS):
            print(f"[SKIP] [GRID {idx}/{len(grid)}] {run_name} already complete.")
            continue

        print(f"\n========== [GRID {idx}/{len(grid)}] {run_name} ==========")
        try:
            summary = train_one_config_grid(cfg=cfg, run_dir=run_dir, device=device)
        except torch.cuda.OutOfMemoryError as e:
            print(f"[OOM] {run_name}: {e}")
            (run_dir / "FAILED_OOM.txt").write_text(str(e) + "\n")
            torch.cuda.empty_cache()
            continue

        row = dict(cfg)
        row["idx"] = idx
        row["run_dir"] = str(run_dir)
        row["best_epoch"] = summary["best_epoch"]
        row["best_val_mean_auc_3to5"] = summary["best_val_mean_auc_3to5"]
        row["best_val_mean_auprc_3to5"] = summary["best_val_mean_auprc_3to5"]
        row["best_val_mean_auc"] = summary.get("best_val_mean_auc", float("nan"))
        row["best_val_mean_auprc"] = summary.get("best_val_mean_auprc", float("nan"))

        table_rows = [r for r in table_rows if r["idx"] != idx]
        table_rows.append(row)
        table_rows.sort(key=lambda r: r["idx"])
        write_table_csv(table_rows, exp_dir / "grid_table.csv")

        if best_overall is None:
            best_overall = summary
        else:
            if summary["best_val_mean_auc_3to5"] > best_overall["best_val_mean_auc_3to5"] + 1e-9:
                best_overall = summary
            elif abs(summary["best_val_mean_auc_3to5"] - best_overall["best_val_mean_auc_3to5"]) <= 1e-9:
                if summary["best_val_mean_auprc_3to5"] > best_overall["best_val_mean_auprc_3to5"] + 1e-12:
                    best_overall = summary

        (exp_dir / "best_config.json").write_text(json.dumps(best_overall, indent=2))

    assert best_overall is not None, "Grid produced no results."

    print("\n[GRID DONE]")
    print(f"[BEST GRID] run_dir={best_overall['run_dir']}")
    print(f"           heads={best_overall['num_heads']} dropout={best_overall['dropout']}")
    print(f"           lr={best_overall['lr']} wd={best_overall['wd']}")
    print(
        f"           sel_auc_3to5={best_overall['best_val_mean_auc_3to5']:.4f} "
        f"sel_auprc_3to5={best_overall['best_val_mean_auprc_3to5']:.4f}"
    )

    final_summary_path = final_dir / "final_summary.json"
    if final_summary_path.exists():
        print(f"\n[SKIP] Final already completed: {final_summary_path}")
        final_summary = _safe_load_json(final_summary_path, default=None)
        if isinstance(final_summary, dict):
            (exp_dir / "final_best_summary.json").write_text(json.dumps(final_summary, indent=2))
        return

    best_cfg = {
        "num_heads": best_overall["num_heads"],
        "dropout": best_overall["dropout"],
        "lr": best_overall["lr"],
        "wd": best_overall["wd"],
    }

    print("\n========== [FINAL TRAIN] best hyperparams ==========")
    final_summary = train_final(best_cfg=best_cfg, final_dir=final_dir, device=device)

    print("\n[FINAL DONE]")
    print(f"[FINAL BEST] dir={final_summary['run_dir']}")
    print(f"            best_epoch={final_summary['best_epoch']}")
    print(
        f"            sel_auc_3to5={final_summary['best_val_mean_auc_3to5']:.4f} "
        f"sel_auprc_3to5={final_summary['best_val_mean_auprc_3to5']:.4f}"
    )
    print(
        f"            val_mean_auc_1to5={final_summary.get('best_val_mean_auc', float('nan')):.4f} "
        f"val_mean_auprc_1to5={final_summary.get('best_val_mean_auprc', float('nan')):.4f}"
    )

    write_table_csv(table_rows, exp_dir / "grid_table.csv")
    (exp_dir / "best_config.json").write_text(json.dumps(best_overall, indent=2))
    (exp_dir / "final_best_summary.json").write_text(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
