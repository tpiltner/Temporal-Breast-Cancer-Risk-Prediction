"""

Use MammoRegNet to temporally align arbitrary many prior exams
to a current exam for EMBED.

"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import math
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

# Set up MammoregNet
# add their repo to path
sys.path.append(os.path.abspath("Longitudinal_Mammogram_Alignment-main"))
from src.models.MammoRegNet import MammoRegNet  # type: ignore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# path to the trained registration checkpoint
CKPT = "experiments/mammoregnet/checkpoints/best.pth"


def load_mammoregnet():
    net = MammoRegNet()
    if os.path.isfile(CKPT):
        print(f"Loading MammoRegNet weights from {CKPT}")
        sd = torch.load(CKPT, map_location=DEVICE)
        sd = sd.get("state_dict", sd)
        net.load_state_dict(sd, strict=False)
    else:
        print(f"WARNING: checkpoint {CKPT} not found – using random weights.")
    net.eval().to(DEVICE)
    return net


REG = load_mammoregnet()

# 1. Flow -> sampling grid and warping helpers
@torch.no_grad()
def _flow_to_grid(flow: torch.Tensor) -> torch.Tensor:
    """
    flow: [B,2,H,W] in *pixels* (dx, dy), moving -> fixed.
    Returns sampling grid in [-1,1] with shape [B,H,W,2] for grid_sample.
    """
    B, C, H, W = flow.shape
    assert C == 2, "flow must have shape [B,2,H,W]"

    xs = torch.linspace(-1, 1, W, device=flow.device)
    ys = torch.linspace(-1, 1, H, device=flow.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [H,W]

    base = torch.stack((xx, yy), dim=-1)           # [H,W,2]
    base = base.unsqueeze(0).repeat(B, 1, 1, 1)    # [B,H,W,2]

    # convert pixel displacements to normalized coordinates
    dx = flow[:, 0] * (2.0 / max(W - 1, 1))
    dy = flow[:, 1] * (2.0 / max(H - 1, 1))
    disp = torch.stack((dx, dy), dim=-1)          # [B,H,W,2]

    return base + disp


@torch.no_grad()
def _warp2d(moving: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp `moving` image(s) with `flow` using bilinear sampling.

    moving: [B,1,H,W] (prior)
    flow:   [B,2,H,W] (predicted by MammoRegNet, moving->fixed)
    """
    grid = _flow_to_grid(flow)
    return F.grid_sample(
        moving,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def _pick_flow(model_out):
    """
    MammoRegNet might return:
      - a tensor [B,2,H,W]
      - a tuple/list where one element is that tensor
      - a dict with key 'flow' / 'disp' / etc.
    This function extracts the flow tensor.
    """
    if torch.is_tensor(model_out):
        return model_out

    if isinstance(model_out, (list, tuple)):
        for t in model_out:
            try:
                f = _pick_flow(t)
                if torch.is_tensor(f):
                    return f
            except Exception:
                continue

    if isinstance(model_out, dict):
        # common key names
        for k in ("flow", "disp", "field", "dvf"):
            if k in model_out and torch.is_tensor(model_out[k]):
                return model_out[k]
        for v in model_out.values():
            if torch.is_tensor(v) and v.ndim == 4 and v.shape[1] == 2:
                return v

    raise ValueError("Could not find flow tensor in model output.")


# 2. Alignment functions
@torch.no_grad()
def align_prior_to_current(cur_np: np.ndarray,
                           pri_np: np.ndarray,
                           reg_model,
                           device: str | None = None):
    """
    Align a single prior to the current image using MammoRegNet.

    cur_np, pri_np: 2D float32 arrays in [0,1], shape (H,W).
    reg_model:      MammoRegNet instance.
    Returns:
      aligned_prior_np: np.ndarray, shape (H,W), [0,1]
      flow:             torch.Tensor [2,H,W] (on CPU)
    """
    device = device or DEVICE

    # shape: [1,1,H,W]
    I_cur = torch.from_numpy(cur_np).unsqueeze(0).unsqueeze(0).to(device)
    I_pri = torch.from_numpy(pri_np).unsqueeze(0).unsqueeze(0).to(device)

    # MammoRegNet is called as (moving, fixed) = (prior, current)
    out = reg_model(I_pri, I_cur)
    flow = _pick_flow(out)        # [1,2,H,W]
    assert flow.ndim == 4 and flow.shape[1] == 2, "Bad flow shape from model"
    flow = flow.to(I_pri.dtype)

    # warp prior to current
    I_warp = _warp2d(I_pri, flow)  # [1,1,H,W]

    aligned = I_warp[0, 0].detach().cpu().numpy()
    return aligned, flow[0].detach().cpu()   # [2,H,W]


def align_all_priors(cur_img: np.ndarray,
                     prior_imgs: list[np.ndarray],
                     reg_model):
    """
    Align an arbitrary list of priors to the same current image.

    cur_img:   (H,W) float32 in [0,1]
    prior_imgs: list of (H,W) float32 in [0,1]
    Returns:
      aligned_list: list of aligned prior images (same length as prior_imgs)
      flows_list:   list of flow tensors [2,H,W]
    """
    H, W = cur_img.shape
    aligned_list = []
    flows_list = []

    for pri_np in prior_imgs:
        # ensure same spatial size (if you cropped differently)
        if pri_np.shape != (H, W):
            pri_np = cv2.resize(pri_np, (W, H), interpolation=cv2.INTER_LINEAR)

        ali, flow = align_prior_to_current(cur_img, pri_np, reg_model)
        aligned_list.append(ali)
        flows_list.append(flow)

    return aligned_list, flows_list


# 3. Simple overlay for visualization
def overlay_current_with(cur, other, alpha=0.6):
    """
    Build an RGB overlay:
      - red   = current
      - blue  = blend(other, current)
    both cur and other are [H,W] in [0,1]
    """
    c = np.clip(cur, 0, 1)
    o = np.clip(other, 0, 1)
    rgb = np.zeros((c.shape[0], c.shape[1], 3), dtype=np.float32)
    rgb[..., 0] = c
    rgb[..., 2] = alpha * o + (1 - alpha) * c
    return np.clip(rgb, 0, 1)


if __name__ == "__main__":
    MANIFEST_PATH = "temporalSequences_riskcohort_5y.csv"     

    from preprocessing import (
        s3_to_local,
        load_dicom_image_local,
        largest_contour_mask,
        crop_to_mask,
    )

    df = pd.read_csv(MANIFEST_PATH, low_memory=False)

    # make sure list-like columns are parsed
    for c in ["prior_paths", "prior_dates", "prior_gaps_months", "prior_accs"]:
        if c in df.columns:
            df[c] = df[c].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else x
            )

    df["num_priors"] = df["prior_paths"].apply(
        lambda xs: len(xs) if isinstance(xs, list) else 0
    )

    # choose an example row with >= priors (example)
    row = df.loc[df["num_priors"] >= 2].iloc[0]
    print("Using example row with empi_anon:", row["empi_anon"],
          "num_priors:", row["num_priors"])

    # load and preprocess current image 
    cur_path = s3_to_local(row["current_path"])
    cur_img, _ = load_dicom_image_local(cur_path)

    cur_mask = largest_contour_mask((cur_img * 255).astype(np.uint8))
    cur_vis = crop_to_mask((cur_img * 255).astype(np.uint8), cur_mask).astype(np.float32) / 255.0
    cur_img = cur_vis

    # load and preprocess all prior images 
    pri_paths = row["prior_paths"]
    pri_dates = row.get("prior_dates", [])
    pri_gaps = row.get("prior_gaps_months", [])

    prior_imgs = []
    for p in pri_paths:
        pri_path = s3_to_local(p)
        im, _ = load_dicom_image_local(pri_path)
        m = largest_contour_mask((im * 255).astype(np.uint8))
        im = crop_to_mask((im * 255).astype(np.uint8), m).astype(np.float32) / 255.0
        prior_imgs.append(im)

    # align all priors to the current
    aligned_list, flows_list = align_all_priors(cur_img, prior_imgs, REG)

    # visualize each aligned prior 
    H, W = cur_img.shape
    for i, (pri_img, ali_img, p_date, p_gap) in enumerate(
        zip(prior_imgs, aligned_list, pri_dates, pri_gaps), start=1
    ):
        # resize to match current if needed (but this should already match)
        if pri_img.shape != (H, W):
            pri_img = cv2.resize(pri_img, (W, H), interpolation=cv2.INTER_LINEAR)

        gap_ok = (p_gap is not None) and not (
            isinstance(p_gap, float) and math.isnan(p_gap)
        )
        gap_str = f"{float(p_gap):.1f} mo back" if gap_ok else "gap N/A"
        date_str = f"Date: {p_date}" if isinstance(p_date, str) else ""

        fig, axes = plt.subplots(1, 5, figsize=(18, 4))

        axes[0].imshow(cur_vis, cmap="gray")
        axes[0].set_title(
            f"Current — {row['ImageLateralityFinal']}-{row['ViewPosition']}\n"
            f"{row.get('current_date', '')}"
        )
        axes[0].axis("off")

        axes[1].imshow(pri_img, cmap="gray")
        axes[1].set_title(f"Prior {i} (unaligned)\n{gap_str}, {date_str}")
        axes[1].axis("off")

        axes[2].imshow(ali_img, cmap="gray")
        axes[2].set_title(f"Aligned Prior {i}")
        axes[2].axis("off")

        axes[3].imshow(overlay_current_with(cur_vis, pri_img))
        axes[3].set_title("Current overlaid with Prior")
        axes[3].axis("off")

        axes[4].imshow(overlay_current_with(cur_vis, ali_img))
        axes[4].set_title("Current overlaid with Aligned Prior")
        axes[4].axis("off")

        plt.tight_layout()
        plt.show()

    print("Aligned", len(aligned_list), "priors for this exam.")
