#!/usr/bin/env python3
"""
precomputeAllPriorsAligned.py

Aligns all priors -> current using MammoRegNet (moving=prior, fixed=current),

Input jobs CSV columns 
  - row_idx
  - exam_id
  - side_view
  - prior_idx
  - cur_png
  - pri_png

Outputs:
  - Writes aligned uint16 PNGs:
      <out_dir>/<side_view>/<exam_id>_p{prior_idx:03d}.png

"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import hashlib

import torch

from datasetModel import (
    _resize_if_needed,
    FIXED_HW,  # (H,W)
)

from models.mammoregnet import MammoRegNet 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAMMO_HW = (512, 1024)

def load_u16_noresize(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.dtype != np.uint16:
        arr = (arr.astype(np.float32) / 255.0 * 65535.0).astype(np.uint16)
    return arr


def load_u16_fixedhw(path: str) -> np.ndarray:
    arr = load_u16_noresize(path)

    if FIXED_HW is not None:
        Ht, Wt = FIXED_HW
        if arr.shape == (Wt, Ht):  
            arr = arr.T

    arr = _resize_if_needed(arr, FIXED_HW)
    return arr.astype(np.uint16)


def save_u16(path: Path, arr_u16: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_u16.astype(np.uint16)).save(str(path))


def _resize_hw_u16(arr_u16: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    Ht, Wt = int(hw[0]), int(hw[1])
    pil = Image.fromarray(arr_u16.astype(np.uint16))
    pil = pil.resize((Wt, Ht), Image.Resampling.BILINEAR)
    return np.array(pil).astype(np.uint16)

def load_model(ckpt_path: str) -> MammoRegNet:
    ckpt_path = str(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = MammoRegNet()
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=True)
    model.to(DEVICE).eval()
    print(f"[MammoRegNet] loaded checkpoint: {ckpt_path}")
    return model


@torch.no_grad()
def align_one_pair(model: MammoRegNet, prior_u16_fixed: np.ndarray, current_u16_fixed: np.ndarray) -> np.ndarray:
    pri_small = _resize_hw_u16(prior_u16_fixed, MAMMO_HW)
    cur_small = _resize_hw_u16(current_u16_fixed, MAMMO_HW)

    pri_f = (pri_small.astype(np.float32) / 65535.0) * 255.0
    cur_f = (cur_small.astype(np.float32) / 65535.0) * 255.0

    img_fix = torch.from_numpy(cur_f)[None, None, :, :].to(DEVICE)  # fixed = current
    img_mov = torch.from_numpy(pri_f)[None, None, :, :].to(DEVICE)  # moving = prior

    pred = model(img_fix, img_mov)
    warped = pred[0][0, 0].detach().float().cpu().numpy()

    warped = np.clip(warped, 0.0, 255.0)
    warped_u16_small = np.clip((warped / 255.0) * 65535.0, 0, 65535).astype(np.uint16)
    warped_u16_fixed = _resize_hw_u16(warped_u16_small, FIXED_HW)
    return warped_u16_fixed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs_csv", required=True, type=str)
    p.add_argument("--out_dir", required=True, type=str)
    p.add_argument("--ckpt", required=True, type=str)

    p.add_argument("--job_start", default=0, type=int, help="Start index in jobs CSV (inclusive).")
    p.add_argument("--job_end", default=-1, type=int, help="End index in jobs CSV (exclusive). -1 = to end.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--progress_every", default=200, type=int)

    p.add_argument("--write_status_csv", default="", type=str,
                   help="If set, writes a CSV with columns + out_path + status for processed jobs only.")
    return p.parse_args()


def main():
    args = parse_args()

    jobs = pd.read_csv(args.jobs_csv, low_memory=False)
    n_total = len(jobs)

    start = max(0, int(args.job_start))
    end = n_total if int(args.job_end) < 0 else min(n_total, int(args.job_end))
    if start >= end:
        print(f"[WARN] Empty range: start={start} end={end} (n_total={n_total})")
        return

    jobs_slice = jobs.iloc[start:end].copy()
    print(f"[Jobs] total={n_total}  processing slice [{start}:{end})  n={len(jobs_slice)}")

    model = load_model(args.ckpt)
    out_dir = Path(args.out_dir)

    # stats
    n_done = 0
    n_skipped = 0
    n_missing = 0
    n_failed = 0

    status_rows = []

    for k, row in enumerate(jobs_slice.itertuples(index=False), start=1):
        try:
            cur_path = getattr(row, "cur_png", "")
            pri_path = getattr(row, "pri_png", "")
            side_view = str(getattr(row, "side_view", "UNK"))
            exam_id = str(getattr(row, "exam_id", "UNK"))
            prior_idx = int(getattr(row, "prior_idx", 0))

            if not (isinstance(cur_path, str) and cur_path and os.path.isfile(cur_path)):
                n_missing += 1
                if args.write_status_csv:
                    status_rows.append({**row._asdict(), "out_path": "", "status": "missing_cur"})
                continue
            if not (isinstance(pri_path, str) and pri_path and os.path.isfile(pri_path)):
                n_missing += 1
                if args.write_status_csv:
                    status_rows.append({**row._asdict(), "out_path": "", "status": "missing_pri"})
                continue

            pri_hash = hashlib.md5(pri_path.encode("utf-8")).hexdigest()[:12]
            cur_hash = hashlib.md5(cur_path.encode("utf-8")).hexdigest()[:12]
            out_path = out_dir / side_view / f"{exam_id}_p{prior_idx:03d}_c{cur_hash}_p{pri_hash}.png"


            if out_path.exists() and (not args.overwrite):
                n_skipped += 1
                if args.write_status_csv:
                    status_rows.append({**row._asdict(), "out_path": str(out_path), "status": "skipped_exists"})
                continue

            cur_u16 = load_u16_fixedhw(cur_path)
            pri_u16 = load_u16_fixedhw(pri_path)

            aligned_u16 = align_one_pair(model, pri_u16, cur_u16)
            save_u16(out_path, aligned_u16)

            n_done += 1
            if args.write_status_csv:
                status_rows.append({**row._asdict(), "out_path": str(out_path), "status": "done"})

            if args.progress_every and (n_done % args.progress_every == 0):
                print(f"[done {n_done}] example_out={out_path}")

        except Exception as e:
            n_failed += 1
            if args.write_status_csv:
                status_rows.append({**row._asdict(), "out_path": "", "status": f"failed:{type(e).__name__}"})

            continue

    print("\nDone.")
    print(f"  out_dir  : {out_dir}")
    print(f"  processed: {len(jobs_slice)} (slice)")
    print(f"  aligned  : {n_done}")
    print(f"  skipped  : {n_skipped}")
    print(f"  missing  : {n_missing}")
    print(f"  failed   : {n_failed}")

    if args.write_status_csv:
        out_csv = Path(args.write_status_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(status_rows).to_csv(out_csv, index=False)
        print("  status_csv:", str(out_csv))


if __name__ == "__main__":
    main()
