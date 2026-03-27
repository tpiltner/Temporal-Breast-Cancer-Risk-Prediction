#datasetModel.py 

import ast
import os
from pathlib import Path
from typing import Optional, Tuple, List
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

CSV_PATH = Path(os.environ.get(
    "EMBED_CSV_PATH",
    "/local/scratch/tpiltne/utils/trainingWithPNGS.csv"
))

# all-aligned priors (from precomputeAllPriorsAligned.py status.csv) 
ALL_ALIGNED_DIR = Path("/local/scratch/tpiltne/allAlignedPriors")
ALL_ALIGNED_STATUS_CSV = ALL_ALIGNED_DIR / "status.csv"


HORIZONS = [1, 2, 3, 4, 5]
RISK_POS_COLS = [f"risk_{h}y_pos" for h in HORIZONS]
RISK_NEG_COLS = [f"risk_{h}y_neg" for h in HORIZONS]

RISK_COLS = RISK_POS_COLS

EXAM_ID_COL = "acc_anon"
VIEW_ORDER = ["L-CC", "R-CC", "L-MLO", "R-MLO"]

PATIENT_COL = "empi_anon"
CUR_PNG_COL = "cur_png"
PRIOR_COL = "pri_png"

LAT_COL = "ImageLateralityFinal"
VIEWPOS_COL = "ViewPosition"

FOLLOWUP_YEARS_CANDIDATES = ["followup_years_exam"]

# label semantics
LABEL_COLS_ARE_EVENT = False
AUTO_INFER_LABEL_FLIP = True
INFER_SAMPLE_ROWS = 5000

# image sizing
FIXED_HW: Optional[Tuple[int, int]] = (1664, 2048)  # (H, W)
USE_LANCZOS_RESIZE = True

# caching
ENABLE_IMAGE_CACHE = True
IMAGE_CACHE_MAX_ITEMS = 2048


# Temporal input tensor order is:
#   [0:4]   current (4 views)
#   [4:8]   prior   (4 views)
#   [8:12]  diff    (4 views)
#
DIFF_MODE = "signed"   
CLIP_DIFF = False
DIFF_CLIP_MIN = -1.0
DIFF_CLIP_MAX =  1.0

# Helpers
def _to_int01(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    out = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    out[out != 0] = 1
    return out


def _resize_if_needed(img_u16: np.ndarray, fixed_hw: Optional[Tuple[int, int]]) -> np.ndarray:
    if fixed_hw is None:
        return img_u16
    Ht, Wt = fixed_hw
    if img_u16.shape[0] == Ht and img_u16.shape[1] == Wt:
        return img_u16
    pil = Image.fromarray(img_u16)
    resample = Image.Resampling.LANCZOS if USE_LANCZOS_RESIZE else Image.Resampling.BILINEAR
    pil = pil.resize((Wt, Ht), resample=resample)
    return np.array(pil)


def _normalize01(img_u16: np.ndarray) -> np.ndarray:
    return img_u16.astype(np.float32) / 65535.0


def _load_png_16bit_to_chw(png_path: Path, fixed_hw: Optional[Tuple[int, int]] = FIXED_HW) -> np.ndarray:
    arr = np.array(Image.open(png_path))  # uint16 (H,W)

    # If image is (2048,1664) but fixed is (1664,2048), transpose it before resizing.
    if fixed_hw is not None:
        Ht, Wt = fixed_hw
        if arr.shape == (Wt, Ht):   # (2048,1664) when fixed is (1664,2048)
            arr = arr.T
            
    arr = _resize_if_needed(arr, fixed_hw)
    img = _normalize01(arr)
    return np.expand_dims(img, axis=0)  # [1,H,W]


def _infer_hw_from_png_path(p: Path, fixed_hw: Optional[Tuple[int, int]] = FIXED_HW) -> Tuple[int, int]:
    if fixed_hw is not None:
        return int(fixed_hw[0]), int(fixed_hw[1])
    arr = np.array(Image.open(p))
    return int(arr.shape[0]), int(arr.shape[1])


def _try_get_followup_years_row(row: pd.Series) -> Optional[float]:
    for col in FOLLOWUP_YEARS_CANDIDATES:
        if col in row.index:
            try:
                v = float(row[col])
            except Exception:
                continue
            if np.isnan(v):
                continue
            if "day" in col.lower():
                return v / 365.25
            return v
    return None


def _mask_from_followup(y_event: np.ndarray, followup_years: Optional[float]) -> Optional[np.ndarray]:
    """
    Observed at horizon t if:
      - followup_years >= t, OR
      - event occurred by t
    """
    if followup_years is None:
        return None
    follow = np.array([(followup_years >= t) for t in HORIZONS], dtype=np.float32)
    event = (y_event > 0).astype(np.float32)
    return np.maximum(follow, event)


def _mask_from_posneg(y_event: np.ndarray, y_neg: np.ndarray) -> np.ndarray:
    return np.maximum((y_neg > 0).astype(np.float32), (y_event > 0).astype(np.float32))


def _infer_need_flip(df: pd.DataFrame) -> bool:
    cols = [c for c in RISK_POS_COLS if c in df.columns]
    if not cols:
        return False
    sub = df[cols].head(INFER_SAMPLE_ROWS).copy()
    for c in cols:
        sub[c] = _to_int01(sub[c])
    c_use = cols[-1]  # prefer 5y
    mean_val = float(sub[c_use].mean())
    # if mean > 0.5, it's probably "negatives" not "events"
    return mean_val > 0.5


def _get_y_event_from_row(row0: pd.Series, flip: bool) -> np.ndarray:
    y_raw = np.array([float(row0[c]) for c in RISK_POS_COLS], dtype=np.float32)
    return (1.0 - y_raw) if flip else y_raw


# Prior parsing + "most recent" selection
def _parse_listish(raw):
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    try:
        out = ast.literal_eval(s)
        return out if isinstance(out, (list, tuple)) else None
    except Exception:
        return None


def _parse_prior_paths(raw) -> List[str]:
    if not isinstance(raw, str):
        return []
    s = raw.strip()
    if s == "" or s.lower() == "nan":
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            lst = ast.literal_eval(s)
            if isinstance(lst, (list, tuple)):
                return [str(x).strip() for x in lst if isinstance(x, str) and str(x).strip() != ""]
        except Exception:
            s2 = s.strip("[]").strip()
            parts = [p.strip().strip("'").strip('"') for p in s2.split(",")]
            return [p for p in parts if p]
    return [s]


def _choose_most_recent_prior_from_row(row: pd.Series) -> Optional[str]:
    pri_paths = _parse_prior_paths(row.get(PRIOR_COL, None))
    if not pri_paths:
        return None

    if "prior_gaps_months" in row.index:
        gaps = _parse_listish(row["prior_gaps_months"])
        if isinstance(gaps, (list, tuple)) and len(gaps) == len(pri_paths):
            try:
                g = np.array(gaps, dtype=np.float32)
                j = int(np.argmin(g))
                return pri_paths[j]
            except Exception:
                pass

    if "prior_dates" in row.index:
        dates = _parse_listish(row["prior_dates"])
        if isinstance(dates, (list, tuple)) and len(dates) == len(pri_paths):
            parsed = []
            for d, p in zip(dates, pri_paths):
                try:
                    parsed.append((datetime.fromisoformat(str(d)[:10]), p))
                except Exception:
                    pass
            if parsed:
                parsed.sort(key=lambda x: x[0])
                return parsed[-1][1]

    return pri_paths[0]


def _get_most_recent_gap_months_from_row(row: pd.Series) -> float:
    if "prior_gaps_months" in row.index:
        gaps = _parse_listish(row["prior_gaps_months"])
        if isinstance(gaps, (list, tuple)) and len(gaps) > 0:
            try:
                return float(np.min(np.array(gaps, dtype=np.float32)))
            except Exception:
                pass
    return 0.0

def _load_allaligned_index(status_csv: Path):
    """
    Reads status.csv produced by precomputeAllPriorsAligned.py and builds:
      idx[(exam_id, side_view)][prior_idx] = out_path
    Only includes rows with status == "done" and out_path exists.
    """
    status_csv = Path(status_csv)
    if not status_csv.is_file():
        raise FileNotFoundError(f"ALL_ALIGNED_STATUS_CSV not found: {status_csv}")

    sdf = pd.read_csv(status_csv, low_memory=False)

    need = {"exam_id", "side_view", "prior_idx", "out_path", "status"}
    missing = need - set(sdf.columns)
    if missing:
        raise RuntimeError(f"status.csv missing columns: {sorted(missing)}")

    sdf = sdf[sdf["status"].astype(str) == "done"].copy()
    sdf["out_path"] = sdf["out_path"].astype(str).str.strip()

    idx = {}  # (exam_id, side_view) -> {prior_idx: out_path}
    for r in sdf.itertuples(index=False):
        exam_id = str(getattr(r, "exam_id"))
        side_view = str(getattr(r, "side_view"))
        prior_idx = int(getattr(r, "prior_idx"))
        out_path = str(getattr(r, "out_path"))

        if out_path and os.path.isfile(out_path):
            key = (exam_id, side_view)
            if key not in idx:
                idx[key] = {}
            idx[key][prior_idx] = out_path

    return idx


def _get_allaligned_paths_for_view(idx_map, exam_id: str, side_view: str):
    """
    Returns aligned paths sorted by prior_idx ascending.
    """
    d = idx_map.get((str(exam_id), str(side_view)), None)
    if not d:
        return []
    return [d[k] for k in sorted(d.keys())]


def _choose_most_recent_allaligned(idx_map, exam_id: str, side_view: str):
    """
    Picks the 'most recent' aligned prior.
    Assumption: prior_idx=0 is most recent 
    If not present, falls back to smallest prior_idx available.
    """
    paths = _get_allaligned_paths_for_view(idx_map, exam_id, side_view)
    return paths[0] if paths else None


# diff construction 
def _make_diff(cur_img: np.ndarray, pri_img: np.ndarray, has_prior: float) -> np.ndarray:
    """
    cur_img, pri_img: [1,H,W] float32 in [0,1]
    returns: diff_img [1,H,W]
    """
    if has_prior <= 0.0:
        return np.zeros_like(cur_img, dtype=np.float32)

    diff = (cur_img - pri_img)

    if CLIP_DIFF:
        diff = np.clip(diff, DIFF_CLIP_MIN, DIFF_CLIP_MAX)

    return diff.astype(np.float32)

# Exam grouping
def _build_exam_groups(df: pd.DataFrame, need_flip: bool):
    df = df.copy()
    df["side_view"] = (
        df[LAT_COL].astype(str).str.strip()
        + "-"
        + df[VIEWPOS_COL].astype(str).str.strip()
    )

    has_neg = all(c in df.columns for c in RISK_NEG_COLS)

    exam_groups = []
    for exam_id, df_exam in df.groupby(EXAM_ID_COL):
        df_exam = df_exam.reset_index(drop=True)
        row0 = df_exam.iloc[0]

        y_event = _get_y_event_from_row(row0, flip=need_flip)

        if has_neg:
            y_neg = np.array([float(row0[c]) for c in RISK_NEG_COLS], dtype=np.float32)
            mask = _mask_from_posneg(y_event, y_neg)
        else:
            followup_years = _try_get_followup_years_row(row0)
            mask_fu = _mask_from_followup(y_event, followup_years)
            mask = mask_fu if mask_fu is not None else np.ones_like(y_event, dtype=np.float32)

        p0 = None
        for _, r in df_exam.iterrows():
            v = r[CUR_PNG_COL]
            if isinstance(v, str) and v != "":
                p0 = Path(v)
                break
        if p0 is None:
            continue
        hw = _infer_hw_from_png_path(p0, FIXED_HW)

        exam_groups.append((str(exam_id), df_exam, y_event, mask, hw))

    return exam_groups

class _LRUCache:
    def __init__(self, max_items: int = 1024):
        self.max_items = max_items
        self._d: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def get(self, key: str):
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def put(self, key: str, value: np.ndarray):
        self._d[key] = value
        self._d.move_to_end(key)
        if len(self._d) > self.max_items:
            self._d.popitem(last=False)


def _choose_most_recent_prior_index_from_row(row: pd.Series) -> Optional[int]:
    pri_paths = _parse_prior_paths(row.get(PRIOR_COL, None))
    if not pri_paths:
        return None

    # Prefer gaps if available
    if "prior_gaps_months" in row.index:
        gaps = _parse_listish(row["prior_gaps_months"])
        if isinstance(gaps, (list, tuple)) and len(gaps) == len(pri_paths):
            try:
                g = np.array(gaps, dtype=np.float32)
                return int(np.argmin(g))  # most recent = smallest gap
            except Exception:
                pass

    # Else try prior_dates
    if "prior_dates" in row.index:
        dates = _parse_listish(row["prior_dates"])
        if isinstance(dates, (list, tuple)) and len(dates) == len(pri_paths):
            parsed = []
            for i, d in enumerate(dates):
                try:
                    parsed.append((datetime.fromisoformat(str(d)[:10]), i))
                except Exception:
                    pass
            if parsed:
                parsed.sort(key=lambda x: x[0])
                return parsed[-1][1]  # latest date
    return 0

# Datasets
class Temporal1Dataset(Dataset):
    """
    Unaligned temporal dataset (raw priors)
      returns (imgs, delta_feat, has_prior_views, y, mask)

    imgs:           [12,1,H,W] (4 current + 4 prior + 4 diff)
    delta_feat:     [2]        (delta_years, has_any_prior)
    has_prior_views:[4]        (per-view: 1 if prior exists else 0)
    y:              [5]
    mask:           [5]
    """
    def __init__(self, split: str = "train", csv_path: Path = CSV_PATH):
        df = pd.read_csv(csv_path, low_memory=False)
        df = df[df[CUR_PNG_COL].notna() & (df[CUR_PNG_COL] != "")].copy()

        if "split" not in df.columns:
            raise RuntimeError("CSV has no 'split' column. Run splitDataset.py first.")
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split '{split}'")
        df = df[df["split"] == split].copy()

        for col in [EXAM_ID_COL, LAT_COL, VIEWPOS_COL, PRIOR_COL]:
            if col not in df.columns:
                raise RuntimeError(f"CSV must have '{col}' column")

        missing_pos = [c for c in RISK_POS_COLS if c not in df.columns]
        if missing_pos:
            raise RuntimeError(f"Missing required POS risk columns: {missing_pos}")

        for c in RISK_POS_COLS:
            df[c] = _to_int01(df[c])
        if all(c in df.columns for c in RISK_NEG_COLS):
            for c in RISK_NEG_COLS:
                df[c] = _to_int01(df[c])

        if AUTO_INFER_LABEL_FLIP:
            need_flip = _infer_need_flip(df)
        else:
            need_flip = (not LABEL_COLS_ARE_EVENT)

        self.flip_to_event = need_flip
        self.split = split
        print(f"[Temporal1Dataset:{split}] flip_to_event={self.flip_to_event} FIXED_HW={FIXED_HW} DIFF_MODE={DIFF_MODE}")

        self.exam_groups = _build_exam_groups(df, need_flip=self.flip_to_event)
        self.cache = _LRUCache(IMAGE_CACHE_MAX_ITEMS) if ENABLE_IMAGE_CACHE else None

    def __len__(self) -> int:
        return len(self.exam_groups)

    def _load_cached(self, path: str) -> np.ndarray:
        if self.cache is None:
            return _load_png_16bit_to_chw(Path(path), FIXED_HW)
        v = self.cache.get(path)
        if v is None:
            v = _load_png_16bit_to_chw(Path(path), FIXED_HW)
            self.cache.put(path, v)
        return v

    def __getitem__(self, idx: int):
        exam_id, df_exam, y_event, mask, (H, W) = self.exam_groups[idx]

        df_exam = df_exam.copy()
        df_exam["side_view"] = (
            df_exam[LAT_COL].astype(str).str.strip()
            + "-"
            + df_exam[VIEWPOS_COL].astype(str).str.strip()
        )

        # exam-level delta features 
        row0 = df_exam.iloc[0]
        raw_prior_path_exam = _choose_most_recent_prior_from_row(row0)
        has_any_prior = 1.0 if (isinstance(raw_prior_path_exam, str) and raw_prior_path_exam != "") else 0.0
        gap_months = _get_most_recent_gap_months_from_row(row0) if has_any_prior > 0 else 0.0
        delta_years = float(gap_months) / 12.0
        delta_feat = np.array([delta_years, has_any_prior], dtype=np.float32)

        cur_imgs, pri_imgs, dif_imgs = [], [], []
        has_prior_views = []

        for view_name in VIEW_ORDER:
            df_view = df_exam[df_exam["side_view"] == view_name]

            cur_img = np.zeros((1, H, W), dtype=np.float32)
            pri_img = np.zeros((1, H, W), dtype=np.float32)
            has_p = 0.0

            if len(df_view) != 0:
                row = df_view.iloc[0]

                # current
                cur_path = row[CUR_PNG_COL]
                if isinstance(cur_path, str) and cur_path != "" and os.path.isfile(cur_path):
                    cur_img = self._load_cached(cur_path)

                # prior (raw)
                prior_path = _choose_most_recent_prior_from_row(row)
                if isinstance(prior_path, str) and prior_path != "" and os.path.isfile(prior_path):
                    pri_img = self._load_cached(prior_path)
                    has_p = 1.0

            diff_img = _make_diff(cur_img, pri_img, has_p)

            cur_imgs.append(cur_img)
            pri_imgs.append(pri_img)
            dif_imgs.append(diff_img)
            has_prior_views.append(has_p)

        imgs = np.stack(cur_imgs + pri_imgs + dif_imgs, axis=0)  # [12,1,H,W]
        has_prior_views = np.array(has_prior_views, dtype=np.float32)  # [4]

        return (
            torch.from_numpy(imgs),
            torch.from_numpy(delta_feat),
            torch.from_numpy(has_prior_views),
            torch.from_numpy(y_event.astype(np.float32)),
            torch.from_numpy(mask.astype(np.float32)),
        )

class CurrentOnlyDataset(Dataset):
    """
    returns (imgs, delta_feat, has_prior_views, y, mask)

    imgs:           [12,1,H,W] (4 current + 4 zeros(prior) + 4 zeros(diff))
    delta_feat:     [2]        (0.0, 0.0)
    has_prior_views:[4]        (all zeros)
    y:              [5]
    mask:           [5]
    """
    def __init__(self, split: str = "train", csv_path: Path = CSV_PATH):
        df = pd.read_csv(csv_path, low_memory=False)
        df = df[df[CUR_PNG_COL].notna() & (df[CUR_PNG_COL] != "")].copy()

        if "split" not in df.columns:
            raise RuntimeError("CSV has no 'split' column. Run splitDataset.py first.")
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split '{split}'")
        df = df[df["split"] == split].copy()

        for col in [EXAM_ID_COL, LAT_COL, VIEWPOS_COL]:
            if col not in df.columns:
                raise RuntimeError(f"CSV must have '{col}' column")

        missing_pos = [c for c in RISK_POS_COLS if c not in df.columns]
        if missing_pos:
            raise RuntimeError(f"Missing required POS risk columns: {missing_pos}")

        for c in RISK_POS_COLS:
            df[c] = _to_int01(df[c])
        if all(c in df.columns for c in RISK_NEG_COLS):
            for c in RISK_NEG_COLS:
                df[c] = _to_int01(df[c])

        if AUTO_INFER_LABEL_FLIP:
            need_flip = _infer_need_flip(df)
        else:
            need_flip = (not LABEL_COLS_ARE_EVENT)

        self.flip_to_event = need_flip
        self.split = split
        print(f"[CurrentOnlyDataset:{split}] flip_to_event={self.flip_to_event} FIXED_HW={FIXED_HW}")

        self.exam_groups = _build_exam_groups(df, need_flip=self.flip_to_event)
        self.cache = _LRUCache(IMAGE_CACHE_MAX_ITEMS) if ENABLE_IMAGE_CACHE else None

    def __len__(self) -> int:
        return len(self.exam_groups)

    def _load_cached(self, path: str) -> np.ndarray:
        if self.cache is None:
            return _load_png_16bit_to_chw(Path(path), FIXED_HW)
        v = self.cache.get(path)
        if v is None:
            v = _load_png_16bit_to_chw(Path(path), FIXED_HW)
            self.cache.put(path, v)
        return v

    def __getitem__(self, idx: int):
        exam_id, df_exam, y_event, mask, (H, W) = self.exam_groups[idx]

        df_exam = df_exam.copy()
        df_exam["side_view"] = df_exam[LAT_COL].astype(str) + "-" + df_exam[VIEWPOS_COL].astype(str)

        cur_imgs = []
        for view_name in VIEW_ORDER:
            df_view = df_exam[df_exam["side_view"] == view_name]
            img = np.zeros((1, H, W), dtype=np.float32)

            if len(df_view) != 0:
                row = df_view.iloc[0]
                cur_path = row[CUR_PNG_COL]
                if isinstance(cur_path, str) and cur_path != "" and os.path.isfile(cur_path):
                    img = self._load_cached(cur_path)

            cur_imgs.append(img)

        # priors + diffs are all zeros for current-only
        pri_imgs = [np.zeros((1, H, W), dtype=np.float32) for _ in range(4)]
        dif_imgs = [np.zeros((1, H, W), dtype=np.float32) for _ in range(4)]

        imgs = np.stack(cur_imgs + pri_imgs + dif_imgs, axis=0)  # [12,1,H,W]
        delta_feat = np.array([0.0, 0.0], dtype=np.float32)
        has_prior_views = np.zeros(4, dtype=np.float32)

        return (
            torch.from_numpy(imgs),
            torch.from_numpy(delta_feat),
            torch.from_numpy(has_prior_views),
            torch.from_numpy(y_event.astype(np.float32)),
            torch.from_numpy(mask.astype(np.float32)),
        )


#   imgs: [12,1,H,W] = 4 cur + 4 prior(aligned) + 4 diff (computed same way as Temporal1Dataset)
# The only difference vs Temporal1Dataset is: the prior PNG comes from status.csv aligned outputs.

class Temporal1AlignedDataset(Dataset):
    """
    Aligned temporal dataset 

    Returns:
      imgs: [12,1,H,W] = 4 current + 4 aligned prior (most-recent) + 4 diff (cur - pri )
      delta_feat: [2] = (delta_years, has_any_prior) from raw prior metadata (same as Temporal1Dataset)
      has_prior_views: [4] per-view aligned availability
      y: [5], mask: [5]
    """
    def __init__(
        self,
        split: str = "train",
        csv_path: Path = CSV_PATH,
        status_csv: Path = ALL_ALIGNED_STATUS_CSV,
        strict_aligned: bool = True,
        validate_coverage: bool = True,
    ):
        df_all = pd.read_csv(csv_path, low_memory=False)
        df_all = df_all[df_all[CUR_PNG_COL].notna() & (df_all[CUR_PNG_COL] != "")].copy()

        if "split" not in df_all.columns:
            raise RuntimeError("CSV has no 'split' column. Run splitDataset.py first.")
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split '{split}'")

        for col in [EXAM_ID_COL, LAT_COL, VIEWPOS_COL, PRIOR_COL]:
            if col not in df_all.columns:
                raise RuntimeError(f"CSV must have '{col}' column")

        missing_pos = [c for c in RISK_POS_COLS if c not in df_all.columns]
        if missing_pos:
            raise RuntimeError(f"Missing required POS risk columns: {missing_pos}")

        # normalize labels
        for c in RISK_POS_COLS:
            df_all[c] = _to_int01(df_all[c])
        if all(c in df_all.columns for c in RISK_NEG_COLS):
            for c in RISK_NEG_COLS:
                df_all[c] = _to_int01(df_all[c])

        need_flip = _infer_need_flip(df_all) if AUTO_INFER_LABEL_FLIP else (not LABEL_COLS_ARE_EVENT)

        df = df_all[df_all["split"] == split].copy()

        self.flip_to_event = bool(need_flip)
        self.split = split
        self.strict_aligned = bool(strict_aligned)

        self.exam_groups = _build_exam_groups(df, need_flip=self.flip_to_event)
        self.cache = _LRUCache(IMAGE_CACHE_MAX_ITEMS) if ENABLE_IMAGE_CACHE else None

        # aligned index from status.csv
        self.allaligned_idx = _load_allaligned_index(Path(status_csv))

        if validate_coverage:
            self._validate_alignment_coverage()

        print(
            f"[Temporal1AlignedDataset:{split}] flip_to_event={self.flip_to_event} "
            f"FIXED_HW={FIXED_HW} strict_aligned={self.strict_aligned} "
            f"DIFF_MODE={DIFF_MODE} status_csv={status_csv}"
        )

    def _validate_alignment_coverage(self):
        total_exams = len(self.exam_groups)
        exams_with_aligned = 0
        total_views = 0
        views_with_aligned = 0

        for exam_id, df_exam, _, _, _ in self.exam_groups:
            df_exam = df_exam.copy()
            df_exam["side_view"] = df_exam[LAT_COL].astype(str).str.strip() + "-" + df_exam[VIEWPOS_COL].astype(str).str.strip()

            has_any = False
            for view_name in VIEW_ORDER:
                total_views += 1
                df_view = df_exam[df_exam["side_view"] == view_name]
                if len(df_view) == 0:
                    continue
                row = df_view.iloc[0]

                pri_paths = _parse_prior_paths(row.get(PRIOR_COL, None))
                if not pri_paths:
                    continue

                prior_idx = _choose_most_recent_prior_index_from_row(row)
                d = self.allaligned_idx.get((str(exam_id), str(view_name)), None)
                if d is not None and prior_idx in d:
                    p = d[prior_idx]
                    if p and os.path.isfile(p):
                        views_with_aligned += 1
                        has_any = True

            if has_any:
                exams_with_aligned += 1

        exam_cov = 100.0 * exams_with_aligned / max(total_exams, 1)
        view_cov = 100.0 * views_with_aligned / max(total_views, 1)
        print(f"[VALIDATION] Alignment coverage for {self.split}: exams={exam_cov:.1f}% views={view_cov:.1f}%")

    def __len__(self) -> int:
        return len(self.exam_groups)

    def _load_cached(self, path: str) -> np.ndarray:
        if self.cache is None:
            return _load_png_16bit_to_chw(Path(path), FIXED_HW)
        v = self.cache.get(path)
        if v is None:
            v = _load_png_16bit_to_chw(Path(path), FIXED_HW)
            self.cache.put(path, v)
        return v

    def __getitem__(self, idx: int):
        exam_id, df_exam, y_event, mask, (H, W) = self.exam_groups[idx]

        df_exam = df_exam.copy()
        df_exam["side_view"] = (
            df_exam[LAT_COL].astype(str).str.strip()
            + "-"
            + df_exam[VIEWPOS_COL].astype(str).str.strip()
        )

        # exam-level delta features (same as Temporal1Dataset)
        row0 = df_exam.iloc[0]
        raw_prior_path_exam = _choose_most_recent_prior_from_row(row0)
        has_any_prior = 1.0 if (isinstance(raw_prior_path_exam, str) and raw_prior_path_exam != "") else 0.0
        gap_months = _get_most_recent_gap_months_from_row(row0) if has_any_prior > 0 else 0.0
        delta_years = float(gap_months) / 12.0
        delta_feat = np.array([delta_years, has_any_prior], dtype=np.float32)

        cur_imgs, pri_imgs, dif_imgs = [], [], []
        has_prior_views = []

        for view_name in VIEW_ORDER:
            df_view = df_exam[df_exam["side_view"] == view_name]

            cur_img = np.zeros((1, H, W), dtype=np.float32)
            pri_img = np.zeros((1, H, W), dtype=np.float32)
            has_p = 0.0

            if len(df_view) != 0:
                row = df_view.iloc[0]

                # current (same as Temporal1Dataset)
                cur_path = row[CUR_PNG_COL]
                if isinstance(cur_path, str) and cur_path != "" and os.path.isfile(cur_path):
                    cur_img = self._load_cached(cur_path)

                # aligned prior that matches the "most recent raw prior index"
                prior_idx = _choose_most_recent_prior_index_from_row(row)
                ali_path = None
                d = self.allaligned_idx.get((str(exam_id), str(view_name)), None)
                if d is not None and prior_idx is not None and int(prior_idx) in d:
                    ali_path = d[int(prior_idx)]

                if ali_path is not None and os.path.isfile(ali_path):
                    pri_img = self._load_cached(ali_path)
                    has_p = 1.0
                else:
                    if not self.strict_aligned:
                        raw_path = _choose_most_recent_prior_from_row(row)
                        if isinstance(raw_path, str) and raw_path != "" and os.path.isfile(raw_path):
                            pri_img = self._load_cached(raw_path)
                            has_p = 1.0

            diff_img = _make_diff(cur_img, pri_img, has_p)

            cur_imgs.append(cur_img)
            pri_imgs.append(pri_img)
            dif_imgs.append(diff_img)
            has_prior_views.append(has_p)

        imgs = np.stack(cur_imgs + pri_imgs + dif_imgs, axis=0)  # [12,1,H,W]
        has_prior_views = np.array(has_prior_views, dtype=np.float32)

        return (
            torch.from_numpy(imgs),
            torch.from_numpy(delta_feat),
            torch.from_numpy(has_prior_views),
            torch.from_numpy(y_event.astype(np.float32)),
            torch.from_numpy(mask.astype(np.float32)),
        )

class TemporalMultiPriorDataset(Dataset):
    """
    Returns variable-length priors as visits (unaligned).
      cur_imgs:  [4,1,H,W]
      pri_imgs:  [K,4,1,H,W]  (K varies)
      pri_years: [K]          (years before current)
      y: [5], mask: [5]
    """
    def __init__(self, split="train", csv_path=CSV_PATH):
        df = pd.read_csv(csv_path, low_memory=False)
        df = df[df[CUR_PNG_COL].notna() & (df[CUR_PNG_COL] != "")].copy()
        df = df[df["split"] == split].copy()

        if AUTO_INFER_LABEL_FLIP:
            need_flip = _infer_need_flip(df)
        else:
            need_flip = (not LABEL_COLS_ARE_EVENT)

        self.flip_to_event = need_flip
        self.split = split
        self.exam_groups = _build_exam_groups(df, need_flip=self.flip_to_event)
        self.cache = _LRUCache(IMAGE_CACHE_MAX_ITEMS) if ENABLE_IMAGE_CACHE else None

    def _load_cached(self, path: str) -> np.ndarray:
        if self.cache is None:
            return _load_png_16bit_to_chw(Path(path), FIXED_HW)
        v = self.cache.get(path)
        if v is None:
            v = _load_png_16bit_to_chw(Path(path), FIXED_HW)
            self.cache.put(path, v)
        return v

    def __len__(self):
        return len(self.exam_groups)

    def __getitem__(self, idx):
        exam_id, df_exam, y_event, mask, (H, W) = self.exam_groups[idx]
        df_exam = df_exam.copy()
        df_exam["side_view"] = df_exam[LAT_COL].astype(str) + "-" + df_exam[VIEWPOS_COL].astype(str)

        # load current 4 views 
        cur_imgs = []
        pri_lists_by_view = []
        gaps_lists_by_view = []

        for view_name in VIEW_ORDER:
            df_view = df_exam[df_exam["side_view"] == view_name]
            cur_img = np.zeros((1,H,W), np.float32)
            pri_paths, pri_gaps = [], []

            if len(df_view) != 0:
                row = df_view.iloc[0]
                # current
                cur_path = row[CUR_PNG_COL]
                if isinstance(cur_path, str) and cur_path != "" and os.path.isfile(cur_path):
                    cur_img = self._load_cached(cur_path)

                # all priors for this view
                pri_paths = _parse_prior_paths(row.get(PRIOR_COL, None))

                if "prior_gaps_months" in row.index:
                    gl = _parse_listish(row["prior_gaps_months"])
                    if isinstance(gl,(list,tuple)) and len(gl)==len(pri_paths):
                        pri_gaps = [float(x) for x in gl]
                    else:
                        pri_gaps = []

            cur_imgs.append(cur_img)
            pri_lists_by_view.append(pri_paths)
            gaps_lists_by_view.append(pri_gaps)

        # define K as min priors across all 4 views 
        lens_all = [len(L) for L in pri_lists_by_view]
        K = min(lens_all) if len(lens_all) else 0

        pri_lists_by_view = [L[:K] for L in pri_lists_by_view]
        gaps_lists_by_view = [G[:K] if len(G) >= K else [] for G in gaps_lists_by_view]

        # build prior visits: [K,4,1,H,W] 
        pri_imgs = []
        pri_years = []
        for j in range(K):
            visit_imgs = []
            visit_gaps = []
            for v in range(4):
                path = pri_lists_by_view[v][j] 
                if isinstance(path, str) and path != "" and os.path.isfile(path):
                    visit_imgs.append(self._load_cached(path))
                else:
                    visit_imgs.append(np.zeros((1, H, W), np.float32))

                # gaps: just check index exists
                if len(gaps_lists_by_view[v]) > j:
                    visit_gaps.append(gaps_lists_by_view[v][j])

            pri_imgs.append(np.stack(visit_imgs, axis=0))  # [4,1,H,W]
            if len(visit_gaps) == 4:
                pri_years.append(float(np.mean(visit_gaps)) / 12.0)
            else:
                pri_years.append(0.0)


        cur_imgs = np.stack(cur_imgs, axis=0)  # [4,1,H,W]
        pri_imgs = np.stack(pri_imgs, axis=0) if K>0 else np.zeros((0,4,1,H,W), np.float32)
        pri_years = np.array(pri_years, np.float32)

        return (
            torch.from_numpy(cur_imgs),
            torch.from_numpy(pri_imgs),
            torch.from_numpy(pri_years),
            torch.from_numpy(y_event.astype(np.float32)),
            torch.from_numpy(mask.astype(np.float32)),
        )

def _get_allaligned_dict_for_view(idx_map, exam_id: str, side_view: str):
    """
    Returns dict {prior_idx: out_path} for a given (exam_id, side_view).
    Only includes paths that exist on disk.
    """
    d = idx_map.get((str(exam_id), str(side_view)), None)
    if not d:
        return {}
    out = {}
    for k, p in d.items():
        if p and os.path.isfile(p):
            out[int(k)] = p
    return out


class TemporalMultiPriorAllAlignedDataset(Dataset):
    """
    Multi-prior Aligned dataset.
    """
    def __init__(
        self,
        split: str = "train",
        csv_path: Path = CSV_PATH,
        status_csv: Path = ALL_ALIGNED_STATUS_CSV,
    ):
        df_all = pd.read_csv(csv_path, low_memory=False)
        df_all = df_all[df_all[CUR_PNG_COL].notna() & (df_all[CUR_PNG_COL] != "")].copy()

        if "split" not in df_all.columns:
            raise RuntimeError("CSV has no 'split' column.")
        
        # normalize labels
        for c in RISK_POS_COLS:
            df_all[c] = _to_int01(df_all[c])
        if all(c in df_all.columns for c in RISK_NEG_COLS):
            for c in RISK_NEG_COLS:
                df_all[c] = _to_int01(df_all[c])

        if AUTO_INFER_LABEL_FLIP:
            need_flip = _infer_need_flip(df_all)
        else:
            need_flip = (not LABEL_COLS_ARE_EVENT)

        self.flip_to_event = bool(need_flip)

        df = df_all[df_all["split"] == split].copy()
        self.split = split

        self.exam_groups = _build_exam_groups(df, need_flip=self.flip_to_event)
        self.cache = _LRUCache(IMAGE_CACHE_MAX_ITEMS) if ENABLE_IMAGE_CACHE else None

        # aligned index
        self.allaligned_idx = _load_allaligned_index(Path(status_csv))

        print(
            f"[TemporalMultiPriorAllAlignedDataset:{split}] flip={self.flip_to_event} "
            f"FIXED_HW={FIXED_HW} status_csv={status_csv}"
        )

    def _load_cached(self, path: str) -> np.ndarray:
        if self.cache is None:
            return _load_png_16bit_to_chw(Path(path), FIXED_HW)
        v = self.cache.get(path)
        if v is None:
            v = _load_png_16bit_to_chw(Path(path), FIXED_HW)
            self.cache.put(path, v)
        return v

    def __len__(self):
        return len(self.exam_groups)

    def __getitem__(self, idx):
        exam_id, df_exam, y_event, mask, (H, W) = self.exam_groups[idx]
        df_exam = df_exam.copy()
        df_exam["side_view"] = (
            df_exam[LAT_COL].astype(str).str.strip()
            + "-"
            + df_exam[VIEWPOS_COL].astype(str).str.strip()
        )

        cur_imgs = []
        aligned_dicts_by_view = []
        
        valid_prior_indices = set()
        
        # Scan all views to find the maximum valid priors for this exam
        for view_name in VIEW_ORDER:
            df_view = df_exam[df_exam["side_view"] == view_name]
            if len(df_view) > 0:
                row = df_view.iloc[0]
                if "prior_gaps_months" in row.index:
                    gl = _parse_listish(row["prior_gaps_months"])
                    if isinstance(gl, (list, tuple)):
                        for i in range(len(gl)):
                            valid_prior_indices.add(i)

        gaps_lists_by_view = []

        for view_name in VIEW_ORDER:
            df_view = df_exam[df_exam["side_view"] == view_name]
            cur_img = np.zeros((1, H, W), np.float32)
            pri_gaps = []

            if len(df_view) != 0:
                row = df_view.iloc[0]
                cur_path = row[CUR_PNG_COL]
                if isinstance(cur_path, str) and cur_path != "" and os.path.isfile(cur_path):
                    cur_img = self._load_cached(cur_path)
                
                # Load gaps from CSV
                if "prior_gaps_months" in row.index:
                    gl = _parse_listish(row["prior_gaps_months"])
                    if isinstance(gl, (list, tuple)):
                        pri_gaps = [float(x) for x in gl]

            cur_imgs.append(cur_img)
            gaps_lists_by_view.append(pri_gaps)

            # Get raw disk dictionary
            raw_dict = _get_allaligned_dict_for_view(self.allaligned_idx, exam_id, view_name)
            aligned_dicts_by_view.append(raw_dict)

        cur_imgs = np.stack(cur_imgs, axis=0)

        all_prior_idxs = set()
        for d in aligned_dicts_by_view:
            disk_keys = set(d.keys())
            valid_keys = disk_keys.intersection(valid_prior_indices)
            all_prior_idxs |= valid_keys

        prior_idxs_sorted = sorted(all_prior_idxs)
        K = len(prior_idxs_sorted)

        ali_visits = []
        pri_years = []
        has_prior_views = []

        for prior_idx in prior_idxs_sorted:
            visit_imgs = []
            visit_gaps = []
            visit_has = []

            for v in range(4):
                p = aligned_dicts_by_view[v].get(prior_idx, None)
                if p is not None and os.path.isfile(p):
                    visit_imgs.append(self._load_cached(p))
                    visit_has.append(1.0)
                else:
                    visit_imgs.append(np.zeros((1, H, W), np.float32))
                    visit_has.append(0.0)

                gaps_v = gaps_lists_by_view[v]
                if isinstance(gaps_v, list) and len(gaps_v) > prior_idx:
                    visit_gaps.append(gaps_v[prior_idx])

            ali_visits.append(np.stack(visit_imgs, axis=0))
            has_prior_views.append(np.array(visit_has, np.float32))

            if len(visit_gaps) > 0:
                pri_years.append(float(np.mean(visit_gaps)) / 12.0)
            else:
                pri_years.append(0.0)

        ali_imgs = (
            np.stack(ali_visits, axis=0) if K > 0 else np.zeros((0, 4, 1, H, W), np.float32)
        )
        pri_years = np.array(pri_years, np.float32)
        has_prior_views = (
            np.stack(has_prior_views, axis=0) if K > 0 else np.zeros((0, 4), np.float32)
        )

        return (
            torch.from_numpy(cur_imgs),
            torch.from_numpy(ali_imgs),
            torch.from_numpy(pri_years),
            torch.from_numpy(has_prior_views),
            torch.from_numpy(y_event.astype(np.float32)),
            torch.from_numpy(mask.astype(np.float32)),
        )
