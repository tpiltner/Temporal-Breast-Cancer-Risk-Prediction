import ast
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm

import matplotlib.pyplot as plt


# Risk cohort 
MANIFEST_CSV = "temporalSequences_riskcohort_5y.csv"

# Where all DICOMs were downloaded by download_from_manifest.py
DICOM_ROOT = Path("/local/scratch/tpiltne/embed_cohortSubset")

# utils folder for outputs
UTILS_ROOT = Path("/local/scratch/tpiltne/utils")

# Where to save preprocessed images
OUT_ROOT = UTILS_ROOT / "preproc"

# Target size 
TARGET_W = 1664
TARGET_H = 2048

# test on first N rows (None b/c not testing)
LIMIT_ROWS = None  


def s3_to_local(s3_url: str, root: Path = DICOM_ROOT) -> Path:
    """
    Convert an S3-style path from the manifest into a local path
    under embed_cohortSubset.

    Works for any bucket name, e.g.:
        s3://embed-open-data/embd/.../file.dcm
        s3://embed-dataset-open/images/.../file.dcm

    /local/scratch/tpiltne/embed_cohortSubset/embd/.../file.dcm
    /local/scratch/tpiltne/embed_cohortSubset/images/.../file.dcm
    """
    if not isinstance(s3_url, str):
        raise ValueError(f"Expected string S3 url, got {type(s3_url)}")

    if not s3_url.startswith("s3://"):
        # already looks like a local path
        return Path(s3_url)

    # Strip "s3://" and split into bucket + relative path
    without_scheme = s3_url[len("s3://"):]
    parts = without_scheme.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected S3 url format: {s3_url}")
    bucket, rel = parts[0], parts[1]

    return root / rel

def parse_prior_paths(val):
    """
    prior_paths is usually stored as a string representation of a list,
    e.g. "['s3://...','s3://...']". Convert to Python list of strings.

    Returns [] if parsing fails.
    """
    if isinstance(val, list):
        return val
    if pd.isna(val):
        return []
    if isinstance(val, str):
        try:
            out = ast.literal_eval(val)
            if isinstance(out, list):
                return out
        except Exception:
            pass
    return []


#Dicom loading and preprocessing

def load_dicom_image_local(
    path: Path,
    apply_window: bool = True,
    normalize: bool = True,
    flip_right: bool = True,
):
    """
    Load a DICOM and apply:
      - raw pixel read
      - rescale slope / intercept
      - VOI LUT windowing if available
      - MONOCHROME1 inversion
      - right-breast horizontal flip
      - normalization to [0,1]
    """
    ds = pydicom.dcmread(str(path))

    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    if apply_window:
        try:
            arr = apply_voi_lut(arr, ds).astype(np.float32)
        except Exception:
            pass

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = np.max(arr) - arr

    if flip_right:
        laterality = str(getattr(ds, "ImageLaterality", "")).upper()
        if laterality in {"R", "RIGHT"}:
            arr = np.fliplr(arr)

    if normalize:
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)

    return arr.astype(np.float32), ds


def crop_to_mask(image_u8: np.ndarray, mask: np.ndarray, pad: int = 10) -> np.ndarray:
    """Crop around the non-zero part of the mask ."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return image_u8  # fallback: no crop
    
    y0 = max(0, ys.min() - pad)
    y1 = min(image_u8.shape[0], ys.max() + pad + 1)
    x0 = max(0, xs.min() - pad)
    x1 = min(image_u8.shape[1], xs.max() + pad + 1)
    return image_u8[y0:y1, x0:x1]

def preprocess_single_dicom(dicom_path: Path) -> np.ndarray:
    """
    Preprocessing for one DICOM:
      - Load + rescale + VOI LUT + invert MONOCHROME1 + flip right breast
      - Convert to 8-bit
      - Find all contours, keep largest -> breast mask
      - Isolate breast region
      - Resize to 1664x2048 with ratio preserved
      - Normalize and convert to 16-bit

    Returns:
      uint16 array of shape (TARGET_H, TARGET_W)
    """
    # Using existing loader (normalize=True to get [0,1])
    img01, _ = load_dicom_image_local(dicom_path, normalize=True)

    # 8-bit for contour detection
    img_u8 = (img01 * 255).astype(np.uint8)

    # Largest contour mask (breast)
    mask = largest_contour_mask(img_u8)

    if np.count_nonzero(mask) == 0:
        breast = img_u8
    else:
        masked = cv2.bitwise_and(img_u8, mask)
        breast = crop_to_mask(masked, mask, pad=10)

    # Resize to 1664x2048 
    breast_resized = resize_preserve_aspect(breast, TARGET_H, TARGET_W)

    # Normalize to [0,1] again (after resize) and convert to 16-bit
    breast_resized = breast_resized.astype(np.float32)
    mn, mx = breast_resized.min(), breast_resized.max()
    if mx > mn:
        breast_resized = (breast_resized - mn) / (mx - mn)
    else:
        breast_resized[:] = 0.0

    img16 = (breast_resized * 65535.0).astype(np.uint16)
    return img16


def largest_contour_mask(img_u8: np.ndarray) -> np.ndarray:
    """
    Find all contours on an 8-bit image and keep only the largest one.
    Returns a binary mask (uint8, 0/255).
    """
    # Binary image: non-zero pixels become foreground
    _, binary = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        # no contour -> empty mask
        return np.zeros_like(img_u8, dtype=np.uint8)

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(img_u8, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask


def resize_preserve_aspect(image: np.ndarray,
                           target_h: int = TARGET_H,
                           target_w: int = TARGET_W) -> np.ndarray:
    """
    Resize image to fit inside (target_h, target_w) while preserving aspect ratio.
    Pad with zeros to the exact target size.
    """
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w), dtype=resized.dtype)

    # center in both directions
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas

def parse_str_list(val):
    """Parse a string like "['2015-01-01','2013-10-10']" into a list of strings."""
    if isinstance(val, list):
        return val
    if pd.isna(val):
        return []
    if isinstance(val, str):
        try:
            out = ast.literal_eval(val)
            if isinstance(out, list):
                return out
        except Exception:
            pass
    return []


def parse_float_list(val):
    """Parse a string list into list of floats (for gaps in months)."""
    raw = parse_str_list(val)
    out = []
    for x in raw:
        try:
            out.append(float(x))
        except Exception:
            out.append(np.nan)
    return out


def demo_one_example(row_idx: int = 0, prior_idx: int = 0):
    """
    Visualize how preprocessing changes one current exam and one prior.

    Top row:  raw (windowed + oriented) current / prior
    Bottom row: preprocessed 16-bit outputs (normalized for display)
    Dates and months-back are read from the manifest:
      - current_date  (scalar string)
      - prior_dates   (list-like column)
      - prior_gaps_months (list-like column, months back)
    """
    # load one row from manifest
    df = pd.read_csv(MANIFEST_CSV, low_memory=False)
    row = df.iloc[row_idx]

    cur_local = s3_to_local(row["current_path"])
    pri_list = parse_prior_paths(row.get("prior_paths", "[]"))

    if not pri_list:
        print(f"Row {row_idx} has no priors.")
        return
    if prior_idx >= len(pri_list):
        print(f"Row {row_idx} has only {len(pri_list)} priors, got prior_idx={prior_idx}.")
        return

    pri_local = s3_to_local(pri_list[prior_idx])

    if not cur_local.exists():
        print(f"Missing current DICOM: {cur_local}")
        return
    if not pri_local.exists():
        print(f"Missing prior DICOM: {pri_local}")
        return

    # load raw (windowed, oriented, [0,1])
    cur_raw, cur_ds = load_dicom_image_local(cur_local, normalize=True)
    pri_raw, pri_ds = load_dicom_image_local(pri_local, normalize=True)

    # preprocess 
    cur_proc16 = preprocess_single_dicom(cur_local)   # uint16, 1664x2048
    pri_proc16 = preprocess_single_dicom(pri_local)   # uint16, 1664x2048

    # for display in matplotlib, scale 16-bit back to [0,1]
    cur_proc = cur_proc16.astype(np.float32) / 65535.0
    pri_proc = pri_proc16.astype(np.float32) / 65535.0

    # metadata from DICOM (view / laterality) 
    cur_lat = getattr(cur_ds, "ImageLaterality", "?")
    cur_view = getattr(cur_ds, "ViewPosition", "")
    pri_lat = getattr(pri_ds, "ImageLaterality", "?")
    pri_view = getattr(pri_ds, "ViewPosition", "")

    # metadata from manifest (dates + gaps) 
    cur_date_str = row.get("current_date", "")
    if not isinstance(cur_date_str, str):
        cur_date_str = ""

    # prior_dates: list-like string column -> list[str]
    pri_dates_list = parse_str_list(row.get("prior_dates", "[]"))
    pri_date_str = ""
    if prior_idx < len(pri_dates_list) and isinstance(pri_dates_list[prior_idx], str):
        pri_date_str = pri_dates_list[prior_idx]

    # prior_gaps_months: list-like string column -> list[float]
    pri_gaps_list = parse_float_list(row.get("prior_gaps_months", "[]"))
    months_back = None
    if prior_idx < len(pri_gaps_list):
        gap_val = pri_gaps_list[prior_idx]
        if not (isinstance(gap_val, float) and np.isnan(gap_val)):
            months_back = float(gap_val)

    #  plotting
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Current raw 
    ax = axes[0, 0]
    ax.imshow(cur_raw, cmap="gray")
    ax.axis("off")
    ax.set_title(f"Current — {cur_lat}-{cur_view} (raw)")

    # Prior raw 
    ax = axes[0, 1]
    ax.imshow(pri_raw, cmap="gray")
    ax.axis("off")
    ax.set_title(f"Prior {prior_idx+1} — {pri_lat}-{pri_view} (raw)")

    #  Current preprocessed
    ax = axes[1, 0]
    ax.imshow(cur_proc, cmap="gray")
    ax.axis("off")
    ax.set_title(f"Current — {cur_lat}-{cur_view} (preprocessed)")
    if cur_date_str:
        ax.text(
            0.01, 0.02,
            f"Date: {cur_date_str}",
            transform=ax.transAxes,
            color="white",
            fontsize=10,
            ha="left",
            va="bottom",
            bbox=dict(facecolor="black", alpha=0.6, pad=2),
        )

    # Prior preprocessed 
    ax = axes[1, 1]
    ax.imshow(pri_proc, cmap="gray")
    ax.axis("off")

    if months_back is not None:
        title = f"Prior {prior_idx+1} — {months_back:.1f} mo back"
    else:
        title = f"Prior {prior_idx+1}"
    if pri_view:
        title += f" ({pri_lat}-{pri_view})"
    title += " (preprocessed)"
    ax.set_title(title)

    if pri_date_str:
        ax.text(
            0.01, 0.02,
            f"Date: {pri_date_str}",
            transform=ax.transAxes,
            color="white",
            fontsize=10,
            ha="left",
            va="bottom",
            bbox=dict(facecolor="black", alpha=0.6, pad=2),
        )

    #  save + show 
    plt.tight_layout()
    UTILS_ROOT.mkdir(parents=True, exist_ok=True)
    out_fig = UTILS_ROOT / f"demo_row{row_idx:04d}_prior{prior_idx}.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"Saved demo figure to {out_fig}")
    plt.show()


# loop over all images 

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST_CSV, low_memory=False)
    if LIMIT_ROWS is not None:
        df = df.iloc[:LIMIT_ROWS].copy()

    cur_out_paths: list[str] = []
    pri_out_paths: list[list[str]] = []

    print(f"Total rows to preprocess: {len(df)}")

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        # Current image 
        cur_out = OUT_ROOT / f"row{idx:06d}_current.png"

        if cur_out.exists():
            # already preprocessed this current image
            cur_out_paths.append(str(cur_out))
        else:
            cur_s3 = row["current_path"]
            try:
                cur_local = s3_to_local(cur_s3)
            except Exception as e:
                print(f"[ERROR] row {idx} bad current_path '{cur_s3}': {e}")
                cur_out_paths.append("")
                pri_out_paths.append([])
                continue

            if not cur_local.exists():
                print(f"[WARN] Missing current DICOM: {cur_local}")
                cur_out_paths.append("")
            else:
                try:
                    img16 = preprocess_single_dicom(cur_local)
                    cv2.imwrite(str(cur_out), img16)   # 16-bit PNG
                    cur_out_paths.append(str(cur_out))
                except Exception as e:
                    print(f"[ERROR] Failed on current {cur_local}: {e}")
                    cur_out_paths.append("")

        # Prior images 
        this_pri_outs: list[str] = []
        pri_list = parse_prior_paths(row.get("prior_paths", "[]"))

        for j, pri_s3 in enumerate(pri_list):
            pri_out = OUT_ROOT / f"row{idx:06d}_prior{j+1}.png"
            if pri_out.exists():
                this_pri_outs.append(str(pri_out))
                continue

            try:
                pri_local = s3_to_local(pri_s3)
            except Exception as e:
                print(f"[ERROR] row {idx} bad prior_path '{pri_s3}': {e}")
                continue

            if not pri_local.exists():
                print(f"[WARN] Missing prior DICOM: {pri_local}")
                continue

            try:
                img16 = preprocess_single_dicom(pri_local)
                cv2.imwrite(str(pri_out), img16)
                this_pri_outs.append(str(pri_out))
            except Exception as e:
                print(f"[ERROR] Failed on prior {pri_local}: {e}")

        pri_out_paths.append(this_pri_outs)

    # Attach output paths and save a new CSV
    df["cur_png"] = cur_out_paths
    df["pri_png"] = pri_out_paths

    out_csv = UTILS_ROOT / "temporalSequences_riskcohort_5y_with_preproc_paths.csv"
    df.to_csv(out_csv, index=False)
    print(f"Done. Saved updated manifest with preprocessed paths to: {out_csv}")


if __name__ == "__main__":
    main()
    #demo_one_example(row_idx=0, prior_idx=0)