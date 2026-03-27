"""
1) Loads censored exam-level CSV
   (temporalSequences_riskcohort_5y_examlevel_censored.csv)
2) Preprocesses all unique DICOMs referenced by:
   - current_path   (required)
   - prior_path / prior_paths 
   using existing `preprocessing.preprocess_single_dicom()`
3) Writes PNGs to a UID-based folder:
   - OUT_PNG_DIR/current_uid/<UID>.png
   - OUT_PNG_DIR/prior_uid/<UID>.png
4) Adds cur_png (+ pri_png if applicable) columns to the censored dataframe
5) Filters to rows that have cur_png (so an image model can train)
6) Saves a final CSV to pass through splitDataset.py

"""

from __future__ import annotations

import os
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import cv2
from tqdm import tqdm
from preprocessing import preprocess_single_dicom  


@dataclass
class Config:
    censored_csv: Path = Path(
        "/local/scratch/tpiltne/temporalSequences_riskcohort_5y_examlevel.csv"
    )

    # Where downloaded DICOM is
    dicom_root: Path = Path("/local/scratch/tpiltne/embed_cohortSubset")

    # Output root for UID-based PNGs
    out_png_dir: Path = Path("/local/scratch/tpiltne/utils/preproc_censored_all_uid")

    # Final output CSV 
    out_csv: Path = Path("/local/scratch/tpiltne/utils/trainingWithPNGS.csv")

    # Column names
    patient_col: str = "empi_anon"
    current_path_col: str = "current_path"

    prior_path_col: str = "prior_path"
    prior_paths_col: str = "prior_paths"

    # Missing DICOM report
    missing_report: Path = Path("/local/scratch/tpiltne/utils/missing_dicoms_censored_report.txt")


CFG = Config()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def s3_to_local(s3_url: str, root: Path) -> Path:
    """
    Convert s3://bucket/REL/PATH/file.dcm -> root/REL/PATH/file.dcm
    """
    if not isinstance(s3_url, str):
        raise ValueError(f"Expected string, got {type(s3_url)}")
    s = s3_url.strip()
    if not s.startswith("s3://"):
        return Path(s)
    rel = s[len("s3://") :].split("/", 1)[1]
    return root / rel


def uid_from_path(path_str: str) -> str:
    """Use the DICOM filename UID (last component), strip .dcm ."""
    name = str(path_str).split("/")[-1]
    if name.lower().endswith(".dcm"):
        name = name[:-4]
    return name


def parse_list_maybe(val) -> List[str]:
    """
    prior_paths sometimes stored as "['s3://...','s3://...']".
    """
    if isinstance(val, list):
        return [str(x) for x in val if isinstance(x, (str, Path))]
    if pd.isna(val):
        return []
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                out = ast.literal_eval(s)
                if isinstance(out, (list, tuple)):
                    return [str(x) for x in out if isinstance(x, str)]
            except Exception:
                return []
        return [s]
    return []


def preprocess_one_to_uid_png(
    s3_path: str,
    dicom_root: Path,
    out_dir: Path,
) -> Tuple[str, str]:
    """
    Preprocess one DICOM to a UID-based PNG.
    Returns (s3_path, png_path or "").
    """
    uid = uid_from_path(s3_path)
    out_png = out_dir / f"{uid}.png"

    if out_png.exists():
        return s3_path, str(out_png)

    local = s3_to_local(s3_path, dicom_root)
    if not local.exists():
        return s3_path, ""

    try:
        img16 = preprocess_single_dicom(local)  # uint16 HxW
        ok = cv2.imwrite(str(out_png), img16)
        if not ok:
            return s3_path, ""
        return s3_path, str(out_png)
    except Exception:
        return s3_path, ""


def main():
    print("Loading censored CSV:", CFG.censored_csv)
    df = pd.read_csv(CFG.censored_csv, low_memory=False)

    if CFG.patient_col not in df.columns:
        raise KeyError(f"Missing required patient column: {CFG.patient_col}")
    if CFG.current_path_col not in df.columns:
        raise KeyError(f"Missing required current_path column: {CFG.current_path_col}")

    # Prepare output dirs
    cur_dir = CFG.out_png_dir / "current_uid"
    pri_dir = CFG.out_png_dir / "prior_uid"
    ensure_dir(cur_dir)
    ensure_dir(pri_dir)

    # Detect prior schema
    has_prior_path = CFG.prior_path_col in df.columns
    has_prior_paths = CFG.prior_paths_col in df.columns
    print(f"Detected prior_path col: {has_prior_path}, prior_paths col: {has_prior_paths}")

    # 1) Preprocess all unique current DICOMs
    cur_paths = df[CFG.current_path_col].dropna().astype(str).unique().tolist()
    print("Unique current DICOMs:", len(cur_paths))

    cur_map: Dict[str, str] = {}
    missing_list: List[str] = []

    for s3 in tqdm(cur_paths, desc="Preprocessing current DICOMs"):
        s3_key, png = preprocess_one_to_uid_png(s3, CFG.dicom_root, cur_dir)
        cur_map[s3_key] = png
        if png == "":
            missing_list.append(f"CURRENT\t{s3_key}")

    # 2) Preprocess all unique prior DICOMs 
    pri_map: Dict[str, str] = {}

    if has_prior_path or has_prior_paths:
        prior_all: List[str] = []

        if has_prior_path:
            prior_all.extend(df[CFG.prior_path_col].dropna().astype(str).tolist())

        if has_prior_paths:
            for val in df[CFG.prior_paths_col].values:
                prior_all.extend(parse_list_maybe(val))

        prior_all = sorted({p for p in prior_all if isinstance(p, str) and p.strip()})
        print("Unique prior DICOMs:", len(prior_all))

        for s3 in tqdm(prior_all, desc="Preprocessing prior DICOMs"):
            s3_key, png = preprocess_one_to_uid_png(s3, CFG.dicom_root, pri_dir)
            pri_map[s3_key] = png
            if png == "":
                missing_list.append(f"PRIOR\t{s3_key}")

    # missing report
    ensure_dir(CFG.missing_report.parent)
    CFG.missing_report.write_text("\n".join(missing_list))
    print("Wrote missing report:", CFG.missing_report, "lines:", len(missing_list))

    # 3) Attach cur_png / pri_png columns
    df["cur_png"] = df[CFG.current_path_col].astype(str).map(cur_map).fillna("")

    if has_prior_path:
        df["pri_png"] = df[CFG.prior_path_col].astype(str).map(pri_map).fillna("")
    elif has_prior_paths:
        def map_prior_list(val):
            s3_list = parse_list_maybe(val)
            png_list = [pri_map.get(s3, "") for s3 in s3_list]
            png_list = [p for p in png_list if p] 
            return png_list

        df["pri_png"] = df[CFG.prior_paths_col].apply(map_prior_list)
    else:
        df["pri_png"] = ""  

    # 4) Keep only rows with images (for an image model)
    before = len(df)
    df_img = df[df["cur_png"].notna() & (df["cur_png"] != "")].copy()
    after = len(df_img)

    print(f"Rows total: {before}")
    print(f"Rows with cur_png: {after} ({after/before:.2%})")

    # Enforce the file actually exists on disk
    df_img = df_img[df_img["cur_png"].apply(lambda p: isinstance(p, str) and os.path.isfile(p))].copy()
    print("Rows with cur_png that exists:", len(df_img))

    # 5) Save final CSV 
    ensure_dir(CFG.out_csv.parent)
    df_img.to_csv(CFG.out_csv, index=False)
    print("Saved final dataset CSV (no split):", CFG.out_csv)

    fu_col = None
    for cand in ["followup_years_exam", "followup_years"]:
        if cand in df_img.columns:
            fu_col = cand
            break

    if fu_col:
        fu = pd.to_numeric(df_img[fu_col], errors="coerce")
        print(f"{fu_col} min/median/max:", float(fu.min()), float(fu.median()), float(fu.max()))
    else:
        print("No followup_years_* column found in output; masking will fall back to pos/neg or all-ones.")


if __name__ == "__main__":
    main()
