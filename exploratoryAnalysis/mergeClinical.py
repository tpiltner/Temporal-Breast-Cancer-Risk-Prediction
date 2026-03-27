import pandas as pd
import boto3
from io import BytesIO
import numpy as np

TEMPORAL_PATH = "temporalSequences.csv"
BUCKET        = "embed-dataset-open"
CLINICAL_KEY  = "tables/EMBED_OpenData_clinical.csv"
OUT_PATH      = "temporalSequences_with_clinical.csv"


def main():
    # 1) Load temporal manifest
    print(f"Reading temporal manifest: {TEMPORAL_PATH}")
    temporal = pd.read_csv(TEMPORAL_PATH)
    print("  Temporal shape:", temporal.shape)

    # 2) Load clinical table from S3
    print(f"\nDownloading clinical table from s3://{BUCKET}/{CLINICAL_KEY}")
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET, Key=CLINICAL_KEY)
    clinical = pd.read_csv(BytesIO(obj["Body"].read()), low_memory=False)
    print("  Clinical shape:", clinical.shape)

    # Prepare 'side' keys 
    temporal = temporal.copy()
    temporal["side"] = (
        temporal["ImageLateralityFinal"]
        .astype(str).str.strip().str.upper()
    )

    clinical = clinical.copy()

    # Preserve true missing values before string normalization
    clinical_side_raw = clinical["side"]

    clinical["side"] = clinical_side_raw.where(clinical_side_raw.notna(), np.nan)
    clinical["side"] = clinical["side"].apply(
        lambda x: x.strip().upper() if isinstance(x, str) else np.nan
    )

    # Duplicate bilateral or missing-side rows to both breasts
    mask_both = clinical["side"].isna() | (clinical["side"] == "B")

    clinical_lr = clinical[~mask_both].copy()      # L / R as-is
    clinical_br = clinical[mask_both].copy()       # B / NaN → L + R

    # Create L and R copies for B / NaN rows
    clinical_br_L = clinical_br.copy()
    clinical_br_L["side"] = "L"

    clinical_br_R = clinical_br.copy()
    clinical_br_R["side"] = "R"

    # Combined clinical table that matches the spec:
    clinical_expanded = pd.concat(
        [clinical_lr, clinical_br_L, clinical_br_R],
        ignore_index=True
    )

    print("\nClinical rows before expand:", len(clinical))
    print("Clinical rows after expand :", len(clinical_expanded))

    # 3) Merge with temporal on empi, acc, side
    key_cols = ["empi_anon", "acc_anon", "side"]
    print("\nMerging on keys:", key_cols)

    merged = temporal.merge(
        clinical_expanded,
        on=key_cols,
        how="left",
        suffixes=("", "_clin")
    )
    print("  Merged shape:", merged.shape)

    # 4) Save
    print(f"\nSaving merged CSV to: {OUT_PATH}")
    merged.to_csv(OUT_PATH, index=False)
    print("Done")


if __name__ == "__main__":
    main()

