from pathlib import Path
import numpy as np
import pandas as pd

CSV_PATH = Path("/local/scratch/tpiltne/utils/trainingWithPNGS.csv")

PATIENT_COL = "empi_anon"
CUR_PNG_COL = "cur_png"

TRAIN_FRAC = 0.65
VAL_FRAC   = 0.15
TEST_FRAC  = 0.20
RANDOM_SEED = 42


def split_patient_ids(patient_ids, train_frac=0.65, val_frac=0.15, seed=42):
    # split into train/val/test
    patient_ids = np.array(patient_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ids = set(patient_ids[:n_train])
    val_ids   = set(patient_ids[n_train:n_train + n_val])
    test_ids  = set(patient_ids[n_train + n_val:])

    return train_ids, val_ids, test_ids


def assign_split(patient_id, train_ids, val_ids, test_ids):
    # Return the split for one patient
    if patient_id in train_ids:
        return "train"
    if patient_id in val_ids:
        return "val"
    if patient_id in test_ids:
        return "test"
    return "unassigned"


def main():
    df = pd.read_csv(CSV_PATH, low_memory=False)

    # Check required columns
    for col in [PATIENT_COL, CUR_PNG_COL]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # Keep only rows with an image
    df = df[df[CUR_PNG_COL].notna() & (df[CUR_PNG_COL] != "")].copy()
    if df.empty:
        raise RuntimeError("No rows with cur_png found.")

    # Unique patients
    patient_ids = df[PATIENT_COL].dropna().unique()
    print(f"Total rows with images: {len(df)}")
    print(f"Total unique patients: {len(patient_ids)}")

    # Split patients
    train_ids, val_ids, test_ids = split_patient_ids(
        patient_ids,
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        seed=RANDOM_SEED
    )

    # ensure no patient leakage
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # Assign every exam row based on patient ID
    df["split"] = df[PATIENT_COL].apply(
        lambda pid: assign_split(pid, train_ids, val_ids, test_ids)
    )

    # Print summary
    print("\nPatient counts by split:")
    print(f"train: {len(train_ids)}")
    print(f"val:   {len(val_ids)}")
    print(f"test:  {len(test_ids)}")

    print("\nRow counts by split:")
    print(df["split"].value_counts())

    # Save
    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved updated CSV to: {CSV_PATH}")


if __name__ == "__main__":
    main()