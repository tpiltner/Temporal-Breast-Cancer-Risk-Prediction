import pandas as pd
import numpy as np


PATIENT_COL = "empi_anon"
DATE_COL = "current_date"   
ASSES_COL = "asses"
PATH_COL = "path_severity"

T_MAX = 5  # years

# load 
df_all = pd.read_csv("temporalSequences_with_clinical.csv", low_memory=False)
df_all["exam_date"] = pd.to_datetime(df_all[DATE_COL], errors="coerce")
df_all = df_all.dropna(subset=[PATIENT_COL, "exam_date"]).copy()

# require >=1 prior for model
df_all = df_all[df_all["num_priors"] > 0].copy()

# 1) Compute last observed exam date per patient
patient_last = (
    df_all.groupby(PATIENT_COL)["exam_date"]
          .max()
          .rename("last_exam_date")
          .reset_index()
)
df = df_all.merge(patient_last, on=PATIENT_COL, how="left")

# Exam-level observed follow-up (exam -> last imaging date)
df["followup_years_exam"] = (df["last_exam_date"] - df["exam_date"]).dt.days / 365.25

# 2) Define cancer exams + diagnosis date per patient
df[ASSES_COL] = df[ASSES_COL].astype("string")

df["is_cancer_exam"] = (
    df[PATH_COL].isin([0, 1]) |
    (df[ASSES_COL] == "K")  # BI-RADS 6
)

# Diagnosis date: earliest cancer diagnosis
diagnosis_date = (
    df.loc[df["is_cancer_exam"]]
      .groupby(PATIENT_COL)["exam_date"]
      .min()
      .rename("diagnosis_date")
      .reset_index()
)

df = df.merge(diagnosis_date, on=PATIENT_COL, how="left")

# Drop exams after diagnosis (keep only pre-diagnosis “screening history”)
df = df[df["diagnosis_date"].isna() | (df["exam_date"] <= df["diagnosis_date"])].copy()

# Time-to-cancer from each exam (inf if never diagnosed)
df["time_to_cancer_years"] = np.where(
    df["diagnosis_date"].notna(),
    (df["diagnosis_date"] - df["exam_date"]).dt.days / 365.25,
    np.inf
)

# 3) Negative-candidate exams
# N -> BI-RADS 1, B -> BI-RADS 2, A -> BI-RADS 0
df["is_birads_12"] = df[ASSES_COL].isin(["N", "B"])
df["is_birads_0"]  = df[ASSES_COL].eq("A")

df = df.sort_values([PATIENT_COL, "exam_date"])

df["later_has_12"] = (
    df.groupby(PATIENT_COL)["is_birads_12"]
      .transform(lambda x: x[::-1].cummax()[::-1])
)

df["is_negative_candidate"] = (
    df["is_birads_12"] |
    (df["is_birads_0"] & df["later_has_12"])
)

# 4) Cohort filter: 
#    included women who had either:
#      - diagnosis within 5 years of index exam, or
#      - imaging follow-up for at least 5 years from index exam"

df["has_5y_observed_followup_from_exam"] = df["followup_years_exam"] >= T_MAX
df["has_cancer_within_5y_from_exam"] = (df["time_to_cancer_years"] > 0) & (df["time_to_cancer_years"] <= T_MAX)

df_risk = df[df["has_5y_observed_followup_from_exam"] | df["has_cancer_within_5y_from_exam"]].copy()

# 5) Create 1..5-year labels
#    - Positives: 0 < ttc <= h
#    - Negatives: negative candidate and (ttc > 5 or never cancer)

for h in range(1, 6):
    df_risk[f"risk_{h}y_pos"] = (
        (df_risk["time_to_cancer_years"] > 0) &
        (df_risk["time_to_cancer_years"] <= h)
    )

    # horizon-specific negatives:
    df_risk[f"risk_{h}y_neg"] = (
        df_risk["is_negative_candidate"] &
        (df_risk["time_to_cancer_years"] > h) &
        (df_risk["followup_years_exam"] >= h)   # observed long enough to certify NEG
    )

# 6) Save
print("Before filter (all pre-dx exams):", df.shape[0], "rows,", df[PATIENT_COL].nunique(), "patients")
print("After exam-level 5y filter:", df_risk.shape[0], "rows,", df_risk[PATIENT_COL].nunique(), "patients")

for h in range(1, 6):
    print(f"{h}y pos:", int(df_risk[f"risk_{h}y_pos"].sum()),
          f"{h}y neg:", int(df_risk[f"risk_{h}y_neg"].sum()))

df_risk.to_csv("temporalSequences_riskcohort_5y_examlevel.csv", index=False)
print("Saved: temporalSequences_riskcohort_5y_examlevel.csv")
