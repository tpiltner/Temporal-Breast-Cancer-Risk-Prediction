import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

df = pd.read_csv("temporalSequences_riskcohort_5y_examlevel", low_memory=False)

print(df.head())

# If risk columns are present as 0/1, convert to bool 
for h in range(1, 6):
    for kind in ["pos", "neg"]:
        col = f"risk_{h}y_{kind}"
        if col in df.columns:
            df[col] = df[col].astype(bool)


# 1. Overall ≥5y follow-up cohort stats
n_patients_5y = df["empi_anon"].nunique()
n_exams_5y    = len(df)
n_images_5y   = df["current_path"].nunique()   # unique DICOM images

print("≥5y follow-up cohort (all exams in df):")
print("  Patients:", n_patients_5y)
print("  Exams (rows):", n_exams_5y)
print("  Images (unique current_path):", n_images_5y)

print("Unique exams by acc_anon:", df["acc_anon"].nunique())
print("Unique exam-date pairs:", df[["empi_anon", "acc_anon", "current_date"]].drop_duplicates().shape[0])

print("max num_priors:", df["num_priors"].max())
print("unique num_priors:", sorted(df["num_priors"].unique()))


all_paths = set(df["current_path"])
for s in df["prior_paths"]:
    try:
        paths = ast.literal_eval(s)
        all_paths.update(paths)
    except Exception:
        pass

print("unique current images:", df["current_path"].nunique())
print("unique current + prior images:", len(all_paths))


def min_positive_ttc(s):
    # s is the time_to_cancer_years for one patient
    mask = (s > 0) & np.isfinite(s)
    if mask.any():
        return s[mask].min()
    else:
        return np.inf

patient_ttc = (
    df.groupby("empi_anon")["time_to_cancer_years"]
      .apply(min_positive_ttc)
      .rename("ttc_min")
      .to_frame()
)

def classify_patient(ttc):
    if 0 < ttc <= 5:
        return "positive_5y_patient"
    else:
        return "negative_5y_patient"

patient_ttc["patient_group"] = patient_ttc["ttc_min"].apply(classify_patient)

print("\nPatients with ≥5y follow-up grouped by 5-year outcome:")
print(patient_ttc["patient_group"].value_counts())

# Merge patient group back to exam-level df
df = df.merge(
    patient_ttc[["patient_group"]],
    left_on="empi_anon",
    right_index=True,
    how="left"
)

# 3. Exam- and image-level counts for cohort-positive vs cohort-negative
pos_cohort = df[df["patient_group"] == "positive_5y_patient"].copy()
neg_cohort = df[df["patient_group"] == "negative_5y_patient"].copy()

# Positive cohort
pos_patients = pos_cohort["empi_anon"].nunique()
pos_exams    = len(pos_cohort)
pos_images   = pos_cohort["current_path"].nunique()

print("\nCohort POSITIVE (patients with cancer within 5 years):")
print("  Patients:", pos_patients)
print("  Exams:", pos_exams)
print("  Images:", pos_images)

# Negative cohort
neg_patients = neg_cohort["empi_anon"].nunique()
neg_exams    = len(neg_cohort)
neg_images   = neg_cohort["current_path"].nunique()

print("\nCohort NEGATIVE (patients without cancer within 5 years):")
print("  Patients:", neg_patients)
print("  Exams:", neg_exams)
print("  Images:", neg_images)

# 4. Distribution of number of prior exams (full ≥5y cohort)
labeled_5y = df[df["risk_5y_pos"] | df["risk_5y_neg"]].copy()

plt.figure(figsize=(6,4))
sns.histplot(
    data=labeled_5y,
    x="num_priors",
    hue=np.where(labeled_5y["risk_5y_pos"], "risk_5y_pos", "risk_5y_neg"),
    multiple="dodge",
    bins=range(1, df["num_priors"].max() + 2),
    discrete=True
)
plt.xlabel("Number of prior exams")
plt.ylabel("Number of exams")
plt.title("Num priors for labeled exams: risk_5y_pos vs risk_5y_neg")
plt.tight_layout()
plt.show()

# 5. Violin: num_priors vs patient-level outcome (exam-level view)
exam_cohort = df[df["patient_group"].isin(["positive_5y_patient", "negative_5y_patient"])].copy()

plt.figure(figsize=(6, 4))
sns.violinplot(
    data=exam_cohort,
    x="patient_group",
    y="num_priors",
    cut=0
)
plt.xlabel("Patient group")
plt.ylabel("Number of prior exams")
plt.title("Number of priors for exams from\npatients with vs without cancer in 5 years")
plt.tight_layout()
plt.show()

# 5. Histogram: num_priors vs patient-level outcome (exam-level view)
exam_cohort = df[df["patient_group"].isin(
    ["positive_5y_patient", "negative_5y_patient"]
)].copy()

plt.figure(figsize=(6, 4))
sns.histplot(
    data=exam_cohort,
    x="num_priors",
    hue="patient_group",
    multiple="dodge",               
    bins=range(1, df["num_priors"].max() + 2),
    discrete=True
)
plt.xlabel("Number of prior exams")
plt.ylabel("Number of exams")
plt.title("Number of priors for exams from\npatients with vs without cancer in 5 years")
plt.tight_layout()
plt.show()

# 6. Bar chart: exam counts by horizon and label (1–5 years)
records = []
for h in range(1, 6):
    pos_count = df[f"risk_{h}y_pos"].sum()
    neg_count = df[f"risk_{h}y_neg"].sum()
    records.append({"horizon": f"{h}y", "label": "positive", "count": pos_count})
    records.append({"horizon": f"{h}y", "label": "negative", "count": neg_count})

risk_counts = pd.DataFrame(records)

plt.figure(figsize=(7, 4))
sns.barplot(
    data=risk_counts,
    x="horizon",
    y="count",
    hue="label"
)
plt.xlabel("Risk horizon")
plt.ylabel("Number of exams")
plt.title("Exam counts by risk horizon and label")
plt.tight_layout()
plt.show()

# 7. Patient-level bar chart: ≥5y follow-up, positive vs negative
patient_counts = patient_ttc["patient_group"].value_counts().sort_index()

plt.figure(figsize=(5, 4))
plt.bar(patient_counts.index, patient_counts.values)
plt.ylabel("Number of patients")
plt.xlabel("Patient group")
plt.title("Patients with ≥5y follow-up:\nCancer within 5 years vs no cancer in 5 years")
for x, c in zip(patient_counts.index, patient_counts.values):
    plt.text(x, c, str(c), ha="center", va="bottom")
plt.tight_layout()
plt.show()

# 8. Time-to-cancer distribution for exams from cohort-positive patients
# For exams from patients who DO get cancer within 5 years,
# look at time_to_cancer distribution for exams that are actually pre-cancer.
pos_exams_pre_cancer = exam_cohort[
    (exam_cohort["patient_group"] == "positive_5y_patient") &
    (exam_cohort["time_to_cancer_years"] > 0) &
    (exam_cohort["time_to_cancer_years"] <= 5)
].copy()

if not pos_exams_pre_cancer.empty:
    plt.figure(figsize=(6, 4))
    sns.histplot(pos_exams_pre_cancer["time_to_cancer_years"], bins=20)
    plt.xlabel("Time to cancer (years)")
    plt.ylabel("Number of exams")
    plt.title("Time-to-cancer distribution\nfor exams in positive patients (0–5y)")
    plt.tight_layout()
    plt.show()
else:
    print("No pre-cancer exams found; skipping Time-to-Cancer plot.")

# 9. Distribution of follow-up period per patient

# full temporal with clinical
temp = pd.read_csv("temporalSequences_with_clinical.csv",
                   parse_dates=["current_date"], low_memory=False)

# risk cohort 
risk = pd.read_csv("temporalSequences_riskcohort_5y.csv", low_memory=False)
risk_patients = risk["empi_anon"].unique()

# restrict to those patients
temp = temp[temp["empi_anon"].isin(risk_patients)].copy()

def parse_prior_dates(s):
    try:
        dates = ast.literal_eval(s)
        if not dates:
            return pd.NaT
        return pd.to_datetime(dates)
    except Exception:
        return pd.NaT

temp["prior_dates_list"] = temp["prior_dates"].apply(parse_prior_dates)
temp["earliest_prior"] = temp["prior_dates_list"].apply(
    lambda x: x.min() if isinstance(x, (pd.DatetimeIndex, pd.Series)) else pd.NaT
)

per_patient = (
    temp.groupby("empi_anon")[["earliest_prior", "current_date"]]
        .agg({"earliest_prior": "min", "current_date": "max"})
        .rename(columns={"current_date": "last_exam"})
)

per_patient["followup_years"] = (
    (per_patient["last_exam"] - per_patient["earliest_prior"]).dt.days / 365.25
)

bins   = [-0.01, 0, 1, 2, 3, 4, 5, 6, 7, 8]
labels = ["≤ 0", "(0, 1]", "(1, 2]", "(2, 3]", "(3, 4]",
          "(4, 5]", "(5, 6]", "(6, 7]", "(7, 8]"]

per_patient["followup_bin"] = pd.cut(
    per_patient["followup_years"], bins=bins, labels=labels, include_lowest=True
)

counts = per_patient["followup_bin"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(counts.index.astype(str), counts.values)
plt.title("Distribution of Follow-up Period per Patient\n(Temporal Subset, Risk-Cohort Patients)")
plt.xlabel("Years of Follow-up")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()

