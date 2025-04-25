import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

DATA_DIR = "/user/home/ms13525/scratch/mshds-ml-data-2025"

# Rnaseq
# Read in and transpose
fn_in  = os.path.join(DATA_DIR, "dataset2", "rnaseq.txt")
fn_out = "../result/rnaseq_processed.csv"
os.makedirs(os.path.dirname(fn_out), exist_ok=True)

df = pd.read_csv(fn_in, sep="\t", index_col=0)
df = df.T

# Force to float, non-value to NaN
df = df.apply(pd.to_numeric, errors="coerce")

# Discard genes missing in more than 50% of the samples
missing_frac = df.isna().mean(axis=0)
keep_genes   = missing_frac[missing_frac < 0.5].index
df = df[keep_genes]
df = df.fillna(0)

# CPM + log1p
counts = df
cpm    = counts.div(counts.sum(axis=1), axis=0) * 1e6
log_cpm = np.log1p(cpm)

# Retain the top 150 most variant genes
var = log_cpm.var(axis=0)
top_genes = var.sort_values(ascending=False).head(150).index
df_sel   = log_cpm[top_genes]

# Z-score standardization
scaler    = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_sel),
    index=df_sel.index,
    columns=df_sel.columns
)

# Save result
df_scaled.to_csv(fn_out)
print("Completed,saved in:", fn_out)

# clinical2
# Loading
clin = pd.read_csv(os.path.join(DATA_DIR, "dataset2", "clinical2.csv"))

# Clear useless variables
drop_cols = [
    "Time to Death (days)", "Date of Death", "Date of Last Known Alive",
    "Date of Recurrence", "Recurrence", "Recurrence Location",
    "CT Date", "PET Date"
]
clin = clin.drop(columns=drop_cols)
clin = clin.set_index("Case ID")

# Clarifying the numerical column names
num_cols = [
    "Age at Histological Diagnosis", "Weight (lbs)",
    "Pack Years", "Quit Smoking Year", "Days between CT and surgery"
]
clin[num_cols] = clin[num_cols].apply(
    lambda s: pd.to_numeric(s, errors="coerce")
)

# Fill numerical missing data：median
for c in num_cols:
    med = clin[c].median()
    clin[c] = clin[c].fillna(med)

# Fill class missing data：mode
cat_cols = clin.columns.difference(num_cols + ["Survival Status"])
for c in cat_cols:
    mode = clin[c].mode(dropna=True)
    if not mode.empty:
        clin[c] = clin[c].fillna(mode[0])

# Save result
os.makedirs("../result", exist_ok=True)
clin.to_csv("../result/clinical2_processed.csv")

