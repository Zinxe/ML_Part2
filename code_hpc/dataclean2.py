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


import pandas as pd
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import OneHotEncoder
from sklearn.impute          import SimpleImputer
from sklearn.compose         import ColumnTransformer
from sklearn.pipeline        import Pipeline

# Get top 10 clinical features
# loading
clin = pd.read_csv("../result/clinical2_processed.csv", index_col="Case ID")
y    = (clin["Survival Status"] == "Dead").astype(int)
X    = clin.drop(columns="Survival Status")

num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# RF get features
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), cat_cols)
])

pipe = Pipeline([
    ("prep", pre),
    ("rf",   RandomForestClassifier(
                 n_estimators=200,
                 random_state=42,
                 n_jobs=-1
             ))
])

# train
pipe.fit(X, y)

# get names
ohe_feats = pipe.named_steps["prep"] \
                  .named_transformers_["cat"] \
                  .named_steps["ohe"] \
                  .get_feature_names_out(cat_cols).tolist()

all_feats = num_cols + ohe_feats

# Top 10
importances = pd.Series(
    pipe.named_steps["rf"].feature_importances_,
    index=all_feats
)
top10 = importances.nlargest(10).index.tolist()

print("Top 10", top10)

