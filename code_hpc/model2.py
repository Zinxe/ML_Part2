import os
import pandas as pd
import numpy as np

from sklearn.compose         import ColumnTransformer
from sklearn.pipeline        import Pipeline
from sklearn.impute          import SimpleImputer
from sklearn.preprocessing   import OneHotEncoder, StandardScaler
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

DATA_DIR = "/user/home/ms13525/scratch/mshds-ml-data-2025"

# Read feature matrix
df_img  = pd.read_csv("../result/features2.csv",       index_col="patient_id") 
df_rna  = pd.read_csv("../result/rnaseq_processed.csv", index_col=0)
df_clin = pd.read_csv("../result/clinical2_processed.csv",
                      index_col="Case ID")

# Take the intersection sample
common = df_img.index.intersection(df_rna.index).intersection(df_clin.index)

img_sub  = df_img.loc[common]
rna_sub  = df_rna.loc[common]
clin_sub = df_clin.loc[common]

# 3. Label y（Dead=1, Alive=0）
y = (clin_sub["Survival Status"] == "Dead").astype(int).values

# Clinical pretreatment: median/mode imputation + One-Hot
clin_feats = clin_sub.drop(columns="Survival Status")
num_cols = clin_feats.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = clin_feats.select_dtypes(include=["object"]).columns.tolist()

pre_clin = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), cat_cols)
], remainder="drop")

X_clin_ohe = pre_clin.fit_transform(clin_feats)  # shape = (N, n_clin_ohe)

# numpy array turn to DataFrame
feat_names = (
    num_cols +
    pre_clin.named_transformers_["cat"]
            .named_steps["ohe"]
            .get_feature_names_out(cat_cols)
            .tolist()
)
df_clin_ohe = pd.DataFrame(
    X_clin_ohe,
    index=clin_feats.index,
    columns=feat_names
)

# Standardization of imaging
X_img_scaled = StandardScaler().fit_transform(img_sub.values)

# A list of Top-10 clinical features was defined
top10 = [
    "Days between CT and surgery",
    "Age at Histological Diagnosis",
    "Weight (lbs)",
    "Pack Years",
    "Quit Smoking Year",
    "%GG_0%",
    "Gender_Male",
    "Gender_Female",
    "Pathological T stage_T2b",
    "Pathological N stage_N2"
]

# The matrix corresponding to Top-10 features is extracted
# Joint set containing only Top-10 clinical + other modalities is constructed
X_clin_top10   = df_clin_ohe[top10].values
X_all_top10    = np.hstack([X_img_scaled, rna_sub.values, X_clin_top10])

# Construct four feature sets
X_img_only     = X_img_scaled
X_rna_only     = rna_sub.values
X_clin_only    = X_clin_ohe
X_all_three    = np.hstack([X_img_scaled, rna_sub.values, X_clin_ohe])

# 7. Model setting
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf     = RandomForestClassifier(n_estimators=100,
                                 min_samples_leaf=3,
                                 random_state=42,
                                 n_jobs=-1)
scoring = ["f1", "roc_auc", "precision", "recall"]
results = []

def eval_model(X, name):
    scores = cross_validate(clf, X, y,
                            cv=cv,
                            scoring=scoring,
                            return_train_score=False)
    # mean and std
    row = {"model": name}
    for m in scoring:
        arr = scores[f"test_{m}"]
        row[f"{m}_mean"] = arr.mean()
        row[f"{m}_std"]  = arr.std()
    results.append(row)
    
    print(f"\n=== {name} ===")
    for m in scoring:
        print(f"{m:9s}: {row[f'{m}_mean']:.3f} ± {row[f'{m}_std']:.3f}")

# Evaluate in sequence
eval_model(X_img_only,  "Image Only")
eval_model(X_rna_only,  "RNA Only")
eval_model(X_clin_top10, "Clinical Only")
eval_model(X_all_top10, "Image + RNA + Clinical")

# Save as CSV
df_res = pd.DataFrame(results)
os.makedirs("../result", exist_ok=True)
df_res.to_csv("../result/result2.csv", index=False)
print(f"save as result2.csv")
