import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline        import make_pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.preprocessing   import StandardScaler, OneHotEncoder
from sklearn.impute          import SimpleImputer
from sklearn.ensemble        import RandomForestClassifier

# Loading data
features_df  = pd.read_csv("../result/features1.csv")
clinical_df  = pd.read_csv("../dataset1/clinical1.csv")
labels_df    = clinical_df[["PatientID", "deadstatus.event"]].dropna()

# Return F1 and AUC
def evaluate_model(X, y, name="model", n_splits=5):
    skf   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=100, random_state=42)
    )
    f1  = cross_val_score(model, X, y, cv=skf, scoring="f1")
    auc = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")

    print(f"{name}")
    print(f"  F1:  {f1.mean():.3f}  per fold: {np.round(f1, 3)}")
    print(f"  AUC: {auc.mean():.3f}  per fold: {np.round(auc, 3)}")
    print()

    return f1, auc

results = []

# A. Image Only
df_img = features_df.merge(labels_df,
                           left_on="patient_id",
                           right_on="PatientID")
X_img = df_img.drop(columns=["patient_id", "PatientID", "deadstatus.event"])
y_img = df_img["deadstatus.event"].astype(int).values

f1_img, auc_img = evaluate_model(X_img, y_img, name="Image Only", n_splits=3)
results.append({
    "model":       "Image Only",
    "f1_mean":     f1_img.mean(),
    "f1_per_fold": ",".join(map(str, np.round(f1_img,3))),
    "auc_mean":    auc_img.mean(),
    "auc_per_fold":",".join(map(str, np.round(auc_img,3)))
})

# B. Clinical Only
clinical_vars = clinical_df.set_index("PatientID")\
                          .drop(columns=["Survival.time","deadstatus.event"])
df_clinical   = labels_df.merge(clinical_vars,
                                left_on="PatientID",
                                right_index=True)
X_c           = df_clinical.drop(columns=["PatientID","deadstatus.event"])
y_c           = df_clinical["deadstatus.event"].astype(int).values

num_cols = X_c.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X_c.select_dtypes(include=["object","category"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"),    num_cols),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore")),
     cat_cols)
])

X_c_pre = preprocessor.fit_transform(X_c)

f1_clin, auc_clin = evaluate_model(X_c_pre, y_c, name="Clinical Only", n_splits=5)
results.append({
    "model":        "Clinical Only",
    "f1_mean":      f1_clin.mean(),
    "f1_per_fold":  ",".join(map(str, np.round(f1_clin,3))),
    "auc_mean":     auc_clin.mean(),
    "auc_per_fold": ",".join(map(str, np.round(auc_clin,3)))
})

# C. Image + Clinical
df_merged = features_df.merge(clinical_df,
                              left_on="patient_id",
                              right_on="PatientID")
df_merged = df_merged[df_merged["deadstatus.event"].notna()]

X_all = df_merged.drop(columns=[
    "patient_id","PatientID","Survival.time","deadstatus.event"
])
y_all = df_merged["deadstatus.event"].astype(int).values

num_cols_all = X_all.select_dtypes(include=["number"]).columns.tolist()
cat_cols_all = X_all.select_dtypes(include=["object","category"]).columns.tolist()

preprocessor_all = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols_all),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore")),
     cat_cols_all)
])

X_all_pre = preprocessor_all.fit_transform(X_all)

f1_all, auc_all = evaluate_model(X_all_pre, y_all,
                                 name="Image + Clinical", n_splits=5)
results.append({
    "model":        "Image + Clinical",
    "f1_mean":      f1_all.mean(),
    "f1_per_fold":  ",".join(map(str, np.round(f1_all,3))),
    "auc_mean":     auc_all.mean(),
    "auc_per_fold": ",".join(map(str, np.round(auc_all,3)))
})

# Save results
os.makedirs("../result", exist_ok=True)
df_res = pd.DataFrame(results)
df_res.to_csv("../result/result1.csv", index=False)
print("save as result1.csv")
