import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# n_splits = 5
features_df = pd.read_csv("../result/features1.csv")
clinical_df = pd.read_csv("../testdata/dataset1/clinical1.csv")
labels_df = clinical_df[["PatientID", "deadstatus.event"]].dropna()

def evaluate_model(X, y, name="model", n_splits=3):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
    f1 = cross_val_score(model, X, y, cv=skf, scoring="f1")
    auc = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")
    print(f"{name}")
    print(f"  F1:  {f1.mean():.3f}  per fold: {np.round(f1, 3)}")
    print(f"  AUC: {auc.mean():.3f}  per fold: {np.round(auc, 3)}")
    print()

# A. image
df_img = features_df.merge(labels_df, left_on="patient_id", right_on="PatientID")
X_img = df_img.drop(columns=["patient_id", "PatientID", "deadstatus.event"])
y_img = df_img["deadstatus.event"]

evaluate_model(X_img, y_img, name="image_model", n_splits=3)

# B. clinical
clinical_vars = clinical_df.set_index("PatientID").drop(columns=["Survival.time", "deadstatus.event"])
df_clinical = labels_df.merge(clinical_vars, left_on="PatientID", right_index=True)

X_c = df_clinical.drop(columns=["PatientID", "deadstatus.event"])
y_c = df_clinical["deadstatus.event"]

num_cols = X_c.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X_c.select_dtypes(include=["object", "category"]).columns.tolist()

num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))

preprocessor = ColumnTransformer([("num", num_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)])
model_clinical = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=100, random_state=42))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1 = cross_val_score(model_clinical, X_c, y_c, cv=skf, scoring="f1")
auc = cross_val_score(model_clinical, X_c, y_c, cv=skf, scoring="roc_auc")

print(f"clincial")
print(f"  F1:  {f1.mean():.3f}  per fold: {np.round(f1, 3)}")
print(f"  AUC: {auc.mean():.3f}  per fold: {np.round(auc, 3)}")
print()

# C. image + clinical
df_merged = features_df.merge(clinical_df, left_on="patient_id", right_on="PatientID")
df_merged = df_merged[df_merged["deadstatus.event"].notna()]

X_all = df_merged.drop(columns=["patient_id", "PatientID", "Survival.time", "deadstatus.event"])
y_all = df_merged["deadstatus.event"]

num_cols = X_all.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X_all.select_dtypes(include=["object", "category"]).columns.tolist()

preprocessor_all = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat_cols)
    ])

model_all = make_pipeline(preprocessor_all, RandomForestClassifier(n_estimators=100, random_state=42))

f1 = cross_val_score(model_all, X_all, y_all, cv=skf, scoring="f1")
auc = cross_val_score(model_all, X_all, y_all, cv=skf, scoring="roc_auc")

print(f"image + clinical")
print(f"  F1:  {f1.mean():.3f}  per fold: {np.round(f1, 3)}")
print(f"  AUC: {auc.mean():.3f}  per fold: {np.round(auc, 3)}")