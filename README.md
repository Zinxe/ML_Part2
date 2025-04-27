# ML_Part2: NSCLC Multimodal Prediction Pipeline
This project builds a machine learning pipeline to predict NSCLC patient outcomes (Alive/Dead) using:

Imaging features (radiomics from CT)

RNA-seq features

Clinical features

The pipeline is managed by Snakemake and runs automatically on an HPC cluster with Conda/Mamba environments.

# Workflow Steps
Extract imaging features (image1.py, image2.py)

Preprocess RNA-seq and clinical data (dataclean2.py)

Train Random Forest models (model1.py, model2.py)

# Evaluate four feature sets:

Image only

RNA only

Clinical Top-10 features

All three combined

# How to Run
bash
cd code_hpc
sbatch run_snakemake.sh
Results will be saved under the result/ folder as result1.csv and result2.csv.

# Notes
Uses Stratified 5-Fold CV, scoring by F1, AUC, precision, recall.

Imaging features include shape and GLCM texture.

Clinical top-10 features are selected manually.