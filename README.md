# Early Prediction of ICU Outcomes (MIMIC-III)

This repo contains:

- Project folder with pipeline to run the models on new test data (using unseen_data_evaluation as main executer)
- The project folder contains other stand alone scripts and saved files that were run on the cluster.
- Jupyter notebook that was used to develop the models and pipeline (There you can see the data processing, feature engineering, training and evaluation)



In this project we predict: **mortality**, **prolonged stay (>7 days)**, and **30-day readmission** using only the **first 48 hours** of ICU data (with a **6-hour lead gap**) from **MIMIC-III**. 


---

## Highlights

- **Data**: MIMIC-III; first ICU stays; features up to hour 48 only.
- **Features (~600)**: demographics, labs, vitals, medications, notes based, icu load estimates.
- **Models**: XGBoost and Random Forest; outcome-specific models (one per target).
- **Imbalance handling** (per outcome):
  - **Mortality** → downsampling (1:1) performed best.
  - **Readmission** → SMOTE oversampling performed best.
  - **Prolonged stay** → baseline class balance sufficient.
- **Calibration & Thresholds**: calibration on validation set; task-aware threshold optimization.
- **Interpretability**: SHAP values
---
