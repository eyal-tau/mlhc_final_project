import pandas as pd
import numpy as np
import os
import pickle

from .impute import apply_imputation, apply_imputation_test
from .scale import (apply_feature_scaling, apply_feature_scaling_test)


def align_features_with_training(test_feature_matrix, target: str):
    """
    Align test feature matrix columns with training feature matrix columns.

    Args:
        test_feature_matrix: DataFrame with test features
        target: Target name (mortality, prolonged_stay, readmission)

    Returns:
        DataFrame with columns aligned to training features
    """
    print(f"\n--- Aligning features with training data for {target} ---")

    # Get package directory path
    package_dir = os.path.dirname(os.path.dirname(__file__))
    train_features_path = os.path.join(package_dir, "data", "feature_mats", f"{target}_final_feature_matrix.pkl")

    # Load training feature matrix to get expected columns
    print(f"Loading training features from: {train_features_path}")
    with open(train_features_path, 'rb') as f:
        train_features = pickle.load(f)

    # Get expected feature columns (exclude ID and target columns)
    id_cols = ['subject_id', 'hadm_id']
    target_cols = [target]
    expected_feature_cols = [col for col in train_features.columns if col not in id_cols + target_cols]

    print(f"Expected training features: {len(expected_feature_cols)}")
    print(f"Current test features: {len(test_feature_matrix.columns) - len(id_cols)}")

    # Get current test feature columns
    current_feature_cols = [col for col in test_feature_matrix.columns if col not in id_cols]

    # Find missing and extra features
    missing_features = set(expected_feature_cols) - set(current_feature_cols)
    extra_features = set(current_feature_cols) - set(expected_feature_cols)

    print(f"Missing features in test data: {len(missing_features)}")
    print(f"Extra features in test data: {len(extra_features)}")

    # Start with ID columns
    aligned_matrix = test_feature_matrix[id_cols].copy()

    # Define feature types for intelligent default values
    count_features = [col for col in expected_feature_cols if
                     (('count' in col.lower() or 'total' in col.lower() or 'unique' in col.lower()) and
                      any(x in col for x in ['vital_', 'lab_']))]

    physio_keywords = ['_mean', '_std', '_min', '_max', '_range', '_cv', '_t_range', '_first', '_last']
    physio_features = [col for col in expected_feature_cols if
                      any(keyword in col for keyword in physio_keywords) and
                      any(prefix in col for prefix in ['lab_', 'vital_'])]

    # Add expected features in the correct order
    for feature_col in expected_feature_cols:
        if feature_col in test_feature_matrix.columns:
            # Feature exists, use it
            aligned_matrix[feature_col] = test_feature_matrix[feature_col]
        else:
            # Feature missing, add with intelligent default value
            if feature_col in count_features or feature_col in physio_features:
                # Use NaN for count and physiological features so imputers can handle them
                aligned_matrix[feature_col] = np.nan
            else:
                # Use 0 for other features (demographics, categorical, etc.)
                aligned_matrix[feature_col] = 0

    return aligned_matrix


def preprocess_features_test(feature_matrix, target_col: str, features_missing_threshold=0.2, patients_missing_threshold=0.2):
    """
    Preprocess the feature matrix for modeling with missing value filtering
    """
    print("\n" + "="*50)
    print(f"PREPROCESSING FEATURES - {target_col}")
    print("="*50)

    # Separate features and targets
    id_cols = ['subject_id', 'hadm_id']
    feature_cols = [col for col in feature_matrix.columns if col not in id_cols]

    # Create feature matrix
    X = feature_matrix[feature_cols].copy()
    ids = feature_matrix[id_cols].copy()

    # Check missing values
    missing_summary = X.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]

    print(f"Columns with missing values: {len(missing_cols)}")
    if len(missing_cols) > 0:
        print(f"  Max missing values in any column: {missing_cols.max():,} ({missing_cols.max()/len(X)*100:.1f}%)")

    # FILTER FEATURES WITH >20% MISSING VALUES
    print(f"\nFiltering features with >{features_missing_threshold*100}% missing values...")
    feature_missing_pct = X.isnull().sum() / len(X)
    features_to_keep = feature_missing_pct[feature_missing_pct <= features_missing_threshold].index.tolist()
    features_to_remove = feature_missing_pct[feature_missing_pct > features_missing_threshold].index.tolist()

    print(f"  Removing {len(features_to_remove)} features with >{features_missing_threshold*100}% missing")
    X_filtered = X[features_to_keep].copy()

    # FILTER PATIENTS WITH >20% MISSING VALUES
    print(f"\nFiltering patients with >{patients_missing_threshold*100}% missing values...")
    patient_missing_pct = X_filtered.isnull().sum(axis=1) / len(X_filtered.columns)
    patients_to_keep = patient_missing_pct[patient_missing_pct <= patients_missing_threshold].index
    patients_to_remove = patient_missing_pct[patient_missing_pct > patients_missing_threshold].index

    print(f"  Removing {len(patients_to_remove)} patients with >{patients_missing_threshold*100}% missing")
    X_filtered = X_filtered.loc[patients_to_keep].copy()
    ids_filtered = ids.loc[patients_to_keep].copy()

    # Handle remaining missing values
    print("Handling remaining missing values...")
    remaining_missing = X_filtered.isnull().sum()
    remaining_missing_cols = remaining_missing[remaining_missing > 0]

    print(f"Remaining columns with missing values: {len(remaining_missing_cols)}")

    # Create missingness indicators for columns with substantial missing data
    high_missing_threshold = 0.1  # 10%
    high_missing_cols = remaining_missing_cols[remaining_missing_cols > len(X_filtered) * high_missing_threshold].index

    for col in high_missing_cols:
        X_filtered[f'{col}_missing'] = X_filtered[col].isnull().astype(int)

    filtered_matrix = pd.concat([ids_filtered, X_filtered], axis=1)
    print(f"✓ Filtered feature matrix: {len(filtered_matrix)} patients, {len(X_filtered.columns)} features")
    return filtered_matrix


def preprocess_features(feature_matrix, target_col: str, features_missing_threshold=0.2, patients_missing_threshold=0.2):
    """
    Preprocess the feature matrix for modeling with missing value filtering
    """
    print("\n" + "="*50)
    print(f"PREPROCESSING FEATURES - {target_col}")
    print("="*50)

    # Separate features and targets
    id_cols = ['subject_id', 'hadm_id']
    target_cols = [target_col]
    feature_cols = [col for col in feature_matrix.columns if col not in id_cols + target_cols]

    print(f"Features to process: {len(feature_cols)}")

    # Create feature matrix
    X = feature_matrix[feature_cols].copy()
    y = feature_matrix[target_cols].copy()
    ids = feature_matrix[id_cols].copy()

    # Check missing values
    missing_summary = X.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]

    print(f"Columns with missing values: {len(missing_cols)}")
    if len(missing_cols) > 0:
        print(f"  Max missing values in any column: {missing_cols.max():,} ({missing_cols.max()/len(X)*100:.1f}%)")

    # FILTER FEATURES WITH >20% MISSING VALUES
    print(f"\nFiltering features with >{features_missing_threshold*100}% missing values...")
    feature_missing_pct = X.isnull().sum() / len(X)
    features_to_keep = feature_missing_pct[feature_missing_pct <= features_missing_threshold].index.tolist()
    features_to_remove = feature_missing_pct[feature_missing_pct > features_missing_threshold].index.tolist()

    print(f"  Removing {len(features_to_remove)} features with >{features_missing_threshold*100}% missing")
    X_filtered = X[features_to_keep].copy()

    # FILTER PATIENTS WITH >20% MISSING VALUES
    print(f"\nFiltering patients with >{patients_missing_threshold*100}% missing values...")
    patient_missing_pct = X_filtered.isnull().sum(axis=1) / len(X_filtered.columns)
    patients_to_keep = patient_missing_pct[patient_missing_pct <= patients_missing_threshold].index
    patients_to_remove = patient_missing_pct[patient_missing_pct > patients_missing_threshold].index

    print(f"  Removing {len(patients_to_remove)} patients with >{patients_missing_threshold*100}% missing")
    X_filtered = X_filtered.loc[patients_to_keep].copy()
    y_filtered = y.loc[patients_to_keep].copy()
    ids_filtered = ids.loc[patients_to_keep].copy()

    # Handle remaining missing values
    print("Handling remaining missing values...")
    remaining_missing = X_filtered.isnull().sum()
    remaining_missing_cols = remaining_missing[remaining_missing > 0]

    print(f"Remaining columns with missing values: {len(remaining_missing_cols)}")

    # Create missingness indicators for columns with substantial missing data
    high_missing_threshold = 0.1  # 10%
    high_missing_cols = remaining_missing_cols[remaining_missing_cols > len(X_filtered) * high_missing_threshold].index

    for col in high_missing_cols:
        X_filtered[f'{col}_missing'] = X_filtered[col].isnull().astype(int)

    filtered_matrix = pd.concat([ids_filtered, X_filtered, y_filtered], axis=1)
    print(f"✓ Filtered feature matrix: {len(filtered_matrix)} patients, {len(X_filtered.columns)} features")
    return filtered_matrix


def process_features(feature_matrix, target, features_missing_threshold=0.2, patients_missing_threshold=0.5, run_mode = "train"):
    """
    Execute complete feature engineering pipeline
    """
    print("EXECUTING FEATURE ENGINEERING PIPELINE")

    # Preprocess features WITH MISSING VALUE FILTERING
    if run_mode == "train":
        filtered_feature_matrix = preprocess_features(feature_matrix, target, features_missing_threshold, patients_missing_threshold)
    else:
        filtered_feature_matrix = preprocess_features_test(feature_matrix, target, features_missing_threshold,
                                                      patients_missing_threshold)

    # For test mode: align features with training data before imputation
    if run_mode == "test":
        filtered_feature_matrix = align_features_with_training(filtered_feature_matrix, target)

    # Impute features
    if run_mode == "train":
        imputed_feature_matrix, imputers = apply_imputation(filtered_feature_matrix, target)
    else:
        imputed_feature_matrix = apply_imputation_test(filtered_feature_matrix, target)

    # Perform scaling
    if run_mode == "train":
        final_feature_matrix, scaler = apply_feature_scaling(imputed_feature_matrix, target)
    else:
        final_feature_matrix = apply_feature_scaling_test(imputed_feature_matrix, target)

    if run_mode == "train":
        return final_feature_matrix, scaler, imputers
    else:
        return final_feature_matrix


