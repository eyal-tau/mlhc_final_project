import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import pickle
import os


def apply_imputation_test(filtered_matrix, target):
    id_cols = ['subject_id', 'hadm_id']
    feature_cols = [col for col in filtered_matrix.columns if col not in id_cols]

    X = filtered_matrix[feature_cols].copy()
    ids = filtered_matrix[id_cols].copy()

    # Check initial missing values
    initial_missing = X.isnull().sum().sum()
    print(f"Initial missing values in test data: {initial_missing:,}")

    if initial_missing == 0:
        print("No missing values found - returning original test data")
        return filtered_matrix

    # Get package directory path
    package_dir = os.path.dirname(os.path.dirname(__file__))
    imputer_path = os.path.join(package_dir, 'data', 'imputers', f'{target}_imputer.pkl')
    imputers = pickle.load(open(imputer_path, "rb"))

    # Apply saved imputers
    if 'count' in imputers:
        # 1. Count features -> Fill with 0
        count_features = [col for col in feature_cols if
                          (('count' in col.lower() or 'total' in col.lower() or 'unique' in col.lower()) and
                           any(x in col for x in ['vital_', 'lab_']))]

        X[count_features] = imputers['count'].transform(X[count_features])

    if 'physiological' in imputers:
        # 2. Physiological features -> Fill with median
        physio_keywords = ['_mean', '_std', '_min', '_max', '_range', '_cv', '_t_range', '_first', '_last']
        physio_features = [col for col in feature_cols if
                           any(keyword in col for keyword in physio_keywords) and
                           any(prefix in col for prefix in ['lab_', 'vital_'])]

        X[physio_features] = imputers['physiological'].transform(X[physio_features])

    print(f"Imputation completed: {initial_missing:,} -> {X.isnull().sum().sum()} missing values")

    # Check for remaining missing values
    remaining_missing = X.isnull().sum()
    cols_with_missing = remaining_missing[remaining_missing > 0]

    if len(cols_with_missing) > 0:
        print(f"\nWARNING: {len(cols_with_missing)} columns still have missing values:")
        # For remaining missing values, fill with 0 (conservative approach)
        print("Filling remaining missing values with 0...")
        X = X.fillna(0)

    imputed_matrix = pd.concat([ids, X], axis=1)
    return imputed_matrix


def apply_imputation(filtered_matrix, target_col):
    """
    Apply refined imputation based on actual missing value patterns

    Only handles:
    1. Count features (vital/lab counts) -> 0
    2. Physiological features (mean, std, min, max, range, cv, t_range) -> median
    """
    print(f"\nApplying refined imputation for {target_col}...")

    # Separate components
    id_cols = ['subject_id', 'hadm_id']
    target_cols = [target_col]
    feature_cols = [col for col in filtered_matrix.columns if col not in id_cols + target_cols]

    X = filtered_matrix[feature_cols].copy()
    y = filtered_matrix[target_cols].copy()
    ids = filtered_matrix[id_cols].copy()

    # Check initial missing values
    initial_missing = X.isnull().sum().sum()
    print(f"Initial missing values: {initial_missing:,}")

    if initial_missing == 0:
        print("No missing values found - returning original data")
        return filtered_matrix, {}

    imputers = {}

    # 1. Count features (vital_count_*, lab_count_*) -> Fill with 0
    count_features = [col for col in feature_cols if
                     (('count' in col.lower() or 'total' in col.lower() or 'unique' in col.lower()) and any(x in col for x in ['vital_', 'lab_']))]

    if len(count_features) > 0:
        missing_before = X[count_features].isnull().sum().sum()
        if missing_before > 0:
            zero_imputer = SimpleImputer(strategy='constant', fill_value=0)
            X[count_features] = zero_imputer.fit_transform(X[count_features])
            print(f"Count features ({len(count_features)}): {missing_before:,} missing values filled with 0")
            imputers['count'] = zero_imputer

    # 2. Physiological features (mean, std, min, max, range, cv, slope) -> Fill with median
    physio_keywords = ['_mean', '_std', '_min', '_max', '_range', '_cv', '_t_range', '_first', '_last']
    physio_features = [col for col in feature_cols if
                      any(keyword in col for keyword in physio_keywords) and
                      any(prefix in col for prefix in ['lab_', 'vital_'])]

    if len(physio_features) > 0:
        missing_before = X[physio_features].isnull().sum().sum()
        if missing_before > 0:
            physio_imputer = SimpleImputer(strategy='median')
            X[physio_features] = physio_imputer.fit_transform(X[physio_features])
            print(f"Physiological features ({len(physio_features)}): {missing_before:,} missing values filled with median")
            imputers['physiological'] = physio_imputer

    print(f"Imputation completed: {initial_missing:,} -> {X.isnull().sum().sum()} missing values")

    remaining_missing = X.isnull().sum()
    cols_with_missing = remaining_missing[remaining_missing > 0]

    if len(cols_with_missing) > 0:
        print(f"\nWARNING: {len(cols_with_missing)} columns still have missing values:")
        for col, missing_count in cols_with_missing.items():
            missing_pct = missing_count / len(X) * 100
            print(f"  {col}: {missing_count} ({missing_pct:.2f}%)")
    else:
        print("All missing values successfully imputed!")


    # Reconstruct the matrix
    imputed_matrix = pd.concat([ids, X, y], axis=1)

    return imputed_matrix, imputers
