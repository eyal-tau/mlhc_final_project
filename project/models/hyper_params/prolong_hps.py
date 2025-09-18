#!/usr/bin/env python3
"""
Hyperparameter tuning for Prolonged Stay prediction
Uses baseline strategy (no resampling - best performing)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle
import json
from itertools import product
import time


def model_run(X_train, y_train, X_val, y_val, params):
    """Train XGBoost with given params and return validation scores"""

    # Calculate pos_weight for class imbalance
    pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    # Train XGBoost with pos_weight
    model = xgb.XGBClassifier(
        **params,
        scale_pos_weight=pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # Predict on validation set
    y_val_proba = model.predict_proba(X_val)[:, 1]

    # Calculate metrics
    val_auc = roc_auc_score(y_val, y_val_proba)
    val_ap = average_precision_score(y_val, y_val_proba)

    return val_auc, val_ap


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Prolonged Stay Hyperparameter Tuning')
    parser.add_argument('--input_matrix', required=True, help='Path to prolonged_stay feature matrix pkl file')
    parser.add_argument('--out_folder', required=True, help='Output folder for results')
    parser.add_argument('--n_estimators', nargs='+', type=int, default=[200, 300, 500], help='n_estimators values')
    parser.add_argument('--max_depth', nargs='+', type=int, default=[4, 6, 8], help='max_depth values')
    parser.add_argument('--learning_rate', nargs='+', type=float, default=[0.05, 0.1, 0.15],
                        help='learning_rate values')

    args = parser.parse_args()

    # Create output folder
    import os
    os.makedirs(args.out_folder, exist_ok=True)

    print("=" * 50)
    print("PROLONGED STAY HYPERPARAMETER TUNING")
    print("=" * 50)
    print(f"Input matrix: {args.input_matrix}")
    print(f"Output folder: {args.out_folder}")

    # Load data
    print("Loading prolonged stay data...")
    prolonged_data = pd.read_pickle(args.input_matrix)

    # Prepare data
    id_cols = ['subject_id', 'hadm_id']
    target = 'prolonged_stay'
    feature_cols = [col for col in prolonged_data.columns if col not in id_cols + [target]]

    X = prolonged_data[feature_cols]
    y_all_targets = prolonged_data[[target]]  # Keep as DataFrame for consistency
    ids = prolonged_data[id_cols]

    print(f"Data shape: {X.shape}")
    print(f"Prolonged stay rate: {y_all_targets[target].mean():.3f}")

    # Use same splitting strategy as your original model code
    X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
        X, y_all_targets, ids,
        test_size=0.2,
        stratify=y_all_targets[target],
        random_state=42
    )

    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_temp, y_temp, ids_temp,
        test_size=0.2,
        stratify=y_temp[target],
        random_state=42
    )

    # Convert to series for model training
    y_train = y_train[target]
    y_val = y_val[target]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print("Using baseline strategy (no resampling)")

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1.0, 1.5, 2.0]
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    param_combinations = [dict(zip(keys, combo)) for combo in product(*values)]

    print(f"Testing {len(param_combinations)} parameter combinations...")

    best_score = 0
    best_params = None
    results = []

    start_time = time.time()

    for i, params in enumerate(param_combinations):
        try:
            val_auc, val_ap = model_run(X_train, y_train, X_val, y_val, params)

            # Use AUC-ROC as primary metric (balanced target)
            score = val_auc

            results.append({
                'params': params,
                'val_auc': val_auc,
                'val_ap': val_ap,
                'score': score
            })

            if score > best_score:
                best_score = score
                best_params = params

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Completed {i + 1}/{len(param_combinations)} ({elapsed:.1f}s)")
                print(f"Current best AUC-ROC: {best_score:.4f}")

        except Exception as e:
            print(f"Error with params {params}: {e}")
            continue

    total_time = time.time() - start_time

    # Results
    print(f"\n{'=' * 50}")
    print("PROLONGED STAY TUNING RESULTS")
    print(f"{'=' * 50}")
    print(f"Total combinations tested: {len(results)}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Best AUC-ROC: {best_score:.4f}")

    print(f"\nBest parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # Save results
    best_params_file = os.path.join(args.out_folder, 'prolonged_stay_best_params.json')
    results_file = os.path.join(args.out_folder, 'prolonged_stay_tuning_results.pkl')

    with open(best_params_file, 'w') as f:
        json.dump(best_params, f, indent=2)

    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nSaved:")
    print(f"  {best_params_file}")
    print(f"  {results_file}")


if __name__ == "__main__":
    main()