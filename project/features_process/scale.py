from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import os


def apply_feature_scaling_test(imputed_matrix, target):
    id_cols = ['subject_id', 'hadm_id']
    feature_cols = [col for col in imputed_matrix.columns if col not in id_cols]

    X = imputed_matrix[feature_cols].copy()
    ids = imputed_matrix[id_cols].copy()

    # Get package directory path
    package_dir = os.path.dirname(os.path.dirname(__file__))
    scaler_path = os.path.join(package_dir, 'data', 'scalers', f'{target}_scaler.pkl')
    scaler = pickle.load(open(scaler_path, 'rb'))

    # Apply saved scaler
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index
    )

    scaled_matrix = pd.concat([ids, X_scaled], axis=1)
    return scaled_matrix


def apply_feature_scaling(imputed_matrix, target_col):
    """
    Apply standard scaling to features
    """
    print(f"\nApplying feature scaling for {target_col}...")

    # Separate components
    id_cols = ['subject_id', 'hadm_id']
    target_cols = [target_col]
    feature_cols = [col for col in imputed_matrix.columns if col not in id_cols + target_cols]

    X = imputed_matrix[feature_cols].copy()
    y = imputed_matrix[target_cols].copy()
    ids = imputed_matrix[id_cols].copy()

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    print(f"Scaled {len(feature_cols)} features")

    # Reconstruct matrix
    scaled_matrix = pd.concat([ids, X_scaled, y], axis=1)
    return scaled_matrix, scaler
