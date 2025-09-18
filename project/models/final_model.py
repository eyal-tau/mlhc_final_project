import pickle
import pandas as pd
import os


def use_model(feature_mat, target):
    # Get package directory path
    package_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(package_dir, 'data', 'models', f'{target}_calibrated_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Remove ID columns for prediction
    id_cols = ['subject_id', 'hadm_id']
    feature_cols = [col for col in feature_mat.columns if col not in id_cols]
    X = feature_mat[feature_cols]

    # Make predictions
    predictions = model.predict_proba(X)[:, 1]
    return predictions


def gen_predictions(mort_feature_mat, prolonged_stay_feature_mat, readmission_feature_mat):
    # Get predictions and subject IDs for each target
    mort_probs = use_model(mort_feature_mat, "mortality")
    mort_subjects = mort_feature_mat['subject_id'].values

    prol_probs = use_model(prolonged_stay_feature_mat, "prolonged_stay")
    prol_subjects = prolonged_stay_feature_mat['subject_id'].values

    readm_probs = use_model(readmission_feature_mat, "readmission")
    readm_subjects = readmission_feature_mat['subject_id'].values

    # Create individual DataFrames
    mort_df = pd.DataFrame({'subject_id': mort_subjects, 'mortality_proba': mort_probs})
    prol_df = pd.DataFrame({'subject_id': prol_subjects, 'prolonged_LOS_proba': prol_probs})
    readm_df = pd.DataFrame({'subject_id': readm_subjects, 'readmission_proba': readm_probs})

    # Merge on subject_id to handle different patient sets/orders
    prediction_df = mort_df.merge(prol_df, on='subject_id', how='outer')
    prediction_df = prediction_df.merge(readm_df, on='subject_id', how='outer')

    # Fill any missing values with NaN (for patients not in all datasets)
    return prediction_df