import pandas as pd
import os
from .data_process.basic_data_fetch import get_icu_data, get_lab_data, get_vit_data
from .data_process.filter_basic_data import basic_filter
from .data_process.clinical_target_filter import create_separate_target_datasets
from .features_process.general import process_features
from .modalities.general import extract_all_modalities_data, get_targets_modalities, get_specific_target_modalities
from .features.general import execute_feature_engineering
from .models.final_model import gen_predictions


def run_pipeline_on_unseen_data(subject_ids ,client):
    """
    Run your full pipeline, from data loading to prediction.

    :param subject_ids: A list of subject IDs of an unseen test set.
    :type subject_ids: List[int]

    :param client: A BigQuery client object for accessing the MIMIC-III dataset.
    :type client: google.cloud.bigquery.client.Client

    :return: DataFrame with the following columns:
                - subject_id: Subject IDs, which in some cases can be different due to your analysis.
                - mortality_proba: Prediction probabilities for mortality.
                - prolonged_LOS_proba: Prediction probabilities for prolonged length of stay.
                - readmission_proba: Prediction probabilities for readmission.
    :rtype: pandas.DataFrame
    """

    # 1) extract data
    # Get the directory where this module is located
    package_dir = os.path.dirname(__file__)
    lab_meta = pd.read_csv(os.path.join(package_dir, "data", "labs_metadata.csv"))
    vit_meta = pd.read_csv(os.path.join(package_dir, "data", "vital_metadata.csv"))
    icu_data = get_icu_data(subject_ids, client)
    all_admissions_for_cohort = icu_data.copy()
    lab_data = get_lab_data(subject_ids, client, lab_meta)
    vit_data = get_vit_data(subject_ids, client, vit_meta)

    # 2) perform basic filter
    final_cohort, lab_filtered, vit_filtered = basic_filter(icu_data, lab_data, vit_data)

    datasets = create_separate_target_datasets(final_cohort, all_admissions_for_cohort, run_mode="test")

    # Access individual datasets
    mortality_dataset = datasets['mortality']
    prolonged_stay_dataset = datasets['prolonged_stay']
    readmission_dataset = datasets['readmission']

    # 3) extract modalities
    modalities_data = extract_all_modalities_data(final_cohort, client)
    target_modality_data = get_targets_modalities(datasets, modalities_data)
    mortality_icu_load, mortality_notes, mortality_medications = get_specific_target_modalities(target_modality_data, "mortality")
    prolonged_stay_icu_load, prolonged_stay_notes, prolonged_stay_medications = get_specific_target_modalities(target_modality_data, "prolonged_stay")
    readmission_icu_load, readmission_notes, readmission_medications = get_specific_target_modalities(target_modality_data,"readmission")

    # 4) generate features
    mortality_feature_matrix = execute_feature_engineering(mortality_dataset, "mortality", lab_filtered, vit_filtered,
                                                           mortality_medications, mortality_notes, mortality_icu_load)
    prolonged_stay_feature_matrix = execute_feature_engineering(prolonged_stay_dataset, "prolonged_stay", lab_filtered,
                                                                vit_filtered, prolonged_stay_medications, prolonged_stay_notes, prolonged_stay_icu_load)
    readmission_feature_matrix = execute_feature_engineering(readmission_dataset, "readmission", lab_filtered,
                                                             vit_filtered, readmission_medications, readmission_notes, readmission_icu_load)

    # 5) process features, impute and scale
    mortality_final_feature_matrix = process_features(mortality_feature_matrix, "mortality", run_mode="test")
    prolonged_stay_final_feature_matrix = process_features(prolonged_stay_feature_matrix, "prolonged_stay", run_mode="test")
    readmission_final_feature_matrix = process_features(readmission_feature_matrix, "readmission", run_mode="test")

    # 6) run models
    predictions_df = gen_predictions(mortality_final_feature_matrix, prolonged_stay_final_feature_matrix, readmission_final_feature_matrix)
    return predictions_df
