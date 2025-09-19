import pandas as pd
import pickle
import warnings
import os
warnings.filterwarnings('ignore')
from .demo import create_demographic_features
from .icu_load import create_icu_load_features
from .lab import create_lab_features
from .medication import create_medication_features
from .notes import create_notes_based_features
from .vital import create_vital_features


def combine_all_features(final_cohort_with_targets, target: str, lab_features, vital_features,
                        medication_features, demographic_features, notes_based_features,
                        icu_load_features):
    """
    Combine all feature sets into a unified feature matrix
    """
    print("\n" + "="*50)
    print("COMBINING ALL FEATURES")
    print("="*50)

    # Start with targets
    feature_matrix = final_cohort_with_targets[['subject_id', 'hadm_id']].copy()

    print(f"Starting with {len(feature_matrix)} patients")

    # Merge demographic features
    feature_matrix = feature_matrix.merge(demographic_features, on=['subject_id', 'hadm_id'], how='left')
    print(f"After demographics: {len(feature_matrix.columns)} columns")

    # Merge lab features
    feature_matrix = feature_matrix.merge(lab_features, on=['subject_id', 'hadm_id'], how='left')
    print(f"After lab features: {len(feature_matrix.columns)} columns")

    # Merge vital features
    feature_matrix = feature_matrix.merge(vital_features, on=['subject_id', 'hadm_id'], how='left')
    print(f"After vital features: {len(feature_matrix.columns)} columns")

    # Merge medication features
    feature_matrix = feature_matrix.merge(medication_features, on=['subject_id', 'hadm_id'], how='left')
    print(f"After medication features: {len(feature_matrix.columns)} columns")

    # Merge notes features
    feature_matrix = feature_matrix.merge(notes_based_features, on=['subject_id', 'hadm_id'], how='left')
    print(f"After notes based features: {len(feature_matrix.columns)} columns")

    # Merge icu load features
    feature_matrix = feature_matrix.merge(icu_load_features, on=['subject_id', 'hadm_id'], how='left')
    print(f"After icu load features: {len(feature_matrix.columns)} columns")

    total_features = len(feature_matrix.columns) - 3  # Subtract ID columns and target
    print(f"✓ Total features created: {total_features}")

    return feature_matrix


# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def execute_feature_engineering(final_cohort_with_targets, target: str, lab_filtered, vit_filtered,
                               medication_data, notes_data, icu_load_data: dict):
    """
    Execute complete feature engineering pipeline
    """
    print("EXECUTING FEATURE ENGINEERING PIPELINE")

    # Step 1: Create temporal features
    lab_features = create_lab_features(lab_filtered, final_cohort_with_targets)
    vital_features = create_vital_features(vit_filtered, final_cohort_with_targets)

    # Step 2: Create demographic features
    demographic_features = create_demographic_features(final_cohort_with_targets)

    # Step 3: Create medications features
    medication_features = create_medication_features(final_cohort_with_targets, medication_data)

    # Step 4: Create ICU Load features
    # Get package directory path
    package_dir = os.path.dirname(os.path.dirname(__file__))
    icu_load_mapping_path = os.path.join(package_dir, "data", "icu_load_mapping.pkl")
    icu_load_mapping = pickle.load(open(icu_load_mapping_path, "rb"))
    icu_load_features = create_icu_load_features(final_cohort_with_targets, icu_load_data, icu_load_mapping)

    # Step 5: Create Notes based features
    NOTE_CATEGORIES = ["Nursing/other", "Radiology", "Nursing", "ECG", "Physician", "Discharge summary", "Echo", "Respiratory", "Nutrition", "General", "Rehab Services", "Social Work", "Case Management", "Pharmacy", "Consult"]
    radiology_map_path = os.path.join(package_dir, "data", "radiology_descriptions_classified_new_fixed.csv")
    radiology_map_df = pd.read_csv(radiology_map_path)
    notes_features = create_notes_based_features(
        final_cohort_with_targets=final_cohort_with_targets,
        notes_df=notes_data,
        radiology_map_df=radiology_map_df,
        note_categories=NOTE_CATEGORIES,
        radiology_category_label="Radiology",    # change if your label differs
        description_col_in_notes="description",  # change if your column differs
        description_col_in_map="description",
        anatomy_col_in_map="class",
        invasive_col_in_map="is_invasive"
    )

    # Step 6: Combine all features
    feature_matrix = combine_all_features(
        final_cohort_with_targets, target, lab_features, vital_features,
        medication_features, demographic_features, notes_features, icu_load_features)

    return feature_matrix
