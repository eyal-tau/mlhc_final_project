from .icu_occupancy import extract_all_icu_occupancy_data
from .notes import extract_notes_data
from .medications import extract_medications_data

def extract_all_modalities_data(final_cohort, con):
    """
    Extract all modality data once from the unified cohort
    """
    print("EXTRACTING ALL MODALITY DATA FROM UNIFIED COHORT")
    print("=" * 60)

    print(f"Processing {len(final_cohort)} patients ({len(set(final_cohort['hadm_id']))} unique admissions)")

    # Extract all modalities
    icu_load_data = extract_all_icu_occupancy_data(final_cohort, con)
    notes_data = extract_notes_data(final_cohort, con)
    medications_data = extract_medications_data(final_cohort, con)

    print(f"\n{'=' * 70}")
    print("CONSOLIDATED DATA EXTRACTION COMPLETED!")
    print(f"{'=' * 70}")
    print(f"✅ ICU Load Data: {len(icu_load_data)} variants extracted")
    print(f"✅ Notes Data: {len(notes_data):,} records")
    print(f"✅ Medications Data: {len(medications_data):,} records")

    return {
        'icu_load_data': icu_load_data,
        'notes_data': notes_data,
        'medications_data': medications_data
    }
    # return icu_load_data, notes_data, medications_data


# =============================================================================
# TARGET-SPECIFIC DATA FILTERING
# =============================================================================

def filter_modality_data_for_target(modalities_data, target_dataset, target_name):
    """
    Filter consolidated modality data for a specific target dataset
    """
    print(f"\n--- Filtering data for {target_name.upper()} ---")

    # Get target-specific patient identifiers
    target_ids = target_dataset[['subject_id', 'hadm_id']].drop_duplicates()
    print(f"Target dataset size: {len(target_ids)} patients")

    # Filter each modality to this target's patients
    target_data = {}

    # ICU Load Data
    target_icu_load = {}
    for icu_type, icu_df in modalities_data['icu_load_data'].items():
        target_icu_load[icu_type] = icu_df.merge(
            target_ids, on=['subject_id', 'hadm_id'], how='inner'
        )
        print(f"  {icu_type}: {len(target_icu_load[icu_type])} records")

    target_data['icu_load_data'] = target_icu_load

    # Notes Data
    target_data['notes_data'] = modalities_data['notes_data'].merge(
        target_ids, on=['subject_id', 'hadm_id'], how='inner'
    )
    print(f"  Notes: {len(target_data['notes_data'])} records")

    # Medications Data
    target_data['medications_data'] = modalities_data['medications_data'].merge(
        target_ids, on=['subject_id', 'hadm_id'], how='inner'
    )
    print(f"  Medications: {len(target_data['medications_data'])} records")
    return target_data


def create_all_target_specific_data(datasets_dict, consolidated_modality_data):
    """
    Create target-specific modality data for all targets from consolidated extractions
    """
    print("\n" + "=" * 70)
    print("CREATING TARGET-SPECIFIC MODALITY DATA")
    print("=" * 70)

    target_modality_data = {}

    for target_name, target_dataset in datasets_dict.items():
        target_data = filter_modality_data_for_target(
            consolidated_modality_data, target_dataset, target_name
        )
        target_modality_data[target_name] = target_data

    print(f"\n{'=' * 70}")
    print("TARGET-SPECIFIC MODALITY DATA CREATION COMPLETED!")
    print(f"{'=' * 70}")

    return target_modality_data


def get_targets_modalities(datasets, modalities_data):
    target_modality_data = create_all_target_specific_data(
        {
            'mortality': datasets['mortality'],
            'prolonged_stay': datasets['prolonged_stay'],
            'readmission': datasets['readmission']
        },
        modalities_data
    )
    return target_modality_data


def get_specific_target_modalities(targets_modality_data, target_name):
    # Extract target-specific data for easy access
    icu_load = targets_modality_data[target_name]['icu_load_data']
    notes = targets_modality_data[target_name]['notes_data']
    medications = targets_modality_data[target_name]['medications_data']
    return icu_load, notes, medications