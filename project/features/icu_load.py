def score_value(x, mapping):
    for ub, sc in mapping:
        if x <= ub:
            return sc

def create_icu_load_features(final_cohort_with_targets, icu_load_dict, mapping_dict):
    """
    Build ICU occupancy score features from multiple occupancy types.

    Parameters:
    - final_cohort_with_targets: base cohort dataframe
    - icu_load_dict: dict with keys like {'icu_lw_12h': df, 'icu_naw_12h': df, ...}
    - mapping_dict: dict with same keys containing score mappings

    Returns:
    - DataFrame with columns: ['subject_id', 'hadm_id', 'icu_lw_12h_score', 'icu_naw_12h_score', ...]
    """
    print("\n" + "="*50)
    print("CREATING ICU OCCUPANCY FEATURES")
    print("="*50)

    # Start with base cohort
    features = final_cohort_with_targets[['subject_id','hadm_id']].drop_duplicates().copy()

    # Process each occupancy type
    for occ_type, occ_df in icu_load_dict.items():
        print(f"Processing {occ_type}...")

        # Get corresponding mapping
        if occ_type not in mapping_dict:
            print(f"Warning: No mapping found for {occ_type}, skipping...")
            continue

        mapping = mapping_dict[occ_type]

        # Prepare occupancy data with scores
        occ_data = occ_df[['subject_id','hadm_id','icu_occupancy_count']].copy()
        occ_data[f'{occ_type}_score'] = occ_data['icu_occupancy_count'].apply(
            lambda v: score_value(v, mapping)
        ).astype(int)

        # Keep only the score column for merging
        occ_scores = occ_data[['subject_id', 'hadm_id', f'{occ_type}_score']]

        # Merge with features
        features = features.merge(occ_scores, on=['subject_id','hadm_id'], how='left')

        # Fill missing values with score 1 (lowest score)
        features[f'{occ_type}_score'] = features[f'{occ_type}_score'].fillna(1).astype(int)

        print(f"  ✓ Added {occ_type}_score column")

    total_features = features.shape[1] - 2  # subtract subject_id and hadm_id
    print(f"✓ Features shape: {features.shape[0]:,} rows × {total_features} features")
    print(f"✓ Feature columns: {[col for col in features.columns if col.endswith('_score')]}")

    return features
