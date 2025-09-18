def create_demographic_features(final_cohort_with_targets):
    """
    Create demographic features from cohort data
    """
    print("\n" + "="*50)
    print("CREATING DEMOGRAPHIC FEATURES")
    print("="*50)

    demo_features = final_cohort_with_targets[['subject_id', 'hadm_id']].copy()

    # Age at admission (already calculated)
    demo_features['age'] = final_cohort_with_targets['age_at_admission']

    # Age categories
    demo_features['age_group_young'] = (demo_features['age'] < 45).astype(int)
    demo_features['age_group_middle'] = ((demo_features['age'] >= 45) & (demo_features['age'] < 65)).astype(int)
    demo_features['age_group_elderly'] = ((demo_features['age'] >= 65) & (demo_features['age'] < 80)).astype(int)
    demo_features['age_group_very_elderly'] = (demo_features['age'] >= 80).astype(int)

    # Gender
    demo_features['gender_male'] = (final_cohort_with_targets['gender'] == 'M').astype(int)
    demo_features['gender_female'] = (final_cohort_with_targets['gender'] == 'F').astype(int)

    # Ethnicity (simplified categories)
    ethnicity_map = {
        'WHITE': 'white',
        'BLACK': 'black',
        'HISPANIC': 'hispanic',
        'ASIAN': 'asian'
    }

    demo_features['ethnicity_white'] = 0
    demo_features['ethnicity_black'] = 0
    demo_features['ethnicity_hispanic'] = 0
    demo_features['ethnicity_asian'] = 0
    demo_features['ethnicity_other'] = 0

    for _, row in final_cohort_with_targets.iterrows():
        ethnicity = str(row['ethnicity']).upper()
        found = False
        for key, value in ethnicity_map.items():
            if key in ethnicity:
                demo_features.loc[demo_features['subject_id'] == row['subject_id'], f'ethnicity_{value}'] = 1
                found = True
                break
        if not found:
            demo_features.loc[demo_features['subject_id'] == row['subject_id'], 'ethnicity_other'] = 1

    print(f"✓ Created {len(demo_features.columns)-2} demographic features")

    return demo_features
