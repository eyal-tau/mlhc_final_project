def create_medication_features(final_cohort_with_targets, medications_data):
    """
    Create medication features from medications data using vectorized pandas operations
    """
    print("\n" + "="*50)
    print("CREATING MEDICATION FEATURES")
    print("="*50)

    # Start with base cohort structure
    med_features = final_cohort_with_targets[['subject_id', 'hadm_id']].copy()

    # Define core medication categories
    vasopressors = [
        'norepinephrine', 'noradrenaline', 'levophed',
        'epinephrine', 'adrenaline',
        'dopamine', 'intropin',
        'vasopressin', 'pitressin'
    ]

    sedatives = [
        'propofol', 'diprivan',
        'midazolam', 'versed', 'dormicum',
        'dexmedetomidine', 'precedex'
    ]

    # Helper function to check if drug contains any target drugs
    def contains_any_drug(drug_series, drug_list):
        """Vectorized function to check if drugs contain any target drugs"""
        drug_series_clean = drug_series.fillna('').str.lower()
        return drug_series_clean.str.contains('|'.join(drug_list), regex=True, na=False)

    # Create drug indicator columns in medications data
    medications_work = medications_data.copy()
    medications_work['is_vasopressor'] = contains_any_drug(medications_work['drug'], vasopressors)
    medications_work['is_norepinephrine'] = contains_any_drug(medications_work['drug'],
                                                            ['norepinephrine', 'noradrenaline', 'levophed'])
    medications_work['is_sedative'] = contains_any_drug(medications_work['drug'], sedatives)
    medications_work['is_vancomycin'] = contains_any_drug(medications_work['drug'],
                                                        ['vancomycin', 'vancocin'])

    # FEATURE 1: Total medications count
    total_meds = medications_work.groupby(['subject_id', 'hadm_id']).size().reset_index(name='total_medications')

    # FEATURE 2-5: Binary indicators (any occurrence)
    binary_features = medications_work.groupby(['subject_id', 'hadm_id']).agg({
        'is_vasopressor': 'any',
        'is_norepinephrine': 'any',
        'is_sedative': 'any',
        'is_vancomycin': 'any'
    }).reset_index()

    # Rename columns
    binary_features = binary_features.rename(columns={
        'is_vasopressor': 'has_vasopressors',
        'is_norepinephrine': 'has_norepinephrine',
        'is_sedative': 'has_sedatives',
        'is_vancomycin': 'has_vancomycin'
    })

    # FEATURE 6: Multiple vasopressors (count unique vasopressor drugs)
    vasopressor_variety = (medications_work[medications_work['is_vasopressor']]
                          .groupby(['subject_id', 'hadm_id'])['drug']
                          .nunique()
                          .reset_index(name='vasopressor_count'))
    vasopressor_variety['multiple_vasopressors'] = (vasopressor_variety['vasopressor_count'] > 1).astype(int)

    # Merge all features
    med_features = med_features.merge(total_meds, on=['subject_id', 'hadm_id'], how='left')
    med_features = med_features.merge(binary_features, on=['subject_id', 'hadm_id'], how='left')
    med_features = med_features.merge(vasopressor_variety[['subject_id', 'hadm_id', 'multiple_vasopressors']],
                                    on=['subject_id', 'hadm_id'], how='left')

    # Fill missing values with 0 (patients with no medications)
    feature_cols = ['total_medications', 'has_vasopressors', 'has_norepinephrine',
                   'has_sedatives', 'has_vancomycin', 'multiple_vasopressors']
    med_features[feature_cols] = med_features[feature_cols].fillna(0).astype(int)

    # FEATURE 7: Critical Care Triad (vectorized)
    med_features['critical_care_triad'] = (
        (med_features['has_vasopressors'] +
         med_features['has_sedatives'] +
         med_features['has_vancomycin']) >= 2
    ).astype(int)

    # FEATURE 8: Septic Shock Syndrome (vectorized)
    med_features['septic_shock_syndrome'] = (
        med_features['has_vasopressors'] & med_features['has_vancomycin']
    ).astype(int)

    # Print summary
    print(f"✓ Created {len(med_features.columns)-2} medication features")

    # Show prevalence for key features
    key_features = ['has_vasopressors', 'has_norepinephrine', 'has_sedatives',
                   'has_vancomycin', 'multiple_vasopressors', 'critical_care_triad']

    print(f"Feature prevalences:")
    for feature in key_features:
        count = med_features[feature].sum()
        prevalence = count / len(med_features) * 100
        print(f"  {feature}: {count} patients ({prevalence:.1f}%)")

    return med_features