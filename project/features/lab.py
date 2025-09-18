from .helper import aggregate_temporal_features


def create_lab_features(lab_filtered, final_cohort_with_targets):
    """
    Create features from lab data
    """
    print("\n" + "="*50)
    print("CREATING LAB FEATURES")
    print("="*50)

    # Aggregate lab measurements
    lab_agg = aggregate_temporal_features(
        lab_filtered,
        ['subject_id', 'hadm_id', 'itemid'],
        'valuenum',
        'lab'
    )

    # Pivot to create one column per lab test per statistic
    lab_pivot = lab_agg.pivot_table(
        index=['subject_id', 'hadm_id'],
        columns='itemid',
        values=['count', 'mean', 'std', 'min', 'max', 'range', 'cv', 't_range'],
        fill_value=0
    )

    # Flatten column names
    lab_pivot.columns = [f'lab_{stat}_{itemid}' for stat, itemid in lab_pivot.columns]
    lab_pivot = lab_pivot.reset_index()

    # Create summary features across all labs
    value_cols = [col for col in lab_pivot.columns if col.startswith('lab_')]
    lab_pivot['total_lab_measurements'] = lab_pivot[[col for col in value_cols if 'count' in col]].sum(axis=1)
    lab_pivot['unique_lab_types'] = (lab_pivot[[col for col in value_cols if 'count' in col]] > 0).sum(axis=1)

    print(f"✓ Created {len(lab_pivot.columns)-2} lab features")

    return lab_pivot