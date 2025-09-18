from .helper import aggregate_temporal_features


def create_vital_features(vit_filtered, final_cohort_with_targets):
    """
    Create features from vital signs data
    """
    print("\n" + "="*50)
    print("CREATING VITAL SIGNS FEATURES")
    print("="*50)

    # Aggregate vital measurements
    vital_agg = aggregate_temporal_features(
        vit_filtered,
        ['subject_id', 'hadm_id', 'itemid'],
        'valuenum',
        'vital'
    )

    # Pivot to create one column per vital sign per statistic
    vital_pivot = vital_agg.pivot_table(
        index=['subject_id', 'hadm_id'],
        columns='itemid',
        values=['count', 'mean', 'std', 'min', 'max', 'range', 'cv', 't_range'],
        fill_value=0
    )

    # Flatten column names
    vital_pivot.columns = [f'vital_{stat}_{itemid}' for stat, itemid in vital_pivot.columns]
    vital_pivot = vital_pivot.reset_index()

    # Create summary features across all vitals
    value_cols = [col for col in vital_pivot.columns if col.startswith('vital_')]
    vital_pivot['total_vital_measurements'] = vital_pivot[[col for col in value_cols if 'count' in col]].sum(axis=1)
    vital_pivot['unique_vital_types'] = (vital_pivot[[col for col in value_cols if 'count' in col]] > 0).sum(axis=1)

    print(f"✓ Created {len(vital_pivot.columns)-2} vital features")

    return vital_pivot
