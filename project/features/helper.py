import numpy as np

def aggregate_temporal_features(data, groupby_cols, value_col='valuenum', prefix=''):
    """
    Aggregate time-series data into statistical features

    Args:
        data: DataFrame with temporal measurements
        groupby_cols: List of columns to group by (usually ['subject_id', 'hadm_id', 'itemid'])
        value_col: Column containing values to aggregate
        prefix: Prefix for feature names
    """
    print(f"  Aggregating {len(data):,} {prefix} measurements...")

    # Aggregate features for each patient-item combination
    agg_features = data.groupby(groupby_cols)[value_col].agg([
        'count',    # Number of measurements
        'mean',     # Average value
        'std',      # Variability
        'min',      # Minimum value
        'max',      # Maximum value
        'first',    # First measurement
        'last',     # Last measurement
    ]).reset_index()

    agg_features['count'] = agg_features['count'].fillna(0).astype(int)

    # Add derived features
    agg_features['range'] = agg_features['max'] - agg_features['min']
    agg_features['cv'] = agg_features['std'] / agg_features['mean']  # Coefficient of variation
    agg_features['t_range'] = (agg_features['last'] - agg_features['first'])

    # Handle infinite/NaN values from division
    agg_features = agg_features.replace([np.inf, -np.inf], np.nan)

    print(f"  ✓ Created {len(agg_features):,} aggregated measurements")

    return agg_features