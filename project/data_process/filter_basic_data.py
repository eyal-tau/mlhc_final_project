import pandas as pd


def analyze_multiple_admissions(icu):
    """Analyze patients with multiple admissions"""
    print("\n" + "="*60)
    print("MULTIPLE ADMISSIONS ANALYSIS")
    print("="*60)

    # Count admissions per patient
    admissions_per_patient = icu.groupby('subject_id')['hadm_id'].nunique().reset_index()
    admissions_per_patient.columns = ['subject_id', 'num_admissions']

    # Statistics
    print(f"Total unique patients: {len(admissions_per_patient):,}")
    print(f"Patients with 1 admission: {(admissions_per_patient['num_admissions'] == 1).sum():,}")
    print(f"Patients with >1 admission: {(admissions_per_patient['num_admissions'] > 1).sum():,}")
    print(f"Max admissions per patient: {admissions_per_patient['num_admissions'].max()}")

    # Distribution of number of admissions
    admission_dist = admissions_per_patient['num_admissions'].value_counts().sort_index()
    print(f"\nDistribution of admissions per patient:")
    for num_adm, count in admission_dist.head(10).items():
        print(f"  {num_adm} admission(s): {count:,} patients")

    return admissions_per_patient

def filter_age(cohort):
    # Age calculation
    cohort = cohort.copy()
    cohort['admittime'] = pd.to_datetime(cohort['admittime'])
    cohort['dob'] = pd.to_datetime(cohort['dob'])
    cohort['age_at_admission'] = (
        cohort['admittime'] - cohort['dob']
    ).dt.days / 365.25

    # Filter for patients under 18 OR over 90
    age_filter = (cohort['age_at_admission'] >= 18) & (cohort['age_at_admission'] < 90)
    return cohort[age_filter]

def filter_first_admissions(icu):
    """Filter to keep only first hospital admissions per patient"""
    print("\n" + "="*60)
    print("FILTERING FOR FIRST ADMISSIONS")
    print("="*60)

    # Sort by patient and admission time to get first admission
    icu_sorted = icu.sort_values(['subject_id', 'admittime'])
    first_admissions = icu_sorted.groupby('subject_id').first().reset_index()

    print(f"Original ICU records: {len(icu):,}")
    print(f"After filtering to first admissions: {len(first_admissions):,}")
    print(f"Unique patients: {first_admissions['subject_id'].nunique():,}")

    return first_admissions

def calculate_precise_los_and_filter(first_admissions):
    """Calculate precise length of stay and apply 54-hour filter"""
    print("\n" + "="*60)
    print("PRECISE LENGTH OF STAY CALCULATION AND 54-HOUR FILTER")
    print("="*60)

    # Convert to datetime for calculation
    first_admissions = first_admissions.copy()
    first_admissions['admittime'] = pd.to_datetime(first_admissions['admittime'])
    first_admissions['dischtime'] = pd.to_datetime(first_admissions['dischtime'])

    # Calculate length of stay
    time_diff = first_admissions['dischtime'] - first_admissions['admittime']
    first_admissions['los_hours'] = time_diff.dt.total_seconds() / 3600
    first_admissions['los_days'] = first_admissions['los_hours'] / 24
    first_admissions['los_minutes'] = time_diff.dt.total_seconds() / 60

    # Show example calculations
    print("Example precise calculations (first 5 patients):")
    sample = first_admissions[['subject_id', 'admittime', 'dischtime', 'los_hours', 'los_days']].head(5)
    for idx, row in sample.iterrows():
        print(f"  Patient {row['subject_id']}:")
        print(f"    {row['admittime']} → {row['dischtime']}")
        print(f"    LOS: {row['los_hours']:.2f} hours ({row['los_days']:.2f} days)")

    # Statistics with precise timestamps
    print(f"\nPrecise LOS Statistics:")
    print(f"  Mean: {first_admissions['los_hours'].mean():.2f} hours ({first_admissions['los_days'].mean():.2f} days)")
    print(f"  Median: {first_admissions['los_hours'].median():.2f} hours ({first_admissions['los_days'].median():.2f} days)")
    print(f"  Min: {first_admissions['los_hours'].min():.2f} hours ({first_admissions['los_days'].min():.2f} days)")
    print(f"  Max: {first_admissions['los_hours'].max():.2f} hours ({first_admissions['los_days'].max():.2f} days)")

    # Apply 54-hour filter
    los_54h_filter = first_admissions['los_hours'] >= 54
    filtered_cohort = first_admissions[los_54h_filter].copy()

    meets_54h = len(filtered_cohort)
    total = len(first_admissions)

    print(f"\n54-Hour Filter Results:")
    print(f"  Before ≥54h filter: {total:,} patients")
    print(f"  After ≥54h filter: {meets_54h:,} patients")
    print(f"  Excluded: {total - meets_54h:,} patients ({(total - meets_54h)/total*100:.1f}%)")

    # Show distribution around the 54-hour cutoff
    print(f"\nLOS distribution around 54-hour cutoff:")
    ranges = [
        (0, 24, "0-24 hours"),
        (24, 48, "24-48 hours"),
        (48, 54, "48-54 hours (excluded)"),
        (54, 72, "54-72 hours (included)"),
        (72, 168, "72-168 hours (3-7 days)"),
        (168, float('inf'), "> 168 hours (> 7 days)")
    ]

    for min_h, max_h, label in ranges:
        if max_h == float('inf'):
            count = (first_admissions['los_hours'] >= min_h).sum()
        else:
            count = ((first_admissions['los_hours'] >= min_h) & (first_admissions['los_hours'] < max_h)).sum()
        percentage = count / total * 100
        print(f"  {label}: {count:,} patients ({percentage:.1f}%)")

    return filtered_cohort

def check_48h_data_availability(filtered_cohort, lab, vit):
    """Check which patients have data available in first 48 hours"""
    print("\n" + "="*60)
    print("48-HOUR DATA AVAILABILITY CHECK")
    print("="*60)

    # Get admission IDs from filtered cohort
    valid_hadm_ids = set(filtered_cohort['hadm_id'].unique())

    # Filter lab and vital data to only include valid admissions
    lab_filtered = lab[lab['hadm_id'].isin(valid_hadm_ids)].copy()
    vit_filtered = vit[vit['hadm_id'].isin(valid_hadm_ids)].copy()

    # Check patients with lab data in first 48h
    lab_patients_48h = set(lab_filtered['subject_id'].unique())
    vit_patients_48h = set(vit_filtered['subject_id'].unique())

    # Patients in filtered cohort
    cohort_patients = set(filtered_cohort['subject_id'].unique())

    print(f"Filtered cohort patients: {len(cohort_patients):,}")
    print(f"Patients with lab data in 48h: {len(lab_patients_48h):,}")
    print(f"Patients with vital data in 48h: {len(vit_patients_48h):,}")
    print(f"Patients with both lab & vital data: {len(lab_patients_48h & vit_patients_48h):,}")

    # Final cohort: patients with at least some data in first 48h
    patients_with_data = lab_patients_48h | vit_patients_48h
    final_cohort_patients = cohort_patients & patients_with_data

    print(f"Final cohort with 48h data: {len(final_cohort_patients):,}")

    # Filter final cohort
    final_cohort = filtered_cohort[
        filtered_cohort['subject_id'].isin(final_cohort_patients)
    ].copy()

    return final_cohort, lab_filtered, vit_filtered


def analyze_final_cohort_characteristics(final_cohort):
    # Age calculation
    final_cohort = final_cohort.copy()
    final_cohort['admittime'] = pd.to_datetime(final_cohort['admittime'])
    final_cohort['dob'] = pd.to_datetime(final_cohort['dob'])
    final_cohort['age_at_admission'] = (
        final_cohort['admittime'] - final_cohort['dob']
    ).dt.days / 365.25

    return final_cohort

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================
def basic_filter(icu, lab, vit):
    print("\nEXECUTING COMPLETE COHORT ANALYSIS")
    # Step 1: Filter age < 18 and > 90
    cohort = filter_age(icu)
    # Step 2: Analyze multiple admissions
    initial_count = len(set(cohort['subject_id'].unique()))
    admissions_analysis = analyze_multiple_admissions(cohort)

    # Step 3: Filter for first admissions
    first_admissions = filter_first_admissions(cohort)
    first_adm_count = len(first_admissions)

    # Step 4: Calculate precise LOS and filter for ≥54 hours
    filtered_cohort = calculate_precise_los_and_filter(first_admissions)
    los_54h_count = len(filtered_cohort)

    # Step 5: Check 48-hour data availability
    final_cohort, lab_filtered, vit_filtered = check_48h_data_availability(
        filtered_cohort, lab, vit
    )
    final_count = len(final_cohort)

    # Step 7: Analyze final cohort
    final_cohort = analyze_final_cohort_characteristics(final_cohort)

    return final_cohort, lab_filtered, vit_filtered
