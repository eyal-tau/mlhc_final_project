import pandas as pd
import matplotlib.pyplot as plt


def filter_early_deaths(cohort):
    """
    Filters out patients who died before 54 hours from admission.
    """
    print("\n" + "="*50)
    print("FILTERING EARLY DEATHS (<54 HOURS)")
    print("="*50)

    cohort = cohort.copy()

    # Ensure datetime types and calculate prediction gap end
    cohort['admittime'] = pd.to_datetime(cohort['admittime'])
    cohort['deathtime'] = pd.to_datetime(cohort['deathtime'])
    cohort['prediction_gap_end'] = cohort['admittime'] + pd.Timedelta(hours=54)

    # Identify patients who died before the prediction gap ended
    early_deaths_filter = (
        cohort['deathtime'].notna() &
        (cohort['deathtime'] < cohort['prediction_gap_end'])
    )

    # Filter out these patients
    filtered_cohort = cohort[~early_deaths_filter].copy()

    # Analysis
    total_patients = len(cohort)
    early_deaths_count = early_deaths_filter.sum()

    print(f"Total patients before filtering: {total_patients:,}")
    print(f"Patients dying before 54 hours: {early_deaths_count:,}")
    print(f"Patients after filtering: {len(filtered_cohort):,}")
    print(f"Excluded: {early_deaths_count:,} ({(early_deaths_count/total_patients)*100:.1f}%)")

    return filtered_cohort


def create_mortality_dataset(final_cohort, run_mode = "train"):
    """
    Create mortality prediction dataset
    Target: Death during hospitalization OR within 30 days after discharge
    """
    print("\n" + "="*50)
    print("CREATING MORTALITY PREDICTION DATASET")
    print("="*50)

    original_columns = final_cohort.columns.tolist()

    # filtering death before prediction gap ends (<54)
    cohort = filter_early_deaths(final_cohort)

    # Ensure datetime types
    cohort['admittime'] = pd.to_datetime(cohort['admittime'])
    cohort['dischtime'] = pd.to_datetime(cohort['dischtime'])
    cohort['deathtime'] = pd.to_datetime(cohort['deathtime'])
    cohort['dod'] = pd.to_datetime(cohort['dod'])

    # Calculate 54-hour mark (end of prediction gap)
    cohort['prediction_gap_end'] = cohort['admittime'] + pd.Timedelta(hours=54)

    # Mortality during hospitalization
    death_during_hosp = (
        cohort['deathtime'].notna() &
        (cohort['deathtime'] >= cohort['prediction_gap_end'])
    )

    # Mortality within 30 days after discharge
    cohort['death_cutoff_30d'] = cohort['dischtime'] + pd.Timedelta(days=30)
    death_within_30d_post_discharge = (
        cohort['dod'].notna() &
        (cohort['dod'] <= cohort['death_cutoff_30d']) &
        (cohort['dod'] >= cohort['prediction_gap_end'])
    )

    # Create mortality target
    cohort['mortality'] = (death_during_hosp | death_within_30d_post_discharge).astype(int)


    # Create mortality dataset
    mortality_dataset = cohort.copy()

    if run_mode == 'test':
        return mortality_dataset[original_columns].copy()
    else:
        return mortality_dataset


def create_prolonged_stay_dataset(final_cohort, run_mode = "train"):
    """
    Create prolonged stay prediction dataset
    Target: Length of stay > 7 days (168 hours)
    """
    print("\n" + "="*50)
    print("CREATING PROLONGED STAY PREDICTION DATASET")
    print("="*50)

    original_columns = final_cohort.columns.tolist()

    cohort = final_cohort.copy()
    # Keep only patients who survived the hospitalization (survivors)
    survivors_filter = cohort['deathtime'].isna()
    cohort = cohort[survivors_filter]
    # Create prolonged stay target
    cohort['prolonged_stay'] = (cohort['los_hours'] > 168).astype(int)

    # Additional features specific to LOS prediction
    cohort['los_days'] = cohort['los_hours'] / 24
    cohort['is_weekend_admission'] = pd.to_datetime(cohort['admittime']).dt.dayofweek >= 5

    # Create prolonged stay dataset
    prolonged_stay_dataset = cohort.copy()

    if run_mode == 'test':
        return prolonged_stay_dataset[original_columns].copy()
    else:
        return prolonged_stay_dataset


# =============================================================================
# DATASET 3: READMISSION PREDICTION DATASET
# =============================================================================
def create_readmission_dataset(
    final_cohort: pd.DataFrame,
    all_cohort_admissions: pd.DataFrame,
    min_gap_hours: int = 12,                # treat <12h as transfers (not readmissions)
    max_gap_days: int = 30,                 # 30-day hospital readmission window
    death_post_discharge_as_readdmition: bool = False,  # count death ≥54h post-admit & ≤30d post-discharge as "missed readmission"
    run_mode = "train"
) -> pd.DataFrame:
    """
    Target: Hospital readmission within 30 days after discharge
    (not to be confused with ICU readmission within the same hospital admission).

    Returns:
        DataFrame = final_cohort with:
          - readmission_30d (bool): next hospital admission within window
          - days_to_readmission (int or None): days to first qualifying readmission
          - readmission (bool): readmission_30d OR (optional death-based proxy)
          - plus helper columns: hours_to_next, next_hadm, next_admit (for audit)
    """
    print("\n" + "="*50)
    print("CREATING READMISSION PREDICTION DATASET")
    print("="*50)

    original_columns = final_cohort.columns.tolist()

    cohort = final_cohort.copy()
    admits_src = all_cohort_admissions.copy()

    print(f"Processing {len(cohort):,} patients for readmission analysis...")

    # --- Collapse to ONE ROW per hospital admission (avoid ICU-duplication within hadm) ---
    admits = (
        admits_src
        .groupby(["subject_id", "hadm_id"], as_index=False)
        .agg(admittime=("admittime", "min"), dischtime=("dischtime", "max"))
        .sort_values(["subject_id", "admittime"])
    )

    # --- Next admission per subject (vectorized) ---
    admits["next_admit"] = admits.groupby("subject_id")["admittime"].shift(-1)
    admits["next_hadm"]  = admits.groupby("subject_id")["hadm_id"].shift(-1)

    # --- Gap from this discharge to next admit (hours) ---
    admits["hours_to_next"] = (
        (admits["next_admit"] - admits["dischtime"]).dt.total_seconds() / 3600.0
    )

    max_gap_hours = max_gap_days * 24

    # --- Readmission within window (policy: after discharge, >= min_gap, <=30d) ---
    admits["readmission_30d"] = (
        admits["hours_to_next"].ge(min_gap_hours) &
        admits["hours_to_next"].le(max_gap_hours)
    ).fillna(False)

    # --- Merge back to cohort (bring helper cols too) ---
    out = cohort.merge(
    admits[["hadm_id", "readmission_30d", "hours_to_next", "next_hadm", "next_admit"]],
    on="hadm_id",
    how="left"
    )

    out["readmission_30d"] = out["readmission_30d"].fillna(False)

    # death-based "missed readmission" augmentation
    if death_post_discharge_as_readdmition:
        death_ts = pd.to_datetime(out.get("dod"), errors="coerce")
        out["hours_to_death_after_discharge"] = (
            (death_ts - out["dischtime"]).dt.total_seconds() / 3600.0
        )
        out["hours_from_admit_to_death"] = (
            (death_ts - out["admittime"]).dt.total_seconds() / 3600.0
        )
        out["death_post_discharge"] = (
            out["next_hadm"].isna() &                                  # no captured readmit
            out["hours_to_death_after_discharge"].gt(0) &              # death after discharge
            out["hours_from_admit_to_death"].ge(54) &                  # ≥54h post-index admit
            out["hours_to_death_after_discharge"].le(max_gap_hours)    # within 30 days post-discharge
        ).fillna(False)
        out["readmission"] = out["readmission_30d"] | out["death_post_discharge"]
        real_readmissions = int(out["readmission_30d"].fillna(False).astype(bool).sum())
        print(f"Real readmissions count: {real_readmissions}")
        missed_readmissions = int(out["death_post_discharge"].fillna(False).astype(bool).sum())
        print(f"Death readmissions count: {missed_readmissions}")
    else:
        out["readmission"] = out["readmission_30d"]

    # ----------------------- Stats prints (like your original) -----------------------
    total_patients = len(out)
    readmission_count = int(out["readmission"].sum())
    readmission_rate = readmission_count / total_patients * 100

    print("Readmission Dataset Summary:")
    print(f"  Total patients: {total_patients:,}")
    print(f"  Readmission events: {readmission_count:,}")
    print(f"  Readmission rate: {readmission_rate:.1f}%")

    labels = (
        out[["hadm_id", "readmission"]]
        .drop_duplicates("hadm_id")          # ensure 1 row per admission
    )
    result = final_cohort.copy()
    result = result.merge(labels, on="hadm_id", how="left")
    result["readmission"] = result["readmission"].fillna(False)

    if run_mode == "test":
        return result[original_columns].copy()
    else:
        return result


# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def create_separate_target_datasets(final_cohort, all_admissions_for_cohort, run_mode="train"):
    """
    Execute complete process to create separate target datasets
    """
    print("CREATING SEPARATE TARGET DATASETS")

    # Create individual datasets
    print("Step 1: Creating mortality dataset...")
    mortality_dataset = create_mortality_dataset(final_cohort, run_mode=run_mode)

    print("Step 2: Creating prolonged stay dataset...")
    prolonged_stay_dataset = create_prolonged_stay_dataset(final_cohort, run_mode=run_mode)

    print("Step 3: Creating readmission dataset...")
    readmission_dataset = create_readmission_dataset(final_cohort, all_admissions_for_cohort, death_post_discharge_as_readdmition=True, run_mode=run_mode)

    return {
        'mortality': mortality_dataset,
        'prolonged_stay': prolonged_stay_dataset,
        'readmission': readmission_dataset,
    }
