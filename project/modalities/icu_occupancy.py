import pandas as pd

def extract_icu_occupancy_data(final_cohort, con, clause="load_window", window_hours=12, same_unit=False):
    """
    Extract ICU occupancy data

    For each (subject_id, hadm_id) in the cohort, compute concurrent ICU stays
    around the *earliest ICU stay* of that admission.

    Returns: one row per (subject_id, hadm_id)
             ['icustay_id','subject_id','hadm_id','intime','first_careunit','icu_occupancy_count']
    """
    cohort_hadm_ids = final_cohort['hadm_id'].tolist()
    cohort_subject_ids = final_cohort['subject_id'].tolist()
    print(f"Extracting occupancy for {len(cohort_hadm_ids):,} admissions...")

    # overlap semantics
    if clause == "load_window":
        overlap_clause = (
            f"s2.intime  <= s1.intime + INTERVAL '{window_hours} hours' "
            f"AND s2.outtime >  s1.intime - INTERVAL '{window_hours} hours'"
        )
    elif clause == "current_load":
        overlap_clause = (
            "s2.intime <= s1.intime "
            "AND s2.outtime > s1.intime"
        )
    elif clause == "new_arrivals_window":
        overlap_clause = (
            f"s2.intime >= s1.intime - INTERVAL '{window_hours} hours' "
            f"AND s2.intime <= s1.intime + INTERVAL '{window_hours} hours'"
        )
    else:
        raise ValueError("clause must be 'load_window' or 'current_load' or 'new_arrivals_window'")

    unit_clause = "AND s2.first_careunit = s1.first_careunit" if same_unit else ""

    query = f"""
    WITH stays AS (
      SELECT
        CAST(icustay_id AS INTEGER)   AS icustay_id,
        CAST(subject_id AS INTEGER)   AS subject_id,
        CAST(hadm_id    AS INTEGER)   AS hadm_id,
        CAST(intime     AS TIMESTAMP) AS intime,
        CAST(outtime    AS TIMESTAMP) AS outtime,
        first_careunit,
        ROW_NUMBER() OVER (PARTITION BY subject_id, hadm_id ORDER BY intime) AS rn
      FROM icustays
      WHERE intime IS NOT NULL AND outtime IS NOT NULL
    ),
    cohort_first AS (
      SELECT *
      FROM stays
      WHERE rn = 1
        AND hadm_id IN ?
        AND subject_id IN ?
    ),
    counts AS (
      SELECT
        s1.icustay_id,
        s1.subject_id,
        s1.hadm_id,
        s1.intime,
        s1.first_careunit,
        COUNT(*) FILTER (WHERE s2.icustay_id IS NOT NULL) AS icu_occupancy_count
      FROM cohort_first s1
      LEFT JOIN stays s2
        ON s2.icustay_id <> s1.icustay_id
       AND {overlap_clause}
       {unit_clause}
      GROUP BY s1.icustay_id, s1.subject_id, s1.hadm_id, s1.intime, s1.first_careunit
    )
    SELECT *
    FROM counts
    ORDER BY subject_id, hadm_id, intime
    """

    return (con.execute(query, [cohort_hadm_ids, cohort_subject_ids])
            .fetchdf()
            .rename(str.lower, axis="columns"))


def extract_all_icu_occupancy_data(final_cohort, con):
    """
    Extract all ICU occupancy variants from the unified cohort
    """
    print("\n--- Extracting ICU Occupancy Data ---")

    # Extract all ICU occupancy variants
    icu_lw_12h = extract_icu_occupancy_data(final_cohort, con, clause="load_window", window_hours=12)
    print(f"✓ ICU load window 12h: {len(icu_lw_12h)} records")

    icu_naw_12h = extract_icu_occupancy_data(final_cohort, con, clause="new_arrivals_window", window_hours=12)
    print(f"✓ ICU new arrivals window 12h: {len(icu_naw_12h)} records")

    icu_naw_24h = extract_icu_occupancy_data(final_cohort, con, clause="new_arrivals_window", window_hours=24)
    print(f"✓ ICU new arrivals window 24h: {len(icu_naw_24h)} records")

    icu_cl = extract_icu_occupancy_data(final_cohort, con, clause="current_load")
    print(f"✓ ICU current load: {len(icu_cl)} records")

    # Extract all ICU occupancy variants considering same unit
    icu_lw_12h_same_unit = extract_icu_occupancy_data(final_cohort, con, clause="load_window", same_unit=True,
                                                      window_hours=12)
    print(f"✓ ICU load window 12h same_unit: {len(icu_lw_12h)} records")

    icu_naw_12h_same_unit = extract_icu_occupancy_data(final_cohort, con, clause="new_arrivals_window", same_unit=True,
                                                       window_hours=12)
    print(f"✓ ICU new arrivals window 12h same_unit: {len(icu_naw_12h)} records")

    icu_naw_24h_same_unit = extract_icu_occupancy_data(final_cohort, con, clause="new_arrivals_window", same_unit=True,
                                                       window_hours=24)
    print(f"✓ ICU new arrivals window 24h same_unit: {len(icu_naw_24h)} records")

    icu_cl_same_unit = extract_icu_occupancy_data(final_cohort, con, clause="current_load", same_unit=True)
    print(f"✓ ICU current load same_unit: {len(icu_cl)} records")

    return {
        "icu_lw_12h": icu_lw_12h,
        "icu_lw_12h_same_unit": icu_lw_12h_same_unit,
        "icu_naw_12h": icu_naw_12h,
        "icu_naw_12h_same_unit": icu_naw_12h_same_unit,
        "icu_naw_24h": icu_naw_24h,
        "icu_naw_24h_same_unit": icu_naw_24h_same_unit,
        "icu_cl": icu_cl,
        "icu_cl_same_unit": icu_cl_same_unit
    }