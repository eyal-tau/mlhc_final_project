import pandas as pd

def extract_medications_data(final_cohort, con):
    """
    Extract medications data for the unified cohort
    """
    print("\n--- Extracting Medications Data ---")

    cohort_hadm_ids = final_cohort['hadm_id'].tolist()
    cohort_subject_ids = final_cohort['subject_id'].tolist()
    print(f"Extracting medications for {len(cohort_hadm_ids):,} admissions...")

    medications_query = """
    SELECT
        p.subject_id::INTEGER AS subject_id,
        p.hadm_id::INTEGER AS hadm_id,
        p.startdate::TIMESTAMP AS startdate,
        p.enddate::TIMESTAMP AS enddate,
        p.drug AS drug,
        p.drug_type AS drug_type,
        p.formulary_drug_cd,
        p.route,
        p.dose_val_rx,
        p.dose_unit_rx,
        a.admittime::TIMESTAMP AS admittime
    FROM prescriptions p
    INNER JOIN admissions a ON p.hadm_id = a.hadm_id
    WHERE p.hadm_id::INTEGER IN ?
    AND p.subject_id::INTEGER IN ?
    AND p.startdate::TIMESTAMP <= a.admittime::TIMESTAMP + interval '48 hours'
    ORDER BY p.subject_id, p.hadm_id, p.startdate
    """

    medications_data = con.execute(
        medications_query,
        [cohort_hadm_ids, cohort_subject_ids]
    ).fetchdf().rename(str.lower, axis='columns')

    print(f"✓ Medications data: {len(medications_data)} prescription records")
    print(f"✓ Patients with medication data: {medications_data['subject_id'].nunique()}")

    return medications_data