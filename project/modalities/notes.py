def extract_notes_data(final_cohort, con, hours=48, categories=None):
    """
    Extract notes data for the unified cohort
    """
    print("\n--- Extracting Notes Data ---")

    cohort_hadm_ids = final_cohort['hadm_id'].tolist()
    cohort_subject_ids = final_cohort['subject_id'].tolist()
    print(f"Extracting notes for {len(cohort_hadm_ids):,} admissions...")

    notes_query = f"""
    WITH notes_cast AS (
      SELECT
          CAST(n.subject_id AS INTEGER)   AS subject_id,
          CAST(n.hadm_id    AS INTEGER)   AS hadm_id,
          n.category,
          n.description,
          CAST(n.charttime  AS TIMESTAMP) AS charttime,
          CAST(a.admittime  AS TIMESTAMP) AS admittime
      FROM noteevents n
      INNER JOIN admissions a
        ON CAST(n.hadm_id AS INTEGER) = CAST(a.hadm_id AS INTEGER)
      WHERE CAST(n.hadm_id    AS INTEGER) IN ?
        AND CAST(n.subject_id AS INTEGER) IN ?
    )
    SELECT
        subject_id,
        hadm_id,
        category,
        description
    FROM notes_cast
    WHERE charttime IS NOT NULL
      AND charttime <= admittime + INTERVAL '{hours} hours'
      {("AND category IN ?" if categories else "")}
    ORDER BY subject_id, hadm_id, charttime
    """

    bind = [cohort_hadm_ids, cohort_subject_ids]
    if categories:
        bind.append(categories)

    notes_data = (
        con.execute(notes_query, bind)
        .fetchdf()
        .rename(str.lower, axis='columns')
    )

    # Ensure description is a string, no NaNs
    notes_data['description'] = notes_data['description'].fillna('').astype(str)

    print(f"✓ Notes data: {len(notes_data)} records")
    print(f"✓ Patients with notes: {notes_data['subject_id'].nunique()}")

    return notes_data
