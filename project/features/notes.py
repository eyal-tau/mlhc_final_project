def create_notes_based_features(
    final_cohort_with_targets,
    notes_df,
    radiology_map_df,
    note_categories,
    radiology_category_label="Radiology",
    description_col_in_notes="description",   # column in notes_df to join on
    description_col_in_map="description",     # column in radiology_map_df to join on
    anatomy_col_in_map="class",               # anatomy/system column in your mapping
    invasive_col_in_map="is_invasive"         # 'Yes'/'No' column in your mapping
):
    """
    Build features:
      - Per-patient/per-admission counts of notes by category (for the categories you list)
      - Radiology anatomy presence flags (0/1) via mapping table
      - had_invasive flag (0/1) if any mapped radiology exam is invasive
    Returns a DataFrame keyed by (subject_id, hadm_id)
    """
    print("\n" + "="*50)
    print("CREATING NOTES-BASED FEATURES")
    print("="*50)

    # Base key
    base = final_cohort_with_targets[["subject_id", "hadm_id"]].drop_duplicates().copy()

    # --------------------------
    # A) Counts by note category
    # --------------------------
    notes = notes_df.copy()
    notes["category"] = notes["category"].astype(str)

    # Keep only categories you care about (stable set of columns)
    notes_for_counts = notes[notes["category"].isin(note_categories)].copy()

    note_counts = (
        notes_for_counts
        .groupby(["subject_id", "hadm_id", "category"])
        .size()
        .unstack("category", fill_value=0)
        .reindex(columns=note_categories, fill_value=0)
    )
    note_counts.columns = [f"notes_cnt__{c.lower().replace(' ','_')}" for c in note_counts.columns]
    print(f"✓ Note-category features: {len(note_counts.columns)} columns")

    # -----------------------------------------------------
    # B) Radiology anatomy flags + had_invasive using map DF
    # -----------------------------------------------------
    # 1) Filter notes to radiology category
    rad_notes = notes[notes["category"] == radiology_category_label].copy()

    # 2) Normalize description in both dataframes to join robustly
    def _norm_desc(s):
        if pd.isna(s):
            return np.nan
        s = str(s).strip().upper()
        s = re.sub(r"\s+", " ", s)
        return s

    rad_notes["_desc_key"] = rad_notes[description_col_in_notes].map(_norm_desc)
    radiology_map_df = radiology_map_df.copy()
    radiology_map_df["_desc_key"] = radiology_map_df[description_col_in_map].map(_norm_desc)

    # 3) Join to get anatomy and invasiveness
    rad_join = rad_notes.merge(
        radiology_map_df[["_desc_key", anatomy_col_in_map, invasive_col_in_map]],
        on="_desc_key",
        how="left",
        validate="m:1"
    )

    # ---- 4) COUNTS per anatomy (integers), excluding Irrelevant ----
    rad_clean = rad_join[
        rad_join[anatomy_col_in_map].notna() &
        ~rad_join[anatomy_col_in_map].astype(str).str.strip().str.upper().eq("IRRELEVANT")
    ].copy()

    rad_counts = (
        rad_clean
        .groupby(["subject_id", "hadm_id", anatomy_col_in_map])
        .size()
        .unstack(anatomy_col_in_map, fill_value=0)
    )

    rad_counts.columns = [
        f"notes_rad_cnt__{str(c).lower().replace('/','_').replace(' ','_')}"
        for c in rad_counts.columns
    ]
    print(f"✓ Radiology anatomy counts: {len(rad_counts.columns)} columns")

    # invasive counts
    rad_clean["_inv"] = rad_clean[invasive_col_in_map].astype(str).str.upper().eq("YES").astype(int)
    had_invasive = (
        rad_clean.groupby(["subject_id","hadm_id"])["_inv"].max()
                 .to_frame("notes_rad_has__invasive")
    )

    inv_total = (
        rad_clean.groupby(["subject_id","hadm_id"])["_inv"].sum()
                 .to_frame("notes_rad_cnt__invasive")
    )
    print(f"✓ Radiology invasive counts added")

    # invasive counts per anatomy
    inv_by_anat = (
        rad_clean.groupby(["subject_id","hadm_id", anatomy_col_in_map])["_inv"].sum()
                 .unstack(anatomy_col_in_map, fill_value=0)
    )
    inv_by_anat.columns = [
        f"notes_rad_cnt_invasive__{str(c).lower().replace('/','_').replace(' ','_')}"
        for c in inv_by_anat.columns
    ]
    print(f"✓ Radiology anatomy invasive counts: {len(inv_by_anat.columns)} columns")


    # -------------------
    # C) Combine features
    # -------------------
    features = (
        base
        .merge(note_counts, on=["subject_id","hadm_id"], how="left")
        # .merge(rad_presence, on=["subject_id","hadm_id"], how="left")
        .merge(rad_counts, on=["subject_id","hadm_id"], how="left")
        # .merge(had_invasive, on=["subject_id","hadm_id"], how="left")
        .merge(inv_total, on=["subject_id","hadm_id"], how="left")
        .merge(inv_by_anat, on=["subject_id","hadm_id"], how="left")
        .fillna(0)
    )

    # Ensure integer type for 0/1/counts
    int_cols = [c for c in features.columns if c not in ["subject_id","hadm_id"]]
    features[int_cols] = features[int_cols].astype(int)

    print(f"✓ Final notes-based feature matrix: {features.shape[0]:,} rows × {features.shape[1]-2} features")
    return features
