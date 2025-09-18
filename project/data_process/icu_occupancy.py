import pickle
import numpy as np


def extract_global_icu_occupancy(con, window_hours=12, *,
                                 clause='load_window',
                                 level='per_patient',      # 'per_patient' or 'per_admission'
                                 same_unit=False):          # overlap within same first_careunit only
    """
    Global ICU occupancy: for each index ICU stay (based on `level`),
    count how many *other* ICU stays overlap around its intime.

    Returns columns:
      ['icustay_id','subject_id','hadm_id','intime','first_careunit','concurrent_count']
    """

    # choose partition for "first" index stay
    if level == 'per_admission':
        partition = "subject_id, hadm_id"
    elif level == 'per_patient':
        partition = "subject_id"
    else:
        raise ValueError("level must be 'per_patient' or 'per_admission'")

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
        ROW_NUMBER() OVER (PARTITION BY {partition} ORDER BY intime) AS rn
      FROM icustays
      WHERE intime IS NOT NULL AND outtime IS NOT NULL
    ),
    index_stays AS (
      SELECT * FROM stays WHERE rn = 1
    ),
    counts AS (
      SELECT
        s1.icustay_id,
        s1.subject_id,
        s1.hadm_id,
        s1.intime,
        s1.first_careunit,
        COUNT(*) FILTER (WHERE s2.icustay_id IS NOT NULL) AS concurrent_count
      FROM index_stays s1
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

    df = (con.execute(query).fetchdf()
            .rename(str.lower, axis="columns"))
    return df


def build_cutoff_mapping(cutoffs, ascending=True):
    cuts = sorted(cutoffs)
    labels = list(range(1, len(cuts) + 2))
    if not ascending:
        labels = labels[::-1]
    # list of (upper_bound, score)
    return list(zip(cuts + [float('inf')], labels))


def score_value(x, mapping):
    for ub, sc in mapping:
        if x <= ub:
            return sc


def gen_cutoffs(occ_concurrent_counts, num_of_bins: int = 10):
    lo, hi = occ_concurrent_counts.min(), occ_concurrent_counts.max()
    edges = np.linspace(lo, hi, num=num_of_bins+1)
    cutoffs = edges[1:-1].tolist()  # inner cutoffs
    return cutoffs


def get_mapping_by_window(con, window_hours=12, clause="load_window", same_unit=False, num_bins: int = 5):
    occ = extract_global_icu_occupancy(con, window_hours=window_hours,
                                       clause=clause,
                                       level='per_admission',
                                       same_unit=same_unit)
    cutoffs = gen_cutoffs(occ["concurrent_count"], num_of_bins=num_bins)
    mapping = build_cutoff_mapping(cutoffs, ascending=True)
    return occ, mapping


def gen_mapping_global(con):
    occ_12h_lw, mapping_12h_lw = get_mapping_by_window(con, window_hours=12, clause="load_window", num_bins=10)
    occ_12h_lw['icu_occupancy_score'] = occ_12h_lw['concurrent_count'].apply(lambda v: score_value(v, mapping_12h_lw)).astype(int)

    occ_12h_naw, mapping_12h_naw = get_mapping_by_window(con, window_hours=12, clause="new_arrivals_window", num_bins=5)
    occ_12h_naw['icu_occupancy_score'] = occ_12h_naw['concurrent_count'].apply(lambda v: score_value(v, mapping_12h_naw)).astype(int)

    occ_24h_naw, mapping_24h_naw = get_mapping_by_window(con, window_hours=24, clause="new_arrivals_window", num_bins=5)
    occ_24h_naw['icu_occupancy_score'] = occ_24h_naw['concurrent_count'].apply(lambda v: score_value(v, mapping_24h_naw)).astype(int)

    occ_cl, mapping_cl = get_mapping_by_window(con, window_hours=12, clause="current_load", num_bins=10)
    occ_cl['icu_occupancy_score'] = occ_cl['concurrent_count'].apply(lambda v: score_value(v, mapping_cl)).astype(int)

    mapping_data = {
        'icu_lw_12h': mapping_12h_lw,
        'icu_naw_12h': mapping_12h_naw,
        'icu_naw_24h': mapping_24h_naw,
        'icu_cl': mapping_cl
    }
    return mapping_data


def gen_mapping(con):
    occ_12h_lw, mapping_12h_lw = get_mapping_by_window(con, window_hours=12, clause="load_window", same_unit=True, num_bins=5)
    occ_12h_lw['icu_occupancy_score'] = occ_12h_lw['concurrent_count'].apply(lambda v: score_value(v, mapping_12h_lw)).astype(int)

    occ_12h_naw, mapping_12h_naw = get_mapping_by_window(con, window_hours=12, clause="new_arrivals_window", same_unit=True, num_bins=3)
    occ_12h_naw['icu_occupancy_score'] = occ_12h_naw['concurrent_count'].apply(lambda v: score_value(v, mapping_12h_naw)).astype(int)

    occ_24h_naw, mapping_24h_naw = get_mapping_by_window(con, window_hours=24, clause="new_arrivals_window", same_unit=True, num_bins=3)
    occ_24h_naw['icu_occupancy_score'] = occ_24h_naw['concurrent_count'].apply(lambda v: score_value(v, mapping_24h_naw)).astype(int)

    occ_cl, mapping_cl = get_mapping_by_window(con, window_hours=12, clause="current_load", same_unit=True, num_bins=5)
    occ_cl['icu_occupancy_score'] = occ_cl['concurrent_count'].apply(lambda v: score_value(v, mapping_cl)).astype(int)

    mapping_data = {
        'icu_lw_12h_same_unit': mapping_12h_lw,
        'icu_naw_12h_same_unit': mapping_12h_naw,
        'icu_naw_24h_same_unit': mapping_24h_naw,
        'icu_cl_same_unit': mapping_cl
    }
    return mapping_data


def prep_map_file(con):
    global_map = gen_mapping_global(con)
    same_unit_map = gen_mapping(con)
    combined_map = {**global_map, **same_unit_map}

    # write pickle object
    with open("icu_load_mapping.pkl", 'wb') as f:
        pickle.dump(combined_map, f)