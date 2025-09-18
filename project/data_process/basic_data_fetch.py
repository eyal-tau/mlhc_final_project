ICUQ = """--sql
SELECT admissions.subject_id::INTEGER AS subject_id,
       admissions.hadm_id::INTEGER AS hadm_id,
       admissions.admittime::TIMESTAMP AS admittime,      -- Changed from ::DATE
       admissions.dischtime::TIMESTAMP AS dischtime,      -- Changed from ::DATE
       admissions.ethnicity,
       admissions.deathtime::TIMESTAMP AS deathtime,      -- Changed from ::DATE
       patients.gender,
       patients.dob::DATE AS dob,
       icustays.icustay_id::INTEGER AS icustay_id,
       patients.dod::DATE as dod,
       icustays.intime::TIMESTAMP AS intime,              -- Changed from ::DATE
       icustays.outtime::TIMESTAMP AS outtime             -- Changed from ::DATE
FROM admissions
INNER JOIN patients
    ON admissions.subject_id = patients.subject_id
LEFT JOIN icustays
    ON admissions.hadm_id = icustays.hadm_id
WHERE admissions.has_chartevents_data = 1
AND admissions.subject_id::INTEGER IN ?
ORDER BY admissions.subject_id, admissions.hadm_id, admissions.admittime;
"""

# Updated Lab Query - now with full timestamps
LABQUERY = """--sql
SELECT labevents.subject_id::INTEGER AS subject_id,
       labevents.hadm_id::INTEGER AS hadm_id,
       labevents.charttime::TIMESTAMP AS charttime,       -- Changed from ::DATE
       labevents.itemid::INTEGER AS itemid,
       labevents.valuenum::DOUBLE AS valuenum,
       admissions.admittime::TIMESTAMP AS admittime       -- Changed from ::DATE
FROM labevents
INNER JOIN admissions
    ON labevents.subject_id = admissions.subject_id
    AND labevents.hadm_id = admissions.hadm_id
    AND labevents.charttime::TIMESTAMP between
        (admissions.admittime::TIMESTAMP)
        AND (admissions.admittime::TIMESTAMP + interval 48 hour)
    AND itemid::INTEGER IN ?
"""

# Updated Vitals Query - now with full timestamps
VITQUERY = """--sql
SELECT chartevents.subject_id::INTEGER AS subject_id,
       chartevents.hadm_id::INTEGER AS hadm_id,
       chartevents.charttime::TIMESTAMP AS charttime,     -- Changed from ::DATE
       chartevents.itemid::INTEGER AS itemid,
       chartevents.valuenum::DOUBLE AS valuenum,
       admissions.admittime::TIMESTAMP AS admittime       -- Changed from ::DATE
FROM chartevents
INNER JOIN admissions
    ON chartevents.subject_id = admissions.subject_id
    AND chartevents.hadm_id = admissions.hadm_id
    AND chartevents.charttime::TIMESTAMP between
       (admissions.admittime::TIMESTAMP)
       AND (admissions.admittime::TIMESTAMP + interval 48 hour)
    AND itemid::INTEGER in ?
AND chartevents.error::INTEGER IS DISTINCT FROM 1
"""


def get_icu_data(subject_ids, con):
    icu = con.execute(ICUQ, [subject_ids]).fetchdf().rename(str.lower, axis='columns')
    all_admissions_for_cohort = icu.copy()
    return all_admissions_for_cohort

def get_lab_data(subject_ids, con, lavbevent_metadata):
    lab = con.execute(LABQUERY, [lavbevent_metadata['itemid'].tolist()]).fetchdf().rename(str.lower, axis='columns')
    return lab

def get_vit_data(subject_ids, con, vital_metadata):
    vit = con.execute(VITQUERY, [vital_metadata['itemid'].tolist()]).fetchdf().rename(str.lower, axis='columns')
    return vit

