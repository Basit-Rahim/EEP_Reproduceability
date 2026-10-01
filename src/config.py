"""Settings shared by all notebooks: paths, districts, predictors and modelling rules."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "raw_data" / "stata data"     # PSLM 2019-20 Stata files (not distributed)
DATA_DIR = ROOT / "data"                       # analysis dataset (notebook 01)
RESULTS_DIR = ROOT / "results"                 # tables (notebooks 02 and 03)

AGE_MIN, AGE_MAX = 5, 16
SEED = 42
TARGET = "education_access"                    # 1 = currently attending school

DISTRICTS = {
    # key: (PSLM district code, name, district type, Multidimensional Poverty Index)
    "karachi_central": (307, "Karachi Central", "Federal and Provincial Capitals", 0.02),
    "islamabad":       (212, "Islamabad", "Federal and Provincial Capitals", 0.01),
    "lahore":          (218, "Lahore", "Federal and Provincial Capitals", 0.03),
    "quetta":          (426, "Quetta", "Federal and Provincial Capitals", 0.20),
    "peshawar":        (125, "Peshawar", "Federal and Provincial Capitals", 0.11),
    "kohistan":        (114, "Kohistan", "High-Deprivation", 0.53),
    "dera_bugti":      (404, "Dera Bugti", "High-Deprivation", 0.36),
    "rajanpur":        (230, "Rajanpur", "High-Deprivation", 0.32),
    "tharparkar":      (327, "Tharparkar", "High-Deprivation", 0.51),
}
LABEL = {k: v[1] for k, v in DISTRICTS.items()}

# ---------------------------------------------------------------------------
# Predictors taken directly from one PSLM question: name -> (module, variable)
# ---------------------------------------------------------------------------
SURVEY_ITEMS = {
    "age": ("roster", "age"),
    "gender": ("roster", "sb1q4"),
    "relationship_to_head": ("roster", "sb1q2"),
    "reason_for_headship": ("roster", "sb1q3"),
    "residence_status": ("roster", "sb1q5"),
    "marital_status": ("roster", "sb1q7"),
    "region": ("roster", "region"),
    "born_in_district": ("secb2", "sb2q01"),
    "born_district_type": ("secb2", "sb2q2a"),
    "migration_reason": ("secb2", "sb2q05"),
    "can_write": ("secc1", "sc1q2a"),
    "can_do_math": ("secc1", "sc1q3a"),
    "years_to_complete_primary": ("secc1", "sc1q06"),
    "used_computer": ("secc2", "sc2q01"),
    "computer_location": ("secc2", "sc2q02"),
    "no_computer_reason": ("secc2", "sc2q04"),
    "has_mobile": ("secc2", "sc2q05"),
    "used_mobile": ("secc2", "sc2q06"),
    "no_mobile_reason": ("secc2", "sc2q07"),
    "sick_last_2wks": ("secd", "sdaq01"),
    "worked_last_month": ("sece", "seaq01"),
    "work_days_last_month": ("sece", "seaq02"),
    "has_job": ("sece", "seaq03"),
    "employment_status": ("sece", "seaq06"),
    "can_report_income": ("sece", "seaq07"),
    "monthly_income": ("sece", "seaq08"),
    "received_in_kind_income": ("sece", "seaq18"),
    "income_used_for_hh": ("sece", "seaq22"),
    "occupancy_status": ("secf1", "sf1q01"),
    "house_owner_gender": ("secf1", "sf1q02"),
    "dwelling_type": ("secf1", "sf1q03"),
    "num_rooms": ("secf1", "sf1q04"),
    "heating_fuel": ("secf1", "sf1q09"),
    "has_internet": ("secf1", "sf1q11_1a"),
    "has_mobile_phone": ("secf1", "sf1q11_1b"),
    "has_computer": ("secf1", "sf1q11_1d"),
    "has_laptop": ("secf1", "sf1q11_1e"),
    "sufficient_drinking_water": ("secf2", "sf2q07"),
    "pay_for_water": ("secf2", "sf2q08"),
    "toilet_type": ("secf2", "sf2q11"),
    "shared_toilet": ("secf2", "sf2q15"),
    "connected_to_sewerage": ("secf2", "sf2q16"),
    "cooking_water_source": ("secf2", "sf2q17"),
    "handwashing_water_source": ("secf2", "sf2q18"),
    "has_handwashing_place": ("secf2", "sf2q19"),
    "worried_about_food": ("seck", "c01"),
    "household_ran_out_of_food": ("seck", "c06"),
    "hungry_but_did_not_eat": ("seck", "c07"),
}

# Predictors constructed from several survey items
CONSTRUCTED = {
    "disability": "Any Washington Group functional difficulty (secb2 sb2q06-11) rated "
                  "'some difficulty' or worse",
    "head_age": "Age of the household head (roster); age 99 (not known) treated as missing",
    "head_gender": "Sex of the household head (roster)",
    "head_edu_level_num": "Education of the household head (secc1): 0 never attended, "
                          "1 classes 1-4, 2 class 5, 3 classes 6-12, 4 degree, 5 other",
    "total_assets": "Number of property and livestock items owned (secg items 1, 4-10)",
    "property_owner_gender": "Gender of the owner of agricultural land (secg item 1)",
    "transport_mode": "Mode of transport to the basic health unit (secl)",
    "log_household_income": "log(1 + annual household income); income summed over all "
                            "members and all sources (sece); missing when a member worked "
                            "but no cash amount was recorded",
}

FEATURES = list(SURVEY_ITEMS) + list(CONSTRUCTED)
NUMERIC_FEATURES = ["age", "num_rooms", "head_age", "head_edu_level_num", "total_assets",
                    "log_household_income", "work_days_last_month", "monthly_income"]
CATEGORICAL_FEATURES = [f for f in FEATURES if f not in NUMERIC_FEATURES]

# Recorded only for children who attend (or attended) school: excluded from the models
SCHOOL_DEPENDENT = ["can_do_math", "can_write", "years_to_complete_primary"]

# The child's own work and own computer / mobile use, measured at the same time as
# attendance (possibly a consequence of it): left out in a sensitivity analysis
CONCURRENT = ["worked_last_month", "work_days_last_month", "has_job", "employment_status",
              "can_report_income", "monthly_income", "received_in_kind_income",
              "income_used_for_hh", "used_computer", "computer_location", "no_computer_reason",
              "used_mobile", "no_mobile_reason"]

# ---------------------------------------------------------------------------
# Modelling rules (identical for all districts)
# ---------------------------------------------------------------------------
MAX_MISSING_SHARE = 0.50    # drop a predictor missing for > 50 % of training children
MIN_ABS_CORRELATION = 0.01  # keep indicator columns with |r| >= 0.01 with the outcome
N_FOLDS = 5                 # 80 % train / 20 % test per fold
N_REPEATS = 5               # repeated with different shuffles: 25 test sets

# ---------------------------------------------------------------------------
# Readable names and domains (Table D1)
# ---------------------------------------------------------------------------
FEATURE_DOMAINS = {
    "Child demographics": ["age", "gender", "marital_status", "relationship_to_head",
                           "residence_status", "born_in_district", "born_district_type",
                           "migration_reason"],
    "Education & skills": ["can_do_math", "can_write", "years_to_complete_primary"],
    "Household head": ["head_age", "head_gender", "head_edu_level_num", "reason_for_headship"],
    "Child work & income": ["worked_last_month", "work_days_last_month", "has_job",
                            "employment_status", "can_report_income", "monthly_income",
                            "received_in_kind_income", "income_used_for_hh"],
    "Household economic status": ["log_household_income", "total_assets", "house_owner_gender",
                                  "occupancy_status", "property_owner_gender"],
    "Food security": ["worried_about_food", "household_ran_out_of_food", "hungry_but_did_not_eat"],
    "Water, sanitation & hygiene": ["toilet_type", "shared_toilet", "connected_to_sewerage",
                                    "has_handwashing_place", "handwashing_water_source",
                                    "cooking_water_source", "sufficient_drinking_water",
                                    "pay_for_water"],
    "Housing & location": ["dwelling_type", "num_rooms", "heating_fuel", "region",
                           "transport_mode"],
    "Digital access": ["has_internet", "has_computer", "has_laptop", "has_mobile_phone",
                       "has_mobile", "used_computer", "used_mobile", "no_computer_reason",
                       "no_mobile_reason", "computer_location"],
    "Health & disability": ["disability", "sick_last_2wks"],
}
FEATURE_NAMES = {
    "age": "Age", "gender": "Gender", "marital_status": "Marital status",
    "relationship_to_head": "Relationship to head", "residence_status": "Residence status",
    "born_in_district": "Born in district", "born_district_type": "Born-district type",
    "migration_reason": "Migration reason", "can_do_math": "Can do math",
    "can_write": "Can write", "years_to_complete_primary": "Years to complete primary",
    "head_age": "Head age", "head_gender": "Head gender",
    "head_edu_level_num": "Head education level", "reason_for_headship": "Reason for headship",
    "worked_last_month": "Worked last month", "work_days_last_month": "Work days last month",
    "has_job": "Has job", "employment_status": "Employment status",
    "can_report_income": "Can report income", "monthly_income": "Monthly income",
    "received_in_kind_income": "Received in-kind wages",
    "income_used_for_hh": "Income used for household",
    "log_household_income": "Household income (log)", "total_assets": "Total assets",
    "house_owner_gender": "House owner gender", "occupancy_status": "Occupancy status",
    "property_owner_gender": "Property owner gender", "worried_about_food": "Worried about food",
    "household_ran_out_of_food": "Household ran out of food",
    "hungry_but_did_not_eat": "Hungry but did not eat", "toilet_type": "Toilet type",
    "shared_toilet": "Shared toilet", "connected_to_sewerage": "Connected to sewerage",
    "has_handwashing_place": "Has handwashing place",
    "handwashing_water_source": "Handwashing water source",
    "cooking_water_source": "Cooking water source",
    "sufficient_drinking_water": "Sufficient drinking water", "pay_for_water": "Pay for water",
    "dwelling_type": "Dwelling type", "num_rooms": "Number of rooms",
    "heating_fuel": "Heating fuel", "region": "Region (urban/rural)",
    "transport_mode": "Transport mode", "has_internet": "Has internet",
    "has_computer": "Has computer", "has_laptop": "Has laptop",
    "has_mobile_phone": "Household has mobile phone", "has_mobile": "Child has own mobile",
    "used_computer": "Used computer", "used_mobile": "Used mobile",
    "no_computer_reason": "No-computer reason", "no_mobile_reason": "No-mobile reason",
    "computer_location": "Computer location", "disability": "Disability",
    "sick_last_2wks": "Sick in last 2 weeks",
}
# Level at which each predictor varies: the child, or the household (shared by siblings)
CHILD_LEVEL = ["age", "gender", "marital_status", "relationship_to_head", "residence_status",
               "born_in_district", "born_district_type", "migration_reason", "can_do_math",
               "can_write", "years_to_complete_primary", "disability", "sick_last_2wks",
               "has_mobile", "used_computer", "computer_location", "no_computer_reason",
               "used_mobile", "no_mobile_reason", "worked_last_month", "work_days_last_month",
               "has_job", "employment_status", "can_report_income", "monthly_income",
               "received_in_kind_income", "income_used_for_hh"]
FEATURE_LEVEL = {f: "child" if f in CHILD_LEVEL else "household" for f in FEATURES}

assert sorted(sum(FEATURE_DOMAINS.values(), [])) == sorted(FEATURES)
assert set(CHILD_LEVEL) <= set(FEATURES) and set(CONCURRENT) <= set(FEATURES)
assert set(FEATURE_NAMES) == set(FEATURES)
