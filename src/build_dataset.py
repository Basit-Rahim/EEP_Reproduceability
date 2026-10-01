"""
Build the child-level analysis dataset from the PSLM 2019-20 Stata files.

One row per child aged 5-16 who is a usual member of a sampled household in the
nine study districts. Categorical predictors are stored as text categories
(PSLM value labels), numeric predictors as numbers.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
import pyreadstat

from .config import (AGE_MAX, AGE_MIN, CATEGORICAL_FEATURES, DISTRICTS, FEATURES,
                     NUMERIC_FEATURES, RAW_DIR, SURVEY_ITEMS, TARGET)

PERSON_MODULES = {"secb2", "secc1", "secc2", "secd", "sece"}

FOOD_ANSWERS = {1: "yes", 2: "no", 98: "don't know", 99: "refused"}
OWNER_GENDER = {1: "male", 2: "female", 3: "jointly", 4: "don't know"}
TRANSPORT = {0: "not reported", 1: "on foot", 2: "mechanical", 3: "non-mechanical"}
SEX = {1: "male", 2: "female"}
CUSTOM_LABELS = {
    "worried_about_food": FOOD_ANSWERS,
    "household_ran_out_of_food": FOOD_ANSWERS,
    "hungry_but_did_not_eat": FOOD_ANSWERS,
}


@lru_cache(maxsize=None)
def read_module(name):
    """Numeric codes of one PSLM module and its value labels {variable: {code: label}}."""
    df, meta = pyreadstat.read_dta(str(RAW_DIR / f"{name}.dta"), encoding="latin1")
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    labels = {var: meta.value_labels[lab] for var, lab in meta.variable_to_label.items()}
    return df, labels


def to_category(codes, labels=None):
    """Codes -> text categories: the value label where there is one, otherwise the code."""
    labels = labels or {}

    def one(v):
        if pd.isna(v):
            return np.nan
        if v in labels:
            return str(labels[v]).strip()
        return str(int(v)) if float(v).is_integer() else str(v)
    return pd.Series(codes).map(one).astype(object)


def child_sample(code):
    """Usual household members aged 5-16 in one district, with sampling weight and PSU."""
    roster, _ = read_module("roster")
    plist, _ = read_module("plist")
    kids = roster[(roster["district"] == code)
                  & roster["age"].between(AGE_MIN, AGE_MAX)
                  & (roster["sb1q11"] == 1)]
    kids = kids.merge(plist[["hhcode", "idc", "weights"]], on=["hhcode", "idc"], how="left")
    return kids.reset_index(drop=True)


def attach(kids, module):
    """Rows of a PSLM module matched to the children: by person or by household."""
    df, _ = read_module(module)
    keys = ["hhcode", "idc"] if module in PERSON_MODULES else ["hhcode"]
    df = df.drop(columns=[c for c in ("psu", "province", "region", "district") if c in df])
    return kids[["hhcode", "idc"]].merge(df.drop_duplicates(keys), on=keys, how="left")


def household_heads():
    roster, _ = read_module("roster")
    secc1, _ = read_module("secc1")
    heads = roster[roster["sb1q2"] == 1].drop_duplicates("hhcode")   # first listed head
    return heads[["hhcode", "idc", "age", "sb1q4"]].merge(
        secc1[["hhcode", "idc", "sc1q01", "sc1q05"]], on=["hhcode", "idc"], how="left")


def head_education_level(attendance, grade):
    """0 never attended; 1 classes 1-4; 2 class 5; 3 classes 6-12; 4 degree; 5 other.
    Heads still studying (or without an education record) are missing."""
    level = np.select(
        [grade.between(1, 4), grade == 5, grade.between(6, 12), grade.between(13, 24),
         grade == 28], [1, 2, 3, 4, 5], default=np.nan)
    level = np.where(attendance == 2, level, np.nan)
    return np.where(attendance == 1, 0, level).astype(float)


def household_income():
    """Annual household income: every member, every income source.
    Main job (monthly pay x months worked, or annual pay), other work, in-kind wages
    sold, pensions, domestic transfers, foreign remittances, rent and other income.
    Zero only when nobody in the household worked; missing when someone worked but no
    cash amount was recorded (paid in kind, or a contributing family worker)."""
    s, _ = read_module("sece")
    main_job = np.select(
        [s["seaq07"] == 1, s["seaq07"] == 2],
        [s["seaq08"].fillna(0) * s["seaq09"].fillna(12), s["seaq10"].fillna(0)],
        default=np.where(s["seaq10"].notna(), s["seaq10"],
                         np.where(s["seaq08"].notna(), s["seaq08"] * s["seaq09"].fillna(12), 0)))
    other_sources = ["seaq15", "seaq17", "seaq19", "seaq21", "seaq23", "seaq24", "seaq25", "seaq26"]
    person = pd.DataFrame({
        "hhcode": s["hhcode"],
        "income": main_job + s[other_sources].fillna(0).sum(axis=1),
        "worked": s["seaq01"] == 1,
        "answered": s[["seaq23", "seaq24", "seaq25", "seaq26"]].notna().any(axis=1),
    })
    hh = person.groupby("hhcode").agg(total=("income", "sum"), worked=("worked", "any"),
                                      answered=("answered", "any"))
    status = np.select([hh["total"] > 0, hh["worked"], ~hh["answered"]],
                       ["reported", "unmeasured", "not answered"], default="zero")
    hh["status"] = status
    hh["income"] = hh["total"].where(np.isin(status, ["reported", "zero"]))
    return hh


def build_district(key):
    """The analysis dataset for one district."""
    code = DISTRICTS[key][0]
    kids = child_sample(code)
    out = kids[["hhcode", "idc", "psu", "weights"]].copy()
    out.insert(0, "district_key", key)

    modules = {m for m, _ in SURVEY_ITEMS.values()} - {"roster"}
    matched = {m: attach(kids, m) for m in modules | {"secb2", "secc1", "sece"}}
    for name, (module, var) in SURVEY_ITEMS.items():
        values = (kids if module == "roster" else matched[module])[var].values
        if name in NUMERIC_FEATURES:
            out[name] = pd.to_numeric(values, errors="coerce")
        else:
            labels = CUSTOM_LABELS.get(name) or read_module(module)[1].get(var)
            out[name] = to_category(values, labels).values

    difficulty = matched["secb2"][[f"sb2q{i:02d}" for i in range(6, 12)]]
    out["disability"] = np.where((difficulty >= 2).any(axis=1), "yes", "no")

    heads = kids[["hhcode"]].merge(household_heads(), on="hhcode", how="left")
    out["head_age"] = heads["age"].where(heads["age"] < 99).values   # 99 = not known
    out["head_gender"] = to_category(heads["sb1q4"].values, SEX).values
    out["head_edu_level_num"] = head_education_level(heads["sc1q01"], heads["sc1q05"])

    secg, _ = read_module("secg")
    owned = secg[secg["itc"].isin([1, 4, 5, 6, 7, 8, 9, 10]) & (secg["sgaq01"] == 1)]
    out["total_assets"] = kids["hhcode"].map(owned.groupby("hhcode").size()).fillna(0).values
    land = secg[secg["itc"] == 1].drop_duplicates("hhcode").set_index("hhcode")["sgaq02"]
    out["property_owner_gender"] = to_category(kids["hhcode"].map(land).values,
                                               OWNER_GENDER).values

    secl, _ = read_module("secl")
    bhu = secl[secl["facilities"] == 1].drop_duplicates("hhcode").set_index("hhcode")["slq6"]
    out["transport_mode"] = to_category(kids["hhcode"].map(bhu).values, TRANSPORT).values

    income = household_income()
    out["household_income"] = kids["hhcode"].map(income["income"]).values
    out["log_household_income"] = np.log1p(out["household_income"])
    out["income_status"] = kids["hhcode"].map(income["status"]).values

    out[TARGET] = (matched["secc1"]["sc1q01"].values == 3).astype(int)
    return out[["district_key", "hhcode", "idc", "psu", "weights", TARGET] + FEATURES
               + ["household_income", "income_status"]]


def build_dataset():
    data = pd.concat([build_district(k) for k in DISTRICTS], ignore_index=True)
    for col in CATEGORICAL_FEATURES:
        data[col] = data[col].astype(object)
    for col in NUMERIC_FEATURES:
        data[col] = data[col].astype(float)
    return data
