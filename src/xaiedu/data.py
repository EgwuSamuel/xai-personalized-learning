"""Download and load the three datasets with leakage-aware feature sets.

Every loader returns a ``Dataset``: the feature frame, the binary target
(1 = dropout / at-risk), and the lists of categorical and numeric columns.
Raw files are never committed; ``download_all()`` fetches them from the
original public sources and checks their SHA-256 where it is fixed.
"""
from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DATA_PROCESSED, DATA_RAW

KAGGLE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/adilshamim8/"
    "personalized-learning-and-adaptive-education-dataset?datasetVersionNumber=1"
)
KAGGLE_SHA256 = "931a36cb0a65e406e6c6f7841496517b3684f1af66c577449811d5f7b7230cdc"
UCI_URL = (
    "https://archive.ics.uci.edu/static/public/697/"
    "predict+students+dropout+and+academic+success.zip"
)
UCI_SHA256 = "3ef126de5cefff26eb11fbb4237f1a1401cb64b488e2f1d598c23cedeb4c45ae"
OULAD_URL = (
    "https://archive.ics.uci.edu/static/public/349/"
    "open+university+learning+analytics+dataset.zip"
)


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: pd.Series
    categorical: list[str]
    numeric: list[str]
    groups: pd.Series | None = None
    meta: dict = field(default_factory=dict)


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_zip(url: str, dest_dir, members: list[str] | None = None) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=600) as r:
        payload = r.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        z.extractall(dest_dir, members=members)


def download_all() -> None:
    kaggle_csv = DATA_RAW / "kaggle" / "personalized_learning_dataset.csv"
    if not kaggle_csv.exists():
        _fetch_zip(KAGGLE_URL, kaggle_csv.parent)
    uci_csv = DATA_RAW / "uci" / "data.csv"
    if not uci_csv.exists():
        _fetch_zip(UCI_URL, uci_csv.parent)
    oulad_dir = DATA_RAW / "oulad"
    if not (oulad_dir / "studentInfo.csv").exists():
        _fetch_zip(OULAD_URL, oulad_dir)
    for path, expected in ((kaggle_csv, KAGGLE_SHA256), (uci_csv, UCI_SHA256)):
        got = _sha256(path)
        if got != expected:
            raise RuntimeError(f"Checksum mismatch for {path}: {got}")


# --------------------------------------------------------------------------
# Kaggle: "Personalized Learning and Adaptive Education Dataset", version 1
# (the file analysed in the original dissertation; later versions of the same
# Kaggle page contain an unrelated dataset).
# --------------------------------------------------------------------------
KAGGLE_CATEGORICAL = ["Gender", "Education_Level", "Course_Name",
                      "Engagement_Level", "Learning_Style"]
KAGGLE_NUMERIC = ["Age", "Time_Spent_on_Videos", "Quiz_Attempts", "Quiz_Scores",
                  "Forum_Participation", "Assignment_Completion_Rate",
                  "Final_Exam_Score", "Feedback_Score"]


def load_kaggle() -> Dataset:
    df = pd.read_csv(DATA_RAW / "kaggle" / "personalized_learning_dataset.csv")
    y = (df["Dropout_Likelihood"] == "Yes").astype(int).rename("dropout")
    X = df[KAGGLE_NUMERIC + KAGGLE_CATEGORICAL].copy()
    return Dataset("kaggle", X, y, KAGGLE_CATEGORICAL, KAGGLE_NUMERIC)


# --------------------------------------------------------------------------
# UCI #697: Predict Students' Dropout and Academic Success (Realinho et al.)
# --------------------------------------------------------------------------
UCI_CATEGORICAL = ["Marital status", "Application mode", "Course",
                   "Previous qualification", "Nacionality",
                   "Mother's qualification", "Father's qualification",
                   "Mother's occupation", "Father's occupation"]
UCI_BINARY = ["Daytime/evening attendance", "Displaced",
              "Educational special needs", "Debtor", "Tuition fees up to date",
              "Gender", "Scholarship holder", "International"]
UCI_ENROLMENT_NUMERIC = ["Application order", "Previous qualification (grade)",
                         "Admission grade", "Age at enrollment",
                         "Unemployment rate", "Inflation rate", "GDP"]
_UNITS = ["credited", "enrolled", "evaluations", "approved", "grade",
          "without evaluations"]
UCI_SEM1 = [f"Curricular units 1st sem ({u})" for u in _UNITS]
UCI_SEM2 = [f"Curricular units 2nd sem ({u})" for u in _UNITS]

# Information available at each decision point (no feature from the future).
UCI_SCENARIOS = {
    "enrolment": [],
    "sem1": UCI_SEM1,
    "sem2": UCI_SEM1 + UCI_SEM2,
}


def load_uci(scenario: str = "sem1", target: str = "binary") -> Dataset:
    df = pd.read_csv(DATA_RAW / "uci" / "data.csv", sep=";")
    df.columns = [c.strip() for c in df.columns]
    if target == "binary":
        y = (df["Target"] == "Dropout").astype(int).rename("dropout")
    else:
        y = df["Target"].map({"Dropout": 0, "Enrolled": 1, "Graduate": 2}).rename("target3")
    numeric = UCI_ENROLMENT_NUMERIC + UCI_BINARY + UCI_SCENARIOS[scenario]
    X = df[numeric + UCI_CATEGORICAL].copy()
    for c in UCI_CATEGORICAL:
        X[c] = X[c].astype(str)
    return Dataset(f"uci_{scenario}", X, y, UCI_CATEGORICAL, numeric,
                   meta={"scenario": scenario})


# --------------------------------------------------------------------------
# OULAD: Open University Learning Analytics Dataset (Kuzilek et al., 2017)
# Features are built from data available up to ``cutoff_day`` only.
# --------------------------------------------------------------------------
OULAD_CATEGORICAL = ["code_module", "gender", "region", "highest_education",
                     "imd_band", "age_band", "disability"]
OULAD_STATIC_NUMERIC = ["num_of_prev_attempts", "studied_credits",
                        "date_registration"]


def _oulad_vle_daily() -> pd.DataFrame:
    """Aggregate the 10M-row click log once to (student, presentation, day, type)."""
    pkl = DATA_PROCESSED / "oulad_vle_daily.pkl"
    if pkl.exists():
        return pd.read_pickle(pkl)
    raw = DATA_RAW / "oulad"
    vle = pd.read_csv(raw / "vle.csv", usecols=["id_site", "activity_type"], na_values="?")
    sv = pd.read_csv(raw / "studentVle.csv",
                     dtype={"code_module": "category", "code_presentation": "category",
                            "id_student": "int32", "id_site": "int32",
                            "date": "int16", "sum_click": "int32"})
    sv = sv.merge(vle, on="id_site", how="left")
    daily = (sv.groupby(["code_module", "code_presentation", "id_student", "date",
                         "activity_type"], observed=True)["sum_click"].sum().reset_index())
    daily.to_pickle(pkl)
    return daily


def load_oulad(cutoff_day: int = 56, top_activity_types: int = 10) -> Dataset:
    # OULAD marks missing values with "?"
    raw = DATA_RAW / "oulad"
    info = pd.read_csv(raw / "studentInfo.csv", na_values="?")
    reg = pd.read_csv(raw / "studentRegistration.csv", na_values="?")
    keys = ["code_module", "code_presentation", "id_student"]
    df = info.merge(reg, on=keys, how="left")

    # Exclude students who had already withdrawn by the cutoff: their outcome
    # is known, so predicting it would be trivial and useless for intervention.
    already_out = df["date_unregistration"].notna() & (df["date_unregistration"] <= cutoff_day)
    df = df.loc[~already_out].copy()
    df["imd_band"] = df["imd_band"].fillna("missing").astype(str).str.replace("%", "", regex=False)
    df["date_registration"] = df["date_registration"].fillna(df["date_registration"].median())

    # --- VLE engagement up to the cutoff ---
    daily = _oulad_vle_daily()
    d = daily[daily["date"] <= cutoff_day]
    tot = d.groupby(keys, observed=True)["sum_click"].sum().rename("clicks_total")
    days = d.groupby(keys, observed=True)["date"].nunique().rename("active_days")
    last = d.groupby(keys, observed=True)["date"].max().rename("last_active_day")
    recent = (d[d["date"] > cutoff_day - 14].groupby(keys, observed=True)["sum_click"]
              .sum().rename("clicks_last_14d"))
    top_types = (daily.groupby("activity_type", observed=True)["sum_click"].sum()
                 .sort_values(ascending=False).index[:top_activity_types])
    by_type = (d[d["activity_type"].isin(top_types)]
               .pivot_table(index=keys, columns="activity_type", values="sum_click",
                            aggfunc="sum", fill_value=0, observed=True))
    by_type.columns = [f"clicks_{c}" for c in by_type.columns]
    eng = pd.concat([tot, days, last, recent, by_type], axis=1).reset_index()
    df = df.merge(eng, on=keys, how="left")
    click_cols = [c for c in df.columns if c.startswith("clicks_")] + ["active_days"]
    df[click_cols] = df[click_cols].fillna(0)
    df["days_since_last_active"] = (cutoff_day - df["last_active_day"]).fillna(cutoff_day + 30)
    df = df.drop(columns="last_active_day")

    # --- Assessments due / submitted up to the cutoff (exams excluded) ---
    ass = pd.read_csv(raw / "assessments.csv", na_values="?")
    sa = pd.read_csv(raw / "studentAssessment.csv", na_values="?")
    ass = ass[(ass["assessment_type"] != "Exam") & ass["date"].notna() & (ass["date"] <= cutoff_day)]
    sa = sa.merge(ass[["id_assessment", "code_module", "code_presentation", "date", "weight"]],
                  on="id_assessment", how="inner")
    sa = sa[sa["date_submitted"] <= cutoff_day]
    sa["late"] = (sa["date_submitted"] > sa["date"]).astype(int)
    sub = sa.groupby(keys).agg(n_submitted=("id_assessment", "nunique"),
                               mean_score=("score", "mean"),
                               n_late=("late", "sum")).reset_index()
    due = ass.groupby(["code_module", "code_presentation"])["id_assessment"].nunique().rename("n_due").reset_index()
    df = df.merge(sub, on=keys, how="left").merge(due, on=["code_module", "code_presentation"], how="left")
    df[["n_submitted", "n_late", "n_due"]] = df[["n_submitted", "n_late", "n_due"]].fillna(0)
    df["n_missed"] = (df["n_due"] - df["n_submitted"]).clip(lower=0)
    df["mean_score"] = df["mean_score"].fillna(0)  # no submission -> 0, flagged by n_submitted

    engagement = ["clicks_total", "active_days", "clicks_last_14d", "days_since_last_active",
                  *by_type.columns, "n_submitted", "n_missed", "n_late", "mean_score"]
    numeric = OULAD_STATIC_NUMERIC + engagement
    y = df["final_result"].isin(["Withdrawn", "Fail"]).astype(int).rename("at_risk")
    X = df[numeric + OULAD_CATEGORICAL].reset_index(drop=True)
    for c in OULAD_CATEGORICAL:
        X[c] = X[c].astype(str)
    return Dataset(f"oulad_d{cutoff_day}", X, y.reset_index(drop=True),
                   OULAD_CATEGORICAL, numeric,
                   groups=df["code_presentation"].reset_index(drop=True),
                   meta={"cutoff_day": cutoff_day})


def sample_ids(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=min(k, n), replace=False)
