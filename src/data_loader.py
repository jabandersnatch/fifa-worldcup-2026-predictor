"""Load and clean international match data (martj42 dataset).

The dataset already contains the 2026 World Cup fixtures: played matches carry
scores, upcoming ones have NA. So "refresh" simply re-downloads this file.
"""
import os
import pandas as pd
import requests

from . import config

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RESULTS_CSV = os.path.join(DATA_DIR, "international_results.csv")
GOALSCORERS_CSV = os.path.join(DATA_DIR, "goalscorers.csv")
GOALSCORERS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv"


def _download_one(url, path):
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)


def download(force=True):
    """Download the latest results + goalscorers CSVs. Used by `main.py refresh`."""
    os.makedirs(DATA_DIR, exist_ok=True)
    _download_one(config.RESULTS_URL, RESULTS_CSV)
    _download_one(GOALSCORERS_URL, GOALSCORERS_CSV)
    return RESULTS_CSV


def load_goalscorers(min_date="2023-01-01"):
    """Non-own-goal scoring events on/after min_date, as a clean DataFrame."""
    if not os.path.exists(GOALSCORERS_CSV):
        _download_one(GOALSCORERS_URL, GOALSCORERS_CSV)
    df = pd.read_csv(GOALSCORERS_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.Timestamp(min_date))
            & (df["own_goal"].astype(str).str.upper() != "TRUE")].copy()
    df = df.dropna(subset=["scorer"])
    return df


def load_matches(min_date="2010-01-01"):
    """All played internationals on/after min_date, as a clean DataFrame."""
    if not os.path.exists(RESULTS_CSV):
        download()
    df = pd.read_csv(RESULTS_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= pd.Timestamp(min_date)].copy()
    played = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)
    # neutral column is "TRUE"/"FALSE" strings or bools
    played["neutral"] = played["neutral"].astype(str).str.upper().eq("TRUE")
    return played.sort_values("date").reset_index(drop=True)


def wc2026_matches():
    """All 2026 World Cup group-stage matches (played + upcoming)."""
    df = pd.read_csv(RESULTS_CSV)
    df["date"] = pd.to_datetime(df["date"])
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= "2026-01-01")].copy()
    wc["neutral"] = wc["neutral"].astype(str).str.upper().eq("TRUE")
    wc["played"] = wc["home_score"].notna() & wc["away_score"].notna()
    return wc.sort_values("date").reset_index(drop=True)
