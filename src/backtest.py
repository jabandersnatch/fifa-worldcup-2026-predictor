"""Validate the model out-of-sample on past World Cups: for each tournament,
train ONLY on data before it started, then measure outcome accuracy, exact-score
accuracy, multiclass Brier, and log-loss. Averaging over several tournaments
gives a robust, honest accuracy figure rather than a single noisy one.
"""
import numpy as np
import pandas as pd

from . import confederations
from .dixon_coles import DixonColes
from .elo import compute_elo
from .predict import Predictor

# tournament -> (start, end) windows
WINDOWS = {
    2014: ("2014-06-12", "2014-07-15"),
    2018: ("2018-06-14", "2018-07-17"),
    2022: ("2022-11-20", "2022-12-20"),
}


def _eval_window(df, start, end, elo_weight):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    train = df[df["date"] < start]
    test = df[(df["tournament"] == "FIFA World Cup")
              & (df["date"] >= start) & (df["date"] <= end)]
    dc = DixonColes().fit(train)
    elo = compute_elo(train)
    elo = confederations.apply_offsets(elo, confederations.estimate_offsets(train))
    pred = Predictor(dc, elo, elo_weight=elo_weight)

    s = dict(n=0, correct=0, exact=0, brier=0.0, logloss=0.0)
    for r in test.itertuples(index=False):
        if r.home_team not in dc.idx or r.away_team not in dc.idx:
            continue
        p = pred.predict(r.home_team, r.away_team, neutral=bool(r.neutral), city=r.city)
        probs = np.array([p["p_home"], p["p_draw"], p["p_away"]])
        a = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
        s["correct"] += int(np.argmax(probs) == a)
        s["exact"] += int(p["ml_home"] == r.home_score and p["ml_away"] == r.away_score)
        oh = np.zeros(3); oh[a] = 1
        s["brier"] += float(np.sum((probs - oh) ** 2))
        s["logloss"] += float(-np.log(max(probs[a], 1e-12)))
        s["n"] += 1
    return s


def run_cv(df, elo_weight=0.70, windows=WINDOWS):
    per = {}
    agg = dict(n=0, correct=0, exact=0, brier=0.0, logloss=0.0)
    for year, (start, end) in windows.items():
        s = _eval_window(df, start, end, elo_weight)
        per[year] = {
            "n": s["n"],
            "outcome_accuracy": s["correct"] / s["n"],
            "exact_score_accuracy": s["exact"] / s["n"],
            "logloss": s["logloss"] / s["n"],
        }
        for k in agg:
            agg[k] += s[k]
    return {
        "per": per,
        "n": agg["n"],
        "outcome_accuracy": agg["correct"] / agg["n"],
        "exact_score_accuracy": agg["exact"] / agg["n"],
        "brier": agg["brier"] / agg["n"],
        "logloss": agg["logloss"] / agg["n"],
    }
