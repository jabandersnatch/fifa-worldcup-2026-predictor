"""Tune the model for SCORELINE quality (not just who-wins), on the 2014/2018/
2022 World Cups. Reports outcome accuracy, exact-score, RPS (proper ordered
metric) and draw calibration (do we predict draws at the right rate?).

Uses the full production pipeline: Dixon-Coles + confederation-calibrated Elo +
the ensemble Predictor. Expert/squad priors are 2026-only, so excluded here.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src import data_loader, confederations
from src.dixon_coles import DixonColes
from src.elo import compute_elo
from src.predict import Predictor

DF = data_loader.load_matches(min_date="2002-01-01")
WIN = {2014: ("2014-06-12", "2014-07-15"),
       2018: ("2018-06-14", "2018-07-17"),
       2022: ("2022-11-20", "2022-12-20")}
_fold = {}


def fold(year):
    if year not in _fold:
        train = DF[DF["date"] < pd.Timestamp(WIN[year][0])]
        dc = DixonColes().fit(train)
        elo = compute_elo(train)
        elo = confederations.apply_offsets(elo, confederations.estimate_offsets(train))
        _fold[year] = (dc, elo)
    return _fold[year]


def evaluate(label, **params):
    n = correct = exact = 0
    rps = pred_draw = act_draw = 0.0
    for year in WIN:
        dc, elo = fold(year)
        pred = Predictor(dc, elo, **params)
        test = DF[(DF["tournament"] == "FIFA World Cup")
                  & (DF["date"] >= pd.Timestamp(WIN[year][0]))
                  & (DF["date"] <= pd.Timestamp(WIN[year][1]))]
        for r in test.itertuples(index=False):
            if r.home_team not in dc.idx or r.away_team not in dc.idx:
                continue
            p = pred.predict(r.home_team, r.away_team, neutral=bool(r.neutral))
            probs = np.array([p["p_home"], p["p_draw"], p["p_away"]])
            a = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
            correct += int(np.argmax(probs) == a)
            exact += int(p["ml_home"] == r.home_score and p["ml_away"] == r.away_score)
            cp = np.cumsum(probs); ca = np.cumsum(np.eye(3)[a])
            rps += 0.5 * float(np.sum((cp - ca) ** 2))
            pred_draw += p["p_draw"]; act_draw += int(a == 1)
            n += 1
    print(f"{label:34s} acc={correct/n*100:4.1f}% exact={exact/n*100:4.1f}% "
          f"RPS={rps/n:.4f}  draw: pred={pred_draw/n*100:.0f}% actual={act_draw/n*100:.0f}%")


if __name__ == "__main__":
    print("=== current default ===")
    evaluate("beta=.005 d0=.28 mix=.5")
    print("\n=== sharpen supremacy (beta) — fewer soft draws ===")
    for beta in (0.006, 0.008, 0.010):
        evaluate(f"beta={beta:.3f}", beta=beta)
    print("\n=== draw_base sweep ===")
    for d0 in (0.22, 0.25, 0.28):
        evaluate(f"d0={d0:.2f}", draw_base=d0)
    print("\n=== ensemble mix sweep ===")
    for mix in (0.35, 0.5, 0.65):
        evaluate(f"mix={mix:.2f}", ensemble_mix=mix)
