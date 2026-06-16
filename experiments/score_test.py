"""Scoreline-focused evaluation: diagnose goals calibration and test tweaks that
might improve exact-score / outcome on the 2014/2018/2022 World Cups.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src import data_loader, confederations
from src.dixon_coles import DixonColes
from src.elo import compute_elo
from src.predict import Predictor

DF = data_loader.load_matches(min_date="2002-01-01")
WINDOWS = {2014: ("2014-06-12", "2014-07-15"),
           2018: ("2018-06-14", "2018-07-17"),
           2022: ("2022-11-20", "2022-12-20")}
_cache = {}


def fold(year):
    if year not in _cache:
        train = DF[DF["date"] < pd.Timestamp(WINDOWS[year][0])]
        dc = DixonColes().fit(train)
        elo = compute_elo(train)
        elo = confederations.apply_offsets(elo, confederations.estimate_offsets(train))
        _cache[year] = (dc, Predictor(dc, elo))
    return _cache[year]


def matrix(lam, mu, rho, max_goals=10):
    gh = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(gh, lam), poisson.pmf(gh, mu))
    m[0, 0] *= 1 - lam * mu * rho; m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + mu * rho; m[1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum()


def evaluate(goal_scale=1.0, label=""):
    n = correct = exact = 0
    rps = 0.0
    pred_tot = act_tot = 0.0
    for year in WINDOWS:
        dc, pred = fold(year)
        test = DF[(DF["tournament"] == "FIFA World Cup")
                  & (DF["date"] >= pd.Timestamp(WINDOWS[year][0]))
                  & (DF["date"] <= pd.Timestamp(WINDOWS[year][1]))]
        for r in test.itertuples(index=False):
            if r.home_team not in dc.idx or r.away_team not in dc.idx:
                continue
            lam, mu = pred.blended_lambdas(r.home_team, r.away_team, bool(r.neutral))
            lam *= goal_scale; mu *= goal_scale
            m = matrix(lam, mu, dc.rho)
            ph = np.tril(m, -1).sum(); pdr = np.trace(m); pa = np.triu(m, 1).sum()
            a = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
            probs = np.array([ph, pdr, pa])
            correct += int(np.argmax(probs) == a)
            i, j = np.unravel_index(np.argmax(m), m.shape)
            exact += int(i == r.home_score and j == r.away_score)
            # ranked probability score (ordered: H, D, A)
            cp = np.cumsum(probs); ca = np.cumsum(np.eye(3)[a])
            rps += 0.5 * float(np.sum((cp - ca) ** 2))
            pred_tot += lam + mu; act_tot += r.home_score + r.away_score
            n += 1
    print(f"{label:28s} acc={correct/n*100:4.1f}%  exact={exact/n*100:4.1f}%  "
          f"RPS={rps/n:.4f}  pred_goals/match={pred_tot/n:.2f} (actual {act_tot/n:.2f})")
    return correct / n, exact / n, rps / n


if __name__ == "__main__":
    print("=== goals calibration diagnostic + scale sweep ===")
    for s in (0.90, 0.95, 1.00, 1.05, 1.10):
        evaluate(goal_scale=s, label=f"goal_scale={s:.2f}")
