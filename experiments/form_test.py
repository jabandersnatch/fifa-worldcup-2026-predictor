"""Native idea: recent-form adjustment to Dixon-Coles ratings.

The DC fit gives one attack/defence per team over a long window. A team scoring
above/below its baseline lately (form/momentum) is mis-rated. We nudge each
team's attack/defence by its recent goal performance vs its own baseline, over
the last K matches before the tournament, and test if it helps the backtest.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import numpy as np
import pandas as pd

from experiments.score_tune import fold, WIN, DF
from src.predict import Predictor


def recent_goals(train, team, cutoff, K=10):
    sub = train[((train["home_team"] == team) | (train["away_team"] == team))
                & (train["date"] < cutoff)].tail(K)
    if len(sub) < 4:
        return None
    gf = ga = 0
    for r in sub.itertuples(index=False):
        if r.home_team == team:
            gf += r.home_score; ga += r.away_score
        else:
            gf += r.away_score; ga += r.home_score
    return gf / len(sub), ga / len(sub)


def form_adjusted(dc, train, cutoff, fw, K=10, cap=0.4):
    mean_def = np.mean(list(dc.defence.values()))
    new = copy.copy(dc)
    new.attack = dict(dc.attack); new.defence = dict(dc.defence)
    for t in dc.attack:
        rg = recent_goals(train, t, cutoff, K)
        if not rg:
            continue
        avg_gf, avg_ga = rg
        exp_gf = np.exp(dc.intercept + dc.attack[t] - mean_def)
        exp_ga = np.exp(dc.intercept - dc.defence[t])
        atk_form = np.clip(np.log(max(avg_gf, 0.2) / exp_gf), -cap, cap)
        def_form = np.clip(np.log(exp_ga / max(avg_ga, 0.2)), -cap, cap)
        new.attack[t] += fw * atk_form
        new.defence[t] += fw * def_form
    return new


def evaluate(fw, K=10):
    n = c = ex = 0; rps = 0.0
    for y in WIN:
        dc, elo = fold(y)
        train = DF[DF["date"] < pd.Timestamp(WIN[y][0])]
        dcf = form_adjusted(dc, train, pd.Timestamp(WIN[y][0]), fw, K) if fw > 0 else dc
        pred = Predictor(dcf, elo)
        test = DF[(DF["tournament"] == "FIFA World Cup")
                  & (DF["date"] >= pd.Timestamp(WIN[y][0]))
                  & (DF["date"] <= pd.Timestamp(WIN[y][1]))]
        for r in test.itertuples(index=False):
            if r.home_team not in dc.idx or r.away_team not in dc.idx:
                continue
            p = pred.predict(r.home_team, r.away_team, neutral=bool(r.neutral))
            pr = np.array([p["p_home"], p["p_draw"], p["p_away"]])
            a = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
            c += int(np.argmax(pr) == a)
            ex += int(p["ml_home"] == r.home_score and p["ml_away"] == r.away_score)
            cp = np.cumsum(pr); ca = np.cumsum(np.eye(3)[a]); rps += 0.5 * float(np.sum((cp - ca) ** 2))
            n += 1
    print(f"  form_weight={fw:.2f} K={K}: acc={c/n*100:4.1f}% exact={ex/n*100:4.1f}% RPS={rps/n:.4f}")


if __name__ == "__main__":
    print("=== recent-form adjustment sweep (fw=0 is current model) ===")
    for fw in (0.0, 0.10, 0.20, 0.30, 0.45):
        evaluate(fw)
    print("\n=== window size K (fw=0.2) ===")
    for K in (6, 10, 15):
        evaluate(0.20, K)
