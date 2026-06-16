"""Does cross-confederation calibration improve out-of-sample WC prediction?

For each World Cup, estimate confederation offsets from data BEFORE it started
(no leakage), apply them to Elo, and compare accuracy/log-loss to the baseline.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from experiments.cv import DF, fit_fold, predict_match, WCS, ENDS, base
from src import confederations


def eval_with_offsets(use_confed):
    agg = dict(n=0, correct=0, exact=0, ll=0.0)
    per = {}
    cfg = base(blend="ensemble", w=0.70, beta=0.005, d0=0.28, mix=0.5)
    for year in WCS:
        dc, elo = fit_fold(year, cfg["half_life"], cfg["ridge"])
        if use_confed:
            train = DF[DF["date"] < pd.Timestamp(WCS[year])]
            offsets = confederations.estimate_offsets(train)
            elo = confederations.apply_offsets(elo, offsets)
        test = DF[(DF["tournament"] == "FIFA World Cup")
                  & (DF["date"] >= pd.Timestamp(WCS[year]))
                  & (DF["date"] <= pd.Timestamp(ENDS[year]))]
        s = dict(n=0, correct=0, exact=0, ll=0.0)
        for r in test.itertuples(index=False):
            if r.home_team not in dc.idx or r.away_team not in dc.idx:
                continue
            ph, pd_, pa, ih, ij = predict_match(dc, elo, r.home_team, r.away_team,
                                                bool(r.neutral), cfg)
            probs = np.array([ph, pd_, pa])
            a = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
            s["correct"] += int(np.argmax(probs) == a)
            s["exact"] += int(ih == r.home_score and ij == r.away_score)
            s["ll"] += float(-np.log(max(probs[a], 1e-12)))
            s["n"] += 1
        per[year] = (s["correct"] / s["n"], s["ll"] / s["n"])
        for k in agg:
            agg[k] += s[k]
    return agg, per


if __name__ == "__main__":
    for label, flag in [("BASELINE (no confederation adj)", False),
                        ("WITH confederation calibration", True)]:
        agg, per = eval_with_offsets(flag)
        print(f"\n{label}:")
        print(f"  accuracy={agg['correct']/agg['n']*100:.1f}%  "
              f"exact={agg['exact']/agg['n']*100:.1f}%  "
              f"logloss={agg['ll']/agg['n']:.4f}")
        for y, (acc, ll) in sorted(per.items()):
            print(f"    WC{y}: acc={acc*100:.1f}%  logloss={ll:.3f}")

    # show the 2026 offsets actually applied
    print("\n2026 confederation offsets (Elo points):")
    offs = confederations.estimate_offsets(DF)
    for c, v in sorted(offs.items(), key=lambda kv: -kv[1]):
        print(f"  {c:9s}: {v:+.0f}")
