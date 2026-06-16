"""Cross-validation harness: evaluate a predictor config across the 2010, 2014,
2018 and 2022 World Cups (256 matches). Trains only on pre-tournament data each
time. Used to measure model improvements honestly before baking them in.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src import data_loader
from src.dixon_coles import DixonColes
from src.elo import compute_elo, outcome_probs as elo_probs

WCS = {2010: "2010-06-11", 2014: "2014-06-12", 2018: "2018-06-14", 2022: "2022-11-20"}
ENDS = {2010: "2010-07-13", 2014: "2014-07-15", 2018: "2018-07-17", 2022: "2022-12-20"}

DF = data_loader.load_matches(min_date="2002-01-01")
_cache = {}


def fit_fold(year, half_life, ridge):
    key = (year, half_life, ridge)
    if key not in _cache:
        train = DF[DF["date"] < pd.Timestamp(WCS[year])]
        dc = DixonColes(half_life_days=half_life, ridge=ridge).fit(train)
        elo = compute_elo(train)
        _cache[key] = (dc, elo)
    return _cache[key]


def _score_matrix(lam, mu, rho, max_goals=10):
    gh = np.arange(max_goals + 1)
    mat = np.outer(poisson.pmf(gh, lam), poisson.pmf(gh, mu))
    mat[0, 0] *= 1 - lam * mu * rho
    mat[0, 1] *= 1 + lam * rho
    mat[1, 0] *= 1 + mu * rho
    mat[1, 1] *= 1 - rho
    mat = np.clip(mat, 0, None)
    return mat / mat.sum()


def _lambda_blend(dc, elo, home, away, neutral, w, beta):
    lam, mu = dc.expected_goals(home, away, neutral)
    if home in elo and away in elo:
        adv = 0.0 if neutral else 65.0
        dr = (elo[home] + adv) - elo[away]
        sup_elo = beta * dr
        total = lam + mu
        sup = (1 - w) * (lam - mu) + w * sup_elo
        lam = max((total + sup) / 2, 0.05)
        mu = max((total - sup) / 2, 0.05)
    return lam, mu


def _probs_from_matrix(mat):
    return np.tril(mat, -1).sum(), np.trace(mat), np.triu(mat, 1).sum()


def predict_match(dc, elo, home, away, neutral, cfg):
    w = cfg["w"]
    # scoreline comes from the lambda-blended Poisson (better point predictions)
    lam, mu = _lambda_blend(dc, elo, home, away, neutral, w, cfg.get("beta", 0.005))
    mat = _score_matrix(lam, mu, dc.rho)
    ph_l, pd_l, pa_l = _probs_from_matrix(mat)

    if cfg["blend"] == "lambda":
        ph, pd_, pa = ph_l, pd_l, pa_l
    elif cfg["blend"] == "prob":
        mat0 = _score_matrix(*dc.expected_goals(home, away, neutral), dc.rho)
        ph, pd_, pa = _probs_from_matrix(mat0)
        if w > 0 and home in elo and away in elo:
            eh, ed, ea = elo_probs(elo[home], elo[away], neutral, draw_base=cfg.get("d0", 0.32))
            ph, pd_, pa = (1 - w) * ph + w * eh, (1 - w) * pd_ + w * ed, (1 - w) * pa + w * ea
    else:  # ensemble: average lambda-blend probs with prob-blend probs
        mat0 = _score_matrix(*dc.expected_goals(home, away, neutral), dc.rho)
        ph_p, pd_p, pa_p = _probs_from_matrix(mat0)
        if home in elo and away in elo:
            eh, ed, ea = elo_probs(elo[home], elo[away], neutral, draw_base=cfg.get("d0", 0.32))
            ph_p, pd_p, pa_p = (1 - w) * ph_p + w * eh, (1 - w) * pd_p + w * ed, (1 - w) * pa_p + w * ea
        a = cfg.get("mix", 0.5)
        ph, pd_, pa = a * ph_l + (1 - a) * ph_p, a * pd_l + (1 - a) * pd_p, a * pa_l + (1 - a) * pa_p

    # temperature scaling for calibration
    t = cfg.get("temp", 1.0)
    p = np.array([ph, pd_, pa], dtype=float)
    p = np.clip(p, 1e-9, None)
    if t != 1.0:
        p = p ** (1.0 / t)
    p = p / p.sum()
    i, j = np.unravel_index(np.argmax(mat), mat.shape)
    return p[0], p[1], p[2], int(i), int(j)


def evaluate(cfg, verbose=True):
    agg = dict(n=0, correct=0, exact=0, brier=0.0, ll=0.0)
    per = {}
    for year in WCS:
        dc, elo = fit_fold(year, cfg["half_life"], cfg["ridge"])
        test = DF[(DF["tournament"] == "FIFA World Cup")
                  & (DF["date"] >= pd.Timestamp(WCS[year]))
                  & (DF["date"] <= pd.Timestamp(ENDS[year]))]
        s = dict(n=0, correct=0, exact=0, brier=0.0, ll=0.0)
        for r in test.itertuples(index=False):
            if r.home_team not in dc.idx or r.away_team not in dc.idx:
                continue
            ph, pd_, pa, ih, ij = predict_match(dc, elo, r.home_team, r.away_team,
                                                bool(r.neutral), cfg)
            probs = np.array([ph, pd_, pa])
            a = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
            s["correct"] += int(np.argmax(probs) == a)
            s["exact"] += int(ih == r.home_score and ij == r.away_score)
            oh = np.zeros(3); oh[a] = 1
            s["brier"] += float(np.sum((probs - oh) ** 2))
            s["ll"] += float(-np.log(max(probs[a], 1e-12)))
            s["n"] += 1
        per[year] = (s["correct"] / s["n"], s["ll"] / s["n"])
        for k in agg:
            agg[k] += s[k]
    acc = agg["correct"] / agg["n"]; exact = agg["exact"] / agg["n"]
    brier = agg["brier"] / agg["n"]; ll = agg["ll"] / agg["n"]
    if verbose:
        tag = f"{cfg['blend']:>6} w={cfg['w']:.2f} hl={cfg['half_life']} d0={cfg.get('d0',0.32):.2f}"
        if cfg["blend"] == "lambda":
            tag += f" beta={cfg.get('beta',0):.4f}"
        print(f"{tag:42s} | acc={acc*100:4.1f}% exact={exact*100:4.1f}% "
              f"brier={brier:.4f} logloss={ll:.4f}")
    return dict(acc=acc, exact=exact, brier=brier, ll=ll, per=per)


def base(**kw):
    cfg = dict(blend="prob", w=0.65, half_life=2555, ridge=0.005, d0=0.32, beta=0.005)
    cfg.update(kw)
    return cfg


if __name__ == "__main__":
    print("=== BASELINE (current model) ===")
    evaluate(base())
