"""Golden Boot / best-striker projection.

A player's projected tournament total = goals already scored in 2026 + their
share of their team's *expected future goals*. Future team goals come from the
Dixon-Coles expected goals for remaining group matches plus the expected number
of knockout matches the team plays (from the Monte Carlo simulation) times a
per-match knockout scoring rate. Player share is estimated from recent national-
team scoring, with current-tournament form up-weighted.
"""
import numpy as np
import pandas as pd

from . import config

FORM_BOOST = 3.0          # weight on a 2026 WC goal vs a historical goal
MIN_HIST_GOALS = 3        # squad-membership proxy for players yet to score in 2026


def _team_future_goals(predictor, sim_result, wc_matches):
    dc = predictor.dc
    mean_def = np.mean([dc.defence[t] for t in config.WC_TEAMS if t in dc.defence])

    # remaining group expected goals per team
    rem_xg = {t: 0.0 for t in config.WC_TEAMS}
    for r in wc_matches.itertuples(index=False):
        if r.played:
            continue
        neutral = bool(r.neutral)
        if r.home_team in config.HOSTS and r.country == r.home_team:
            neutral = False
        lam, mu = predictor.expected_goals(r.home_team, r.away_team, neutral, city=r.city)
        rem_xg[r.home_team] = rem_xg.get(r.home_team, 0.0) + lam
        rem_xg[r.away_team] = rem_xg.get(r.away_team, 0.0) + mu

    future = {}
    for t in config.WC_TEAMS:
        ko_xg_per_match = np.exp(dc.attack.get(t, 0.0) - mean_def)
        exp_ko_matches = sum(sim_result[t][k] for k in ("R32", "R16", "QF", "SF", "Final"))
        future[t] = rem_xg.get(t, 0.0) + ko_xg_per_match * exp_ko_matches
    return future


def project(goalscorers, predictor, sim_result, wc_matches, top_n=20):
    gs = goalscorers[goalscorers["team"].isin(config.WC_TEAMS)].copy()
    is_wc26 = (gs["date"] >= pd.Timestamp("2026-06-01")) & (gs["tournament"] == "FIFA World Cup") \
        if "tournament" in gs.columns else (gs["date"] >= pd.Timestamp("2026-06-01"))

    hist = gs.groupby(["scorer", "team"]).size().rename("hist_goals")
    wc26 = gs[is_wc26].groupby(["scorer", "team"]).size().rename("wc26_goals")
    tbl = pd.concat([hist, wc26], axis=1).fillna(0).reset_index()
    tbl["weight"] = tbl["hist_goals"] + FORM_BOOST * tbl["wc26_goals"]

    # Denominator is the team's FULL scoring spread (every scorer in the window),
    # so a dominant striker on a low-scoring side doesn't soak up an unrealistic
    # share. Goals attributed to non-candidates (fringe/retired) are simply left
    # unassigned.
    team_total = tbl.groupby("team")["weight"].transform("sum")
    tbl["share"] = tbl["weight"] / team_total

    # candidate squad members: scored in 2026 OR a recent regular scorer
    tbl = tbl[(tbl["wc26_goals"] > 0) | (tbl["hist_goals"] >= MIN_HIST_GOALS)].copy()

    future = _team_future_goals(predictor, sim_result, wc_matches)
    tbl["future_goals"] = tbl.apply(
        lambda r: r["share"] * future.get(r["team"], 0.0), axis=1)
    tbl["projected"] = tbl["wc26_goals"] + tbl["future_goals"]

    # --- Expert blend: combine the goals-only model with bookmaker top-scorer
    # odds. The market captures penalty duties, role and pedigree the model
    # can't see; the model captures live form and an easy/hard tournament run. ---
    pool = tbl[(tbl["projected"] > 0.3) | (tbl["scorer"].isin(config.GOLDEN_BOOT_PROB))].copy()
    pool["model_norm"] = pool["projected"] / pool["projected"].sum()
    pool["market_prob"] = pool["scorer"].map(config.GOLDEN_BOOT_PROB).fillna(0.0)
    msum = pool["market_prob"].sum()
    pool["market_norm"] = pool["market_prob"] / msum if msum else 0.0
    wm, wk = config.BOOT_WEIGHTS["model"], config.BOOT_WEIGHTS["market"]
    pool["consensus"] = wm * pool["model_norm"] + wk * pool["market_norm"]
    pool["consensus"] /= pool["consensus"].sum()

    pool = pool.sort_values("consensus", ascending=False).head(top_n)
    return pool[["scorer", "team", "wc26_goals", "projected",
                 "market_prob", "consensus"]].reset_index(drop=True)
