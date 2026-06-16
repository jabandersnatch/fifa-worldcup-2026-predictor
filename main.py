#!/usr/bin/env python3
"""World Cup 2026 score predictor — CLI.

Commands:
  refresh    re-download the latest results (incl. live 2026 scores)
  predict    predicted scorelines + W/D/L for upcoming group matches
  simulate   Monte Carlo tournament: advancement % and title odds
  backtest   measured accuracy on the 2022 World Cup
"""
import argparse
import os
import sys

import pandas as pd
from tabulate import tabulate

from src import config, data_loader, backtest as bt
from src.dixon_coles import DixonColes
from src.elo import compute_elo
from src.predict import Predictor

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _build_predictor(min_date="2010-01-01", elo_weight=0.70, expert_prior=0.0,
                     squad_weight=0.50):
    from src import calibration, confederations
    df = data_loader.load_matches(min_date=min_date)
    print(f"Training on {len(df):,} matches ({df['date'].min().date()} → "
          f"{df['date'].max().date()})…", file=sys.stderr)
    dc = DixonColes().fit(df)
    elo = compute_elo(df)
    # NATIVE MODEL (our data only):
    #  (1) cross-confederation calibration — data-derived Elo correction
    #      (validated +2.9pp WC accuracy).
    elo = confederations.apply_offsets(elo, confederations.estimate_offsets(df))
    #  (2) squad market value — fundamental data that lets the model see squad
    #      quality/ageing (e.g. Argentina's 7th-ranked squad). Not a prediction.
    dc, elo = calibration.calibrate_squad_value(dc, elo, weight=squad_weight)
    # OPTIONAL external-engine prior (Opta + market odds), off by default:
    if expert_prior > 0:
        dc, elo, _ = calibration.calibrate(dc, elo, weight=expert_prior)
    return Predictor(dc, elo, elo_weight=elo_weight), dc, elo


def cmd_refresh(_):
    path = data_loader.download()
    wc = data_loader.wc2026_matches()
    played = wc[wc["played"]]
    print(f"Downloaded latest data → {path}")
    print(f"WC2026 group matches: {len(wc)} total, {len(played)} played, "
          f"{len(wc) - len(played)} upcoming.")
    if len(played):
        last = played.tail(5)[["date", "home_team", "away_team", "home_score", "away_score"]]
        print("\nMost recent results:")
        print(tabulate(last, headers="keys", showindex=False, tablefmt="simple"))


def cmd_predict(args):
    predictor, _, _ = _build_predictor(elo_weight=args.elo_weight,
                                       expert_prior=args.expert_prior,
                                       squad_weight=args.squad_weight)
    wc = data_loader.wc2026_matches()
    upcoming = wc[~wc["played"]]

    fb = None
    if args.compare_forebet:
        from src import forebet
        try:
            print("Fetching Forebet predictions for corroboration…", file=sys.stderr)
            fb = forebet.predictions()
            print(f"  got {len(fb)} Forebet match predictions.", file=sys.stderr)
        except Exception as e:                       # network/layout failure
            print(f"  Forebet unavailable ({e}); showing our model only.", file=sys.stderr)

    rows = []
    agree = total = 0
    for r in upcoming.itertuples(index=False):
        neutral = bool(r.neutral)
        if r.home_team in config.HOSTS and r.country == r.home_team:
            neutral = False
        p = predictor.predict(r.home_team, r.away_team, neutral=neutral)
        top3 = " · ".join(f"{h}-{a} {pr*100:.0f}%" for h, a, pr in p["top_scores"][:3])
        ours = f"{p['ml_home']}-{p['ml_away']}"
        row = {
            "date": r.date.date(),
            "group": config.TEAM_GROUP.get(r.home_team, "?"),
            "match": f"{r.home_team} vs {r.away_team}",
            "ours": ours,
            "xG": f"{p['exp_home']:.1f}-{p['exp_away']:.1f}",
            "P(H/D/A)": f"{p['p_home']*100:.0f}/{p['p_draw']*100:.0f}/{p['p_away']*100:.0f}",
        }
        if fb is not None:
            from src import forebet
            fbs = forebet.lookup(fb, r.home_team, r.away_team)
            row["forebet"] = fbs or "—"
            if fbs:
                total += 1
                # agreement = same result sign (both home win / draw / away win)
                oh, oa = p["ml_home"], p["ml_away"]
                fh, fa = (int(x) for x in fbs.split("-"))
                agree += (oh - oa > 0) == (fh - fa > 0) and (oh == oa) == (fh == fa)
        else:
            row["most likely scorelines"] = top3
        rows.append(row)

    out = pd.DataFrame(rows)
    print(tabulate(out, headers="keys", showindex=False, tablefmt="github"))
    if fb is not None and total:
        print(f"\nForebet corroboration: same predicted result in {agree}/{total} "
              f"matches ({agree/total*100:.0f}%).")
    csv = os.path.join(DATA_DIR, "predictions_upcoming.csv")
    out.to_csv(csv, index=False)
    print(f"Saved {len(out)} predictions → {csv}")


def cmd_simulate(args):
    import math
    from src.simulate import Tournament
    predictor, _, _ = _build_predictor(elo_weight=args.elo_weight,
                                       expert_prior=args.expert_prior,
                                       squad_weight=args.squad_weight)
    wc = data_loader.wc2026_matches()
    print(f"Running {args.sims:,} Monte Carlo simulations…", file=sys.stderr)
    result = Tournament(wc, predictor).run(n_sims=args.sims)
    n = args.sims

    def engine_prob(team):
        vals = [v for v in (config.OPTA_PROB.get(team), config.MARKET_PROB.get(team))
                if v is not None]
        return sum(vals) / len(vals) if vals else None

    rows = []
    for team, p in result.items():
        native = p["Champion"]
        ci = 1.96 * math.sqrt(max(native * (1 - native), 0) / n)
        opta = config.OPTA_PROB.get(team)
        market = config.MARKET_PROB.get(team)
        eng = engine_prob(team)
        # Tier 3 — cross-validated ensemble: weight our native model and the
        # external-engine consensus equally (engines imputed with native if absent).
        ensemble = 0.5 * native + 0.5 * (eng if eng is not None else native)
        rows.append({
            "team": team, "grp": config.TEAM_GROUP[team],
            "Native±95%": f"{native*100:.1f} ±{ci*100:.1f}",
            "Opta": f"{opta*100:.1f}" if opta is not None else "—",
            "Market": f"{market*100:.1f}" if market is not None else "—",
            "Engines": f"{eng*100:.1f}" if eng is not None else "—",
            "Ensemble": f"{ensemble*100:.1f}",
            "_sort": ensemble,
            "_native": native, "_R32": p["R32"], "_R16": p["R16"],
            "_QF": p["QF"], "_SF": p["SF"], "_Final": p["Final"],
        })
    df = pd.DataFrame(rows).sort_values("_sort", ascending=False)
    df["Ensemble"] = (df["_sort"] / df["_sort"].sum() * 100).map(lambda v: f"{v:.1f}")
    show = df[["team", "grp", "Native±95%", "Opta", "Market", "Engines", "Ensemble"]]
    print(tabulate(show.head(args.top), headers="keys", showindex=False, tablefmt="github"))
    print("\n(All %. THREE TIERS — Native = our model only: results + confederation "
          "calibration + squad value, NO prediction engines (±95% CI). "
          "Engines = external prediction consensus (Opta + bookmaker; EA Sports & "
          "Nate Silver also pick Spain). Ensemble = 50/50 cross-validation of "
          "Native and Engines, renormalised over 48 teams.)")
    csv = os.path.join(DATA_DIR, "simulation_odds.csv")
    (df.drop(columns=["_sort"])
       .rename(columns={"_native": "native_raw", "_R32": "R32", "_R16": "R16",
                        "_QF": "QF", "_SF": "SF", "_Final": "Final"})
       .to_csv(csv, index=False))
    print(f"Saved full 48-team three-tier odds → {csv}")


def cmd_goldenboot(args):
    from src.simulate import Tournament
    from src import scorers
    predictor, _, _ = _build_predictor(elo_weight=args.elo_weight,
                                       expert_prior=args.expert_prior,
                                       squad_weight=args.squad_weight)
    wc = data_loader.wc2026_matches()
    print(f"Running {args.sims:,} simulations for expected match counts…", file=sys.stderr)
    result = Tournament(wc, predictor).run(n_sims=args.sims)
    gs = data_loader.load_goalscorers()
    proj = scorers.project(gs, predictor, result, wc, top_n=args.top)
    disp = proj.copy()
    disp["wc26_goals"] = disp["wc26_goals"].astype(int)
    disp["model_xG"] = disp["projected"].round(1)
    disp["market"] = (disp["market_prob"] * 100).round(1).map(lambda v: f"{v}%" if v > 0 else "—")
    disp["consensus"] = (disp["consensus"] * 100).round(1)
    disp = disp[["scorer", "team", "wc26_goals", "model_xG", "market", "consensus"]]
    disp.columns = ["player", "team", "goals_so_far", "model_xG", "market%", "consensus%"]
    print(tabulate(disp, headers="keys", showindex=False, tablefmt="github"))
    print("\n(model_xG = model's expected goals; market% = bookmaker top-scorer implied prob; "
          "consensus% = blended ranking. Sorted by consensus.)")
    csv = os.path.join(DATA_DIR, "golden_boot.csv")
    proj.to_csv(csv, index=False)
    print(f"Saved Golden Boot projection → {csv}")


def cmd_backtest(args):
    df = data_loader.load_matches(min_date="2002-01-01")
    print("Backtesting across the 2014/2018/2022 World Cups "
          "(trained only on pre-tournament data each time)…", file=sys.stderr)
    res = bt.run_cv(df, elo_weight=args.elo_weight)
    print(f"\nOut-of-sample backtest — {res['n']} World Cup matches across 3 tournaments:")
    print(f"  Outcome accuracy (W/D/L):   {res['outcome_accuracy']*100:.1f}%")
    print(f"  Exact-score accuracy:       {res['exact_score_accuracy']*100:.1f}%")
    print(f"  Brier score (lower=better): {res['brier']:.3f}")
    print(f"  Log-loss   (random=1.099):  {res['logloss']:.3f}")
    if args.verbose:
        rows = [[y, p["n"], f"{p['outcome_accuracy']*100:.1f}%",
                 f"{p['exact_score_accuracy']*100:.1f}%", f"{p['logloss']:.3f}"]
                for y, p in res["per"].items()]
        print()
        print(tabulate(rows, headers=["WC", "matches", "accuracy", "exact", "logloss"],
                       tablefmt="simple"))


def main():
    ap = argparse.ArgumentParser(description="2026 World Cup score predictor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("refresh", "predict", "simulate", "backtest", "golden-boot"):
        sp = sub.add_parser(name)
        sp.add_argument("--elo-weight", type=float, default=0.70,
                        help="blend weight for the Elo signal (0–1)")
        if name in ("predict", "simulate", "golden-boot"):
            sp.add_argument("--squad-weight", type=float, default=0.50,
                            help="weight on squad market-value data in the native "
                                 "model (0–1)")
            sp.add_argument("--expert-prior", type=float, default=0.0,
                            help="fold external-engine odds (Opta+market) INTO the "
                                 "native model (0–1); default 0 keeps it native")
        if name in ("simulate", "golden-boot"):
            sp.add_argument("--sims", type=int, default=50000)
            sp.add_argument("--top", type=int, default=20,
                            help="rows to display")
        if name == "predict":
            sp.add_argument("--compare-forebet", action="store_true",
                            help="pull Forebet's score predictions and show them "
                                 "next to ours for each match")
        if name == "backtest":
            sp.add_argument("--verbose", action="store_true")

    args = ap.parse_args()
    {"refresh": cmd_refresh, "predict": cmd_predict,
     "simulate": cmd_simulate, "backtest": cmd_backtest,
     "golden-boot": cmd_goldenboot}[args.cmd](args)


if __name__ == "__main__":
    main()
