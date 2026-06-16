"""Validate that altitude carries predictive signal beyond team strength.

For non-neutral matches the home side plays at its home altitude, so the away
side's "altitude shock" = max(0, home_alt - away_alt). If altitude matters, the
home goal difference should rise with that shock even after controlling for the
Elo strength gap. We regress goal_diff ~ elo_diff + away_shock and check the
altitude coefficient (literature: ~+0.5 goals per 1000 m).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src import data_loader
from src.elo import compute_elo

# Home altitude (m) of national teams that play home games at altitude; others ~0.
TEAM_ALT = {
    "Bolivia": 3640, "Ecuador": 2850, "Colombia": 2640, "Mexico": 2240,
    "Peru": 150, "Chile": 570, "Saudi Arabia": 600, "Iran": 1200, "Afghanistan": 1800,
    "Guatemala": 1500, "Costa Rica": 1170, "Honduras": 990, "El Salvador": 660,
    "Armenia": 990, "Kenya": 1660, "Ethiopia": 2355, "Rwanda": 1567, "Zimbabwe": 1490,
    "Zambia": 1280, "South Africa": 1750, "Eswatini": 1100, "Lesotho": 1600,
    "Yemen": 2250, "Kyrgyzstan": 800, "Nepal": 1400,
}


def alt(team):
    return TEAM_ALT.get(team, 30)


def main():
    df = data_loader.load_matches(min_date="2008-01-01")
    elo = compute_elo(df)
    df = df[~df["neutral"]].copy()                  # home side at its own altitude
    rows = []
    for r in df.itertuples(index=False):
        if r.home_team not in elo or r.away_team not in elo:
            continue
        shock = max(0.0, alt(r.home_team) - alt(r.away_team)) / 1000.0
        rows.append((r.home_score - r.away_score,
                     (elo[r.home_team] + 65 - elo[r.away_team]) / 100.0,
                     shock))
    arr = np.array(rows)
    y, elo_diff, shock = arr[:, 0], arr[:, 1], arr[:, 2]

    # OLS: goal_diff = b0 + b1*elo_diff + b2*shock
    X = np.column_stack([np.ones_like(elo_diff), elo_diff, shock])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - 3
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    names = ["intercept", "elo_diff(/100)", "away_altitude_shock(/1000m)"]
    print(f"n = {len(y):,} non-neutral matches; {int((shock>0).sum()):,} with altitude shock>0\n")
    for nm, b, s in zip(names, beta, se):
        print(f"  {nm:30s} coef={b:+.3f}  SE={s:.3f}  t={b/s:+.1f}")
    print(f"\nAltitude effect: each +1000 m of away shock => {beta[2]:+.2f} goals of "
          f"home goal-difference (literature ~ +0.5).")


if __name__ == "__main__":
    main()
