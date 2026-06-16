"""World-Football-style Elo ratings computed chronologically over match history.

Used as a secondary signal blended into outcome probabilities and as a sanity
check on team strength.
"""
import math
from collections import defaultdict

# K-factor by tournament importance (eloratings.net-inspired).
def _k_factor(tournament):
    t = tournament.lower()
    if "world cup" in t and "qualif" not in t:
        return 60
    if any(s in t for s in ("uefa euro", "copa am", "african cup", "afc asian", "confederations", "nations league")):
        return 50
    if "qualif" in t:
        return 40
    if "friendly" in t:
        return 20
    return 30


def _gd_multiplier(goal_diff):
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11 + g) / 8.0


def compute_elo(matches, home_advantage=65.0, base=1500.0):
    """Return dict team -> final Elo rating. `matches` sorted ascending by date."""
    ratings = defaultdict(lambda: base)
    for row in matches.itertuples(index=False):
        h, a = row.home_team, row.away_team
        rh, ra = ratings[h], ratings[a]
        adv = 0.0 if row.neutral else home_advantage
        exp_h = 1.0 / (1.0 + 10 ** (-((rh + adv) - ra) / 400.0))
        if row.home_score > row.away_score:
            score_h = 1.0
        elif row.home_score < row.away_score:
            score_h = 0.0
        else:
            score_h = 0.5
        k = _k_factor(row.tournament) * _gd_multiplier(row.home_score - row.away_score)
        delta = k * (score_h - exp_h)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
    return dict(ratings)


def outcome_probs(rating_home, rating_away, neutral=True, home_advantage=65.0,
                  draw_base=0.32):
    """Convert two Elo ratings into (P_home, P_draw, P_away)."""
    adv = 0.0 if neutral else home_advantage
    exp_h = 1.0 / (1.0 + 10 ** (-((rating_home + adv) - rating_away) / 400.0))
    # Draw most likely when teams are even; tapers as the gap grows.
    p_draw = draw_base * (1.0 - abs(2 * exp_h - 1.0))
    p_home = exp_h * (1.0 - p_draw)
    p_away = (1.0 - exp_h) * (1.0 - p_draw)
    return p_home, p_draw, p_away
