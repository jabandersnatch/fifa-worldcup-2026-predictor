"""Expert-prior calibration of team ratings.

A results-only model (Elo + Dixon-Coles) cannot observe squad quality, ageing,
injuries or club form. The betting market and the Opta supercomputer *do* price
those in. So we treat them as a Bayesian prior on team strength and shrink each
team's results-based Elo toward the strength the experts imply — fixing
systematic over/under-ratings (e.g. an ageing Argentina, an over-performing
Morocco) throughout the whole simulation, not just in a post-hoc table.

The mapping is scale-preserving: we regress model Elo on log(expert win prob)
across the teams the experts cover, which converts an expert title probability
into an equivalent Elo on our own scale, then blend.
"""
import copy
import numpy as np

from . import config


def _fit_implied(values_by_team, ep):
    """Regress a rating on log(expert prob); return a function prob->rating
    (scale-preserving) and the list of covered teams."""
    teams = [t for t in ep if t in values_by_team]
    x = np.log(np.array([ep[t] for t in teams]))
    y = np.array([values_by_team[t] for t in teams])
    b, a = np.polyfit(x, y, 1)
    return (lambda p: a + b * np.log(p)), teams


def expert_prob():
    """Merge Opta (primary) and bookmaker-implied (secondary) title probabilities."""
    teams = set(config.OPTA_PROB) | set(config.MARKET_PROB)
    out = {}
    for t in teams:
        o, m = config.OPTA_PROB.get(t), config.MARKET_PROB.get(t)
        if o is not None and m is not None:
            out[t] = 0.6 * o + 0.4 * m
        else:
            out[t] = o if o is not None else m
    return out


def _calibrate_to_signal(dc, elo, signal, weight):
    """Shrink BOTH Elo and Dixon-Coles strength toward the strength implied by a
    positive per-team `signal` (e.g. squad value, or expert win prob). The
    mapping (regress rating on log(signal)) is scale-preserving, so it only
    re-orders relative strengths. Returns (dc_copy, elo_copy)."""
    if weight <= 0 or not signal:
        return dc, dict(elo)
    # Elo
    elo_fn, teams_e = _fit_implied(elo, signal)
    elo_cal = dict(elo)
    for t in teams_e:
        elo_cal[t] = (1 - weight) * elo[t] + weight * elo_fn(signal[t])
    # Dixon-Coles overall strength, balance preserved
    strength = {t: dc.attack[t] + dc.defence[t] for t in dc.attack}
    dc_fn, teams_d = _fit_implied(strength, signal)
    dc_cal = copy.copy(dc)
    dc_cal.attack = dict(dc.attack)
    dc_cal.defence = dict(dc.defence)
    for t in teams_d:
        delta = weight * (dc_fn(signal[t]) - strength[t])
        dc_cal.attack[t] += delta / 2
        dc_cal.defence[t] += delta / 2
    return dc_cal, elo_cal


def calibrate_squad_value(dc, elo, weight=0.5):
    """Fold in squad market value — fundamental data the native model can use."""
    return _calibrate_to_signal(dc, elo, config.SQUAD_VALUE_MEUR, weight)


def calibrate(dc, elo, weight=0.5):
    """Apply the external-engine expert prior (Opta + market) to both ratings."""
    dc_cal, elo_cal = _calibrate_to_signal(dc, elo, expert_prob(), weight)
    return dc_cal, elo_cal, {}
