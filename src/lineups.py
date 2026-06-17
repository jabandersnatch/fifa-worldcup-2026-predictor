"""Lineup-aware attack adjustment (forecast-time, per match).

Our model is team-strength based, so the principled way to use a confirmed
starting XI is via goal output: if a key scorer is NOT starting (rested/injured),
reduce the team's attack by that player's recent share of the team's goals
(damped, since a replacement still plays). A full-strength XI => no change.

Limitations (honest): captures attacking output only (not midfield/defence
quality), and cannot be backtested (no historical lineup feed).
"""
import unicodedata

from . import data_loader

DAMP = 0.6          # a benched scorer's share only partly lost (replacement plays)
MIN_MULT = 0.55     # floor on the attack multiplier
PRIMARY = 0.10      # only a missing PRIMARY scorer (>=10% of goals) downgrades attack;
                    # depth scorers rotating out of the XI is normal, not a downgrade


def _tokens(name):
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return [p for p in s.replace("-", " ").split() if len(p) > 2 and p != "al"]


def _surname(name):
    t = _tokens(name)
    return t[-1] if t else ""


def player_shares(team, since="2023-01-01"):
    """{scorer: share of team's goals} over the recent window (non-own-goal)."""
    gs = data_loader.load_goalscorers(min_date=since)
    gs = gs[gs["team"] == team]
    counts = gs.groupby("scorer").size()
    total = counts.sum()
    return {} if total == 0 else (counts / total).to_dict()


def _from_missing(shares, missing_surnames):
    """Multiplier from the set of surnames that are NOT available (primary only)."""
    missing = [(p, sh) for p, sh in shares.items()
               if sh >= PRIMARY and _surname(p) in missing_surnames]
    lost = sum(sh for _, sh in missing)
    return max(MIN_MULT, 1.0 - DAMP * lost), sorted(missing, key=lambda x: -x[1])


def attack_multiplier(team, starting_xi, since="2023-01-01"):
    """Multiplier (<=1) on `team`'s attack given who STARTS (surname match)."""
    shares = player_shares(team, since)
    if not shares:
        return 1.0, []
    xi_surnames = {_surname(p) for p in starting_xi}
    return _from_missing(shares, {_surname(p) for p in shares if _surname(p) not in xi_surnames})


def multiplier_from_out(team, out_players, since="2023-01-01"):
    """Multiplier (<=1) given an explicit list of OUT/injured players (surname
    match) — cleaner than a full XI when you just know who is unavailable."""
    shares = player_shares(team, since)
    if not shares:
        return 1.0, []
    return _from_missing(shares, {_surname(p) for p in out_players})
