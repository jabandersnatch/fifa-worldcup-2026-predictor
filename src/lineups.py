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


def _norm(name):
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return set(p for p in s.replace("-", " ").split() if len(p) > 2)


def player_shares(team, since="2023-01-01"):
    """{scorer: share of team's goals} over the recent window (non-own-goal)."""
    gs = data_loader.load_goalscorers(min_date=since)
    gs = gs[gs["team"] == team]
    counts = gs.groupby("scorer").size()
    total = counts.sum()
    return {} if total == 0 else (counts / total).to_dict()


def _in_xi(scorer, xi_tokens):
    toks = _norm(scorer)
    return any(toks & t for t in xi_tokens)        # shared surname token


def attack_multiplier(team, starting_xi, since="2023-01-01"):
    """Multiplier (<=1) on `team`'s attack given who starts. 1.0 = full strength."""
    shares = player_shares(team, since)
    if not shares:
        return 1.0, []
    xi_tokens = [_norm(p) for p in starting_xi]
    # only a missing PRIMARY scorer downgrades the attack
    missing = [(p, sh) for p, sh in shares.items()
               if sh >= PRIMARY and not _in_xi(p, xi_tokens)]
    lost = sum(sh for _, sh in missing)
    mult = max(MIN_MULT, 1.0 - DAMP * lost)
    notable = sorted(missing, key=lambda x: -x[1])
    return mult, notable
