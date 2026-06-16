"""Team -> FIFA confederation map for the major footballing nations.

Used to estimate and correct cross-confederation rating bias: a results-only Elo
inflates teams that accumulate rating inside a strong but largely closed pool
(e.g. CONMEBOL) or against weak intercontinental friendly opposition. Coverage
focuses on teams that contest competitive internationals; unmapped minnows get
no confederation adjustment.
"""

_MEMBERS = {
    "UEFA": [
        "Spain", "France", "England", "Germany", "Portugal", "Netherlands", "Italy",
        "Belgium", "Croatia", "Switzerland", "Denmark", "Austria", "Norway", "Sweden",
        "Poland", "Serbia", "Ukraine", "Czech Republic", "Turkey", "Scotland", "Wales",
        "Hungary", "Romania", "Greece", "Russia", "Bosnia and Herzegovina", "Slovakia",
        "Slovenia", "Republic of Ireland", "Iceland", "Finland", "Albania",
        "North Macedonia", "Georgia", "Bulgaria", "Northern Ireland", "Montenegro",
        "Kosovo", "Israel", "Luxembourg", "Armenia", "Azerbaijan", "Belarus", "Estonia",
        "Latvia", "Lithuania", "Cyprus", "Kazakhstan", "Moldova", "Malta",
        "Faroe Islands", "Andorra", "San Marino", "Gibraltar", "Liechtenstein",
    ],
    "CONMEBOL": [
        "Brazil", "Argentina", "Uruguay", "Colombia", "Chile", "Peru", "Ecuador",
        "Paraguay", "Venezuela", "Bolivia",
    ],
    "CONCACAF": [
        "United States", "Mexico", "Canada", "Costa Rica", "Jamaica", "Panama",
        "Honduras", "El Salvador", "Haiti", "Curaçao", "Trinidad and Tobago",
        "Guatemala", "Nicaragua", "Cuba", "Suriname", "Saint Kitts and Nevis",
        "Grenada", "Antigua and Barbuda", "Belize", "Dominican Republic",
        "Puerto Rico", "Bermuda", "Aruba", "Saint Lucia", "Barbados", "Guyana",
    ],
    "CAF": [
        "Morocco", "Senegal", "Nigeria", "Egypt", "Cameroon", "Ghana", "Algeria",
        "Tunisia", "Ivory Coast", "Mali", "South Africa", "Cape Verde", "DR Congo",
        "Burkina Faso", "Guinea", "Zambia", "Angola", "Equatorial Guinea", "Gabon",
        "Benin", "Uganda", "Mauritania", "Namibia", "Mozambique", "Madagascar",
        "Kenya", "Zimbabwe", "Tanzania", "Congo", "Sudan", "Libya", "Togo",
        "Sierra Leone", "Niger", "Comoros", "Malawi", "Gambia", "Ethiopia",
        "Rwanda", "Botswana", "Guinea-Bissau", "Liberia",
    ],
    "AFC": [
        "Japan", "South Korea", "Iran", "Saudi Arabia", "Australia", "Qatar", "Iraq",
        "United Arab Emirates", "Uzbekistan", "Jordan", "China PR", "Oman", "Bahrain",
        "Syria", "Lebanon", "Palestine", "Kuwait", "India", "Thailand", "Vietnam",
        "Kyrgyzstan", "Tajikistan", "Turkmenistan", "North Korea", "Yemen",
        "Hong Kong", "Philippines", "Indonesia", "Malaysia", "Myanmar", "Afghanistan",
        "Singapore", "Maldives",
    ],
    "OFC": [
        "New Zealand", "Fiji", "Solomon Islands", "Papua New Guinea", "Tahiti",
        "Vanuatu", "New Caledonia", "Samoa", "Tonga", "Cook Islands",
    ],
}

CONFEDERATION = {team: conf for conf, teams in _MEMBERS.items() for team in teams}

# dWinProb per Elo point near a coin-flip (ln(10)/400 * 0.25).
_PROB_PER_ELO = 0.00144


def confed(team):
    return CONFEDERATION.get(team)


def estimate_offsets(matches, half_life_years=4.0, since="2014-01-01",
                     cap=140.0):
    """Estimate per-confederation Elo offsets from intercontinental matches.

    Runs a plain chronological Elo over `matches` and accumulates each team's
    pre-match residual (actual − expected) in matches against another
    confederation. A positive mean residual means Elo under-rates that
    confederation. Residuals are converted to Elo points, centred, and capped.
    Returns {confederation: elo_offset}. Use only pre-cutoff data to avoid leak.
    """
    import numpy as np
    import pandas as pd
    from collections import defaultdict
    from .elo import _k_factor, _gd_multiplier

    ratings = defaultdict(lambda: 1500.0)
    acc = defaultdict(lambda: [0.0, 0.0])
    ref = matches["date"].max()
    since = pd.Timestamp(since)
    for r in matches.itertuples(index=False):
        h, a = r.home_team, r.away_team
        rh, ra = ratings[h], ratings[a]
        adv = 0.0 if r.neutral else 65.0
        eh = 1.0 / (1.0 + 10 ** (-((rh + adv) - ra) / 400.0))
        sh = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
        k = _k_factor(r.tournament) * _gd_multiplier(r.home_score - r.away_score)
        d = k * (sh - eh)
        ratings[h] = rh + d
        ratings[a] = ra - d
        ch, ca = confed(h), confed(a)
        if ch and ca and ch != ca and r.date >= since:
            w = 0.5 ** (((ref - r.date).days) / (365.0 * half_life_years))
            acc[ch][0] += w * (sh - eh); acc[ch][1] += w
            acc[ca][0] += w * ((1 - sh) - (1 - eh)); acc[ca][1] += w

    raw = {c: (acc[c][0] / acc[c][1]) / _PROB_PER_ELO for c in acc if acc[c][1] > 5}
    if not raw:
        return {}
    mean = float(np.mean(list(raw.values())))
    return {c: float(np.clip(v - mean, -cap, cap)) for c, v in raw.items()}


def apply_offsets(elo, offsets):
    """Return a copy of the Elo dict with each team shifted by its confederation
    offset (teams in unmapped confederations are unchanged)."""
    out = dict(elo)
    for t in out:
        c = confed(t)
        if c in offsets:
            out[t] = out[t] + offsets[c]
    return out
