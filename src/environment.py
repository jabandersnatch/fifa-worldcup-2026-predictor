"""Match-environment adjustments. Currently: ALTITUDE.

Validated on our own data (experiments/altitude_test.py): each +1000 m of an
away team's altitude shock adds ~+0.13 goals of home goal-difference beyond the
Elo/DC strength gap (t=3.7). We apply that marginal effect as a supremacy shift.

Heat/humidity (net goal effect ambiguous in the literature) and rest/travel
(found insignificant in the 2014 World Cup study) are deliberately NOT modelled.
"""

ALT_COEF = 0.13      # goals of home goal-diff per 1000 m of relative shock
ALT_CAP = 0.5        # cap the adjustment (avoid extreme extrapolation)
_DEFAULT_HOME_ALT = 30

# Venue city -> elevation (m). 2026 World Cup venues + key altitude/host cities.
CITY_ALT = {
    # 2026 venues
    "Mexico City": 2240, "Zapopan": 1566, "Guadalupe": 540, "Atlanta": 320,
    "Arlington": 165, "Kansas City": 270, "Toronto": 76, "Seattle": 46,
    "Foxborough": 43, "Inglewood": 40, "Houston": 32, "Philadelphia": 12,
    "Santa Clara": 9, "Vancouver": 4, "East Rutherford": 5, "Miami Gardens": 3,
    # famous high-altitude venues (qualifiers etc.)
    "La Paz": 3640, "Quito": 2850, "Bogotá": 2640, "Bogota": 2640, "Cusco": 3400,
    "Sucre": 2810, "Toluca": 2660, "Addis Ababa": 2355, "Sana'a": 2250,
    # historical WC host cities with notable elevation
    "Brasília": 1172, "Belo Horizonte": 852, "Johannesburg": 1753,
    "Pretoria": 1339, "Rustenburg": 1500, "Nelspruit": 660,
}

# National-team home altitude (m); unmapped teams default to ~sea level.
TEAM_ALT = {
    "Bolivia": 3640, "Ecuador": 2850, "Colombia": 2640, "Mexico": 2240,
    "Peru": 150, "Chile": 570, "Saudi Arabia": 600, "Iran": 1200,
    "Guatemala": 1500, "Costa Rica": 1170, "Honduras": 990, "El Salvador": 660,
    "Armenia": 990, "Kenya": 1660, "Ethiopia": 2355, "Rwanda": 1567,
    "South Africa": 1750, "Zimbabwe": 1490, "Zambia": 1280, "Afghanistan": 1800,
    "Yemen": 2250, "Kyrgyzstan": 800,
}


def team_alt(team):
    return TEAM_ALT.get(team, _DEFAULT_HOME_ALT)


def venue_alt(city, home_team=None, neutral=True):
    """Elevation of the venue. For a known city, use it. For an unknown city in a
    non-neutral match, the home team is at home, so use its altitude."""
    if city in CITY_ALT:
        return CITY_ALT[city]
    if not neutral and home_team is not None:
        return team_alt(home_team)
    return 0.0


def altitude_supremacy(home, away, city, neutral=True):
    """Goal-difference adjustment in favour of the home team from altitude.
    Positive => home favoured (away is less acclimatised to the venue)."""
    v = venue_alt(city, home, neutral)
    if v < 500:                                  # negligible below ~500 m
        return 0.0
    shock_home = max(0.0, v - team_alt(home))
    shock_away = max(0.0, v - team_alt(away))
    adj = ALT_COEF * (shock_away - shock_home) / 1000.0
    return max(-ALT_CAP, min(ALT_CAP, adj))
