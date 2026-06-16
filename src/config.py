"""Static configuration for the 2026 FIFA World Cup.

Group composition was verified two ways: (1) clustering the actual fixture graph
in the martj42 results dataset into 4-team cliques, and (2) ESPN's published group
list. Team names use the dataset's spelling (e.g. "Czech Republic", "Turkey",
"Curaçao") so they join cleanly with the historical data.
"""

# Path to the historical + 2026 results CSV (auto-downloaded / refreshed).
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# Canonical group letter -> 4 teams (dataset spellings).
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Host nations (small home-edge applied when they play in their own country).
HOSTS = {"United States", "Mexico", "Canada"}

# Round of 32 matchups (match number -> (home slot, away slot)), verified from
# ESPN / FIFA. Slot codes: "1A" = winner Group A, "2A" = runner-up Group A,
# "3CEFHI" = a best-third-placed team whose group is one of {C,E,F,H,I}.
R32 = {
    73: ("2A", "2B"),
    74: ("1C", "2F"),
    75: ("1E", "3ABCDF"),
    76: ("1F", "2C"),
    77: ("2E", "2I"),
    78: ("1I", "3CDFGH"),
    79: ("1A", "3CEFHI"),
    80: ("1L", "3EHIJK"),
    81: ("1G", "3AEHIJ"),
    82: ("1D", "3BEFIJ"),
    83: ("1H", "2J"),
    84: ("2K", "2L"),
    85: ("1B", "3EFGIJ"),
    86: ("2D", "2G"),
    87: ("1J", "2H"),
    88: ("1K", "3DEIJL"),
}

# Knockout tree (match -> the two earlier matches whose winners meet).
# This is FIFA's OFFICIAL 2026 bracket adjacency (verified from Wikipedia's
# knockout-stage bracket), so the simulated tournament uses the real halves.
BRACKET_TREE = {
    89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
    93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87),
    97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96),
    101: (97, 98), 102: (99, 100),
    104: (101, 102),
}

# Reverse map: team -> group letter.
TEAM_GROUP = {team: letter for letter, teams in GROUPS.items() for team in teams}

WC_TEAMS = [team for teams in GROUPS.values() for team in teams]

# Bookmaker title odds (American), mid-June 2026, used as an external reference
# and a blended-consensus signal. Sourced from FOX/CBS/Squawka after Spain's
# opening draw with Cape Verde shuffled the board. Includes the bookmaker margin.
MARKET_ODDS_AMERICAN = {
    "France": 430, "Spain": 500, "Portugal": 600, "England": 650,
    "Brazil": 850, "Argentina": 1000, "Germany": 1400,
}


def _american_to_prob(odds):
    return 100.0 / (odds + 100.0) if odds > 0 else (-odds) / (-odds + 100.0)


# team -> implied win probability (raw, with vig)
MARKET_PROB = {t: _american_to_prob(o) for t, o in MARKET_ODDS_AMERICAN.items()}

# Opta supercomputer title probabilities (25,000 sims, mid-June 2026), theanalyst.com.
# An independent expert model, used as a second external signal in the consensus.
OPTA_PROB = {
    "Spain": 0.161, "France": 0.130, "England": 0.112, "Argentina": 0.104,
    "Portugal": 0.070, "Brazil": 0.066, "Germany": 0.051, "Netherlands": 0.036,
    "Norway": 0.035, "Belgium": 0.024, "Colombia": 0.021, "Morocco": 0.019,
    "Croatia": 0.016, "Ecuador": 0.014, "United States": 0.012,
}

# Total squad market value (€M), Transfermarkt via planetfootball, June 2026.
# FUNDAMENTAL DATA (a measurement of squad quality), not a prediction — so it
# belongs in the native model. It is the signal a results-only model is blind to:
# Argentina rank #7 here despite a #1 results record (an ageing squad).
SQUAD_VALUE_MEUR = {
    "France": 1520, "England": 1360, "Spain": 1220, "Portugal": 1010, "Germany": 947,
    "Brazil": 928.2, "Argentina": 807.5, "Netherlands": 754.2, "Norway": 589.9,
    "Belgium": 547.5, "Ivory Coast": 522.1, "Senegal": 478.1, "Turkey": 473.7,
    "Morocco": 447.7, "Sweden": 406.08, "Croatia": 387.3, "United States": 385.6,
    "Ecuador": 368.7, "Uruguay": 359.3, "Switzerland": 332.5, "Colombia": 302.35,
    "Japan": 270.85, "Algeria": 256.9, "Austria": 245.2, "Ghana": 234.5,
    "Canada": 198.65, "Mexico": 191.85, "Czech Republic": 188.18, "Scotland": 170.25,
    "Paraguay": 153.65, "Bosnia and Herzegovina": 146.4, "DR Congo": 143.9,
    "South Korea": 139.05, "Egypt": 116.48, "Uzbekistan": 85.33, "Australia": 77.45,
    "Tunisia": 69.95, "Haiti": 55.9, "Cape Verde": 49.25, "South Africa": 49.25,
    "Saudi Arabia": 40.68, "Panama": 34.55, "New Zealand": 34.45, "Iran": 32.05,
    "Curaçao": 25.78, "Iraq": 21.2, "Jordan": 20.3, "Qatar": 19.93,
}

# Bookmaker Golden Boot / top-scorer odds (American), mid-June 2026, FOX Sports.
# Player names use the goalscorer-dataset spelling so they join cleanly.
GOLDEN_BOOT_ODDS_AMERICAN = {
    "Kylian Mbappé": 550, "Harry Kane": 600, "Kai Havertz": 950, "Erling Haaland": 1300,
    "Julián Alvarez": 1600, "Lionel Messi": 1700, "Mikel Oyarzabal": 1900,
    "Cristiano Ronaldo": 2000, "Folarin Balogun": 2200, "Vinícius Júnior": 2200,
    "Lamine Yamal": 3000, "Michael Olise": 3000, "Raphinha": 3500,
    "Jamal Musiala": 3500, "Lautaro Martínez": 4500,
}
GOLDEN_BOOT_PROB = {p: _american_to_prob(o) for p, o in GOLDEN_BOOT_ODDS_AMERICAN.items()}

# Consensus weights for the headline Golden Boot ranking (market-led; the market
# captures penalty duties, role and pedigree the goals-only model can't see).
BOOT_WEIGHTS = {"model": 0.40, "market": 0.60}

