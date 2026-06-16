# World Cup 2026 Score Predictor

A locally-runnable statistical model that predicts **scorelines** for 2026 FIFA World Cup
matches and simulates the rest of the tournament. It reflects the tournament *as it is
actually unfolding* — it ingests live results so upsets and current form shape every prediction.

## What it does

- **Predicts scorelines** for upcoming group matches (most-likely score + expected goals +
  win/draw/loss probabilities).
- **Simulates the whole tournament** (Monte Carlo) under the real 2026 format and FIFA's
  **official knockout bracket** — 12 groups, top-2 + 8 best third-placed teams into the Round of
  32 — to produce advancement %, title odds with **95% confidence intervals**, a **bookmaker
  market** column, and a **model–market consensus**.
- **Projects the Golden Boot** — each player's expected tournament goals.
- **Reports its own measured accuracy** via a backtest on the 2022 World Cup.

## How it works

| Component | What it does |
|-----------|--------------|
| **Dixon-Coles** (`src/dixon_coles.py`) | Bivariate-Poisson goal model. Estimates each team's attack/defence strength, home advantage, and the low-score-dependence parameter ρ, with **exponential time-decay** (7-year half-life) so recent matches — including the live 2026 results — weigh most. Produces a full scoreline probability matrix. |
| **Elo** (`src/elo.py`) | World-football Elo over all history since 2010, used as a complementary outcome signal and sanity check. |
| **Cross-confederation calibration** (`src/confederations.py`) | **Native model.** Elo can't normalise across largely-closed pools, so it under-rates UEFA and over-rates AFC/CONCACAF/OFC. We measure each confederation's intercontinental over/under-performance vs Elo from our own data and correct it. **Validated: +2.9pp out-of-sample WC accuracy** (53.6% → 56.8%). Estimated from pre-match data only, so it's in the backtest too. |
| **Squad market value** (`src/config.py`, `calibration.py`) | **Native model.** Total Transfermarkt squad value per team — *fundamental data*, not a prediction. It's the squad-quality/ageing signal results can't show (Argentina rank #7 by value despite a #1 results record). Blended into the ratings (`--squad-weight`, default 0.5), this drops Argentina ~21% → ~13% **using data, no prediction engines**. |
| **Expert-prior calibration** (`src/calibration.py`) | *Optional* (`--expert-prior`, default 0). Folds external-engine **odds** (Opta + market) directly into the native ratings. Off by default — the three-tier output shows the engines separately instead. |
| **Predictor** (`src/predict.py`) | Scoreline comes from a **lambda blend** — the Dixon-Coles goal supremacy nudged toward Elo's implied supremacy (better point predictions). Outcome probabilities are an **ensemble** of that blended distribution and a probability-level Elo blend (keeps calibration). All parameters were tuned by cross-validation over four past World Cups — see `experiments/cv.py`. |
| **Simulator** (`src/simulate.py`) | Monte Carlo: samples every remaining match, builds group tables (points → GD → GF → Elo tie-break), picks the 8 best thirds (FIFA slot constraints solved by bipartite matching), seeds the Round of 32, and plays out FIFA's **official** bracket. Knockout draws resolve via an Elo-tilted-but-regressed extra-time/penalty model. |
| **Altitude / environment** (`src/environment.py`) | Adjusts goal supremacy for venue altitude — validated on our data (+0.13 goals/1000m of away-team altitude shock, t=3.7). Matters for 2026: Mexico City (2,240m) and Guadalajara (1,566m) favour acclimatised sides (Mexico, Ecuador, Colombia) over sea-level teams. Heat (ambiguous net goal effect) and rest/travel (insignificant in studies) were researched and deliberately excluded. |
| **Golden Boot** (`src/scorers.py`) | Projects each player's expected goals = goals so far + share of their team's expected future goals, then **blends with bookmaker top-scorer odds** (which capture penalty duties, role and pedigree the goals-only model can't see). |

### Three tiers (what `simulate` shows)

`simulate` prints champion odds three ways, side by side:

1. **Native** — *our model only*: Dixon-Coles + Elo + lambda/ensemble + cross-confederation
   calibration + squad market value. **No prediction engines.** This now puts Spain #1 (~14%) and
   Argentina ~13% on its own.
2. **Engines** — the external prediction consensus (Opta supercomputer + bookmaker odds; EA Sports
   and Nate Silver's model also pick Spain).
3. **Ensemble** — a 50/50 cross-validation of Native and Engines, renormalised over 48 teams.

This directly answers the goal: a model good **on its own** that can be **potentiated** by others.

### How the Argentina gap was actually closed (from our data)

The model used to rate Argentina #1 (~21%) vs experts' ~4th. We diagnosed it instead of patching:

- **Confederation bias** (fixable from results): Elo under-rated UEFA and over-rated
  AFC/CONCACAF/OFC. Correcting it lifted Spain/France/England and improved accuracy +2.9pp.
- **Squad ageing** (needs data results can't show): adding **squad market value** — fundamental
  data, not odds — lets the native model see that Argentina's squad ranks only #7. This pulls them
  to ~13% **without any prediction engine**.

The **Golden Boot** blends `40% model / 60% bookmaker odds`, so designated penalty-takers
(Mbappé, Kane) rank correctly while live form (Havertz, Balogun) still lifts players.

**Data**:
- [martj42/international_results](https://github.com/martj42/international_results) — ~49k
  international matches since 1872 + 47k goalscorer records, **including the live 2026 fixtures/scores**. No API key.
- **Squad market values** (Transfermarkt, all 48 teams) — fundamental squad-quality data for the native model.
- **External engine snapshots** (Opta supercomputer, bookmaker odds, Golden Boot odds) in `src/config.py` — for the Engines/Ensemble tiers.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```bash
.venv/bin/python main.py refresh      # pull the latest results + goals (run before everything)
.venv/bin/python main.py predict      # scoreline + W/D/L for upcoming matches -> data/predictions_upcoming.csv
.venv/bin/python main.py predict --compare-forebet   # show Forebet's score next to ours + agreement %
.venv/bin/python main.py simulate     # title odds + CI + market + consensus    -> data/simulation_odds.csv
.venv/bin/python main.py golden-boot  # projected top scorers                   -> data/golden_boot.csv
.venv/bin/python main.py backtest     # measured accuracy on the 2022 World Cup
```

Useful flags: `--sims 100000` (simulate/golden-boot, default 50k), `--top 30`,
`--squad-weight 0` … `1` (squad-value influence in the native model; default 0.5),
`--expert-prior 0.5` (optionally fold engine odds into the native model; default 0), `backtest --verbose`.

**Keeping it current**: as matches finish, the source dataset updates within ~a day. Run
`refresh` then `predict`/`simulate` again — played matches lock to their real scores and the
remaining predictions shift accordingly.

## Measured accuracy (out-of-sample backtests)

`main.py backtest` trains only on pre-tournament data and predicts each past World Cup. Averaged
over 2014/2018/2022 (192 matches):

| Metric | Pure-results model | **+ confederation + goal calibration** |
|---|---|---|
| Outcome accuracy (W/D/L) | 53.6% | **57.3%** |
| Brier score | 0.598 | **0.580** |
| Log-loss (random = 1.099) | 1.006 | **0.982** |

Goal level is calibrated to the actual World Cup scoring rate (~2.67 goals/match) via an
unregularised intercept plus a tournament scale factor — **corroborated independently by
[Forebet](https://www.forebet.com/en/predictions-world/world-cup)'s score engine**, which picks the
same winners as us with slightly higher scorelines. `predict` shows each match's most-likely
scoreline *plus the top-3 distribution* (e.g. `1-0 16% · 2-0 12% · 1-1 12%`), since no single
scoreline in football is more than ~15–20% likely.

The confederation calibration is the biggest single lift the data allows, and it's measured
out-of-sample (every World Cup match is intercontinental). Exact-score sits ~12–16% depending on
configuration. The **expert prior is deliberately excluded from the backtest** — there are no
historical odds for 2014–2022, and it's a forecast-time input, so including it would be cheating.

This is the **honest** ceiling for football prediction. Exact scorelines are inherently hard;
the model gives the most-likely score plus a calibrated probability distribution, not a certainty.
(The lambda-blend + ensemble upgrade improved exact-score from ~14.8% to ~16% and held calibration,
versus the earlier probability-only blend — measured in `experiments/cv.py`.)

## Known limitations

- **Best-thirds assignment** is solved by constraint matching (valid, FIFA-consistent) rather than
  FIFA's full 495-scenario lookup table — structurally correct, may differ in rare edge cases.
- **Golden Boot** projections are *expected* goals per player; the actual winner typically exceeds
  their expectation (they're the max of many random variables), so read the **ranking**, not the
  absolute totals. Squad membership is inferred from recent scoring, not official team sheets.
- **Market odds** are a fixed mid-June-2026 snapshot (incl. bookmaker margin); refresh them in
  `src/config.py` if you want the consensus to track the live board.
- No injuries / lineups / weather. Recent-form time-decay is the proxy for current condition.

The Round-of-32 matchups, the R16→Final tree, and the group composition are all the **official**
2026 structures.
