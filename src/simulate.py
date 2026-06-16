"""Monte Carlo simulation of the remainder of the 2026 World Cup.

Group stage and Round-of-32 *entry* follow the exact 2026 rules (top 2 per group
+ 8 best third-placed teams, with FIFA's third-place slot constraints solved by
bipartite matching). The R16->Final tree uses a balanced layout (see config).
"""
import numpy as np
from collections import defaultdict

from . import config


def _third_slots():
    """List of (match_no, allowed_group_set) for the eight third-place slots."""
    slots = []
    for m, (a, b) in config.R32.items():
        for code in (a, b):
            if code.startswith("3"):
                slots.append((m, set(code[1:])))
    return slots


def _match_thirds(qualifying_groups, slots):
    """Assign 8 qualifying group letters to the 8 third-place slots respecting
    each slot's allowed set (Kuhn's bipartite matching). Returns slot_match_no ->
    group_letter, or None if no perfect matching exists."""
    groups = list(qualifying_groups)
    slot_to_group = {}
    group_used = {}

    def try_assign(g, visited):
        for (m, allowed) in slots:
            if g in allowed and m not in visited:
                visited.add(m)
                if m not in slot_to_group or try_assign(slot_to_group[m], visited):
                    slot_to_group[m] = g
                    group_used[g] = m
                    return True
        return False

    for g in groups:
        if not try_assign(g, set()):
            return None
    return {m: g for g, m in group_used.items()}


class Tournament:
    def __init__(self, wc_matches, predictor, seed=12345):
        self.predictor = predictor
        self.elo = predictor.elo
        self.rng = np.random.default_rng(seed)
        self.slots = _third_slots()
        self._ko_cache = {}

        # Split played vs remaining group matches; record fixed results.
        self.played = []      # (home, away, hg, ag)
        self.remaining = []   # (home, away, neutral)
        for r in wc_matches.itertuples(index=False):
            neutral = bool(r.neutral)
            if r.home_team in config.HOSTS and r.country == r.home_team:
                neutral = False
            if r.played:
                self.played.append((r.home_team, r.away_team,
                                    int(r.home_score), int(r.away_score)))
            else:
                self.remaining.append((r.home_team, r.away_team, neutral))

        # Pre-sample scorelines for every remaining group fixture (vectorized).
        self._presample = {}

    def _prepare_samples(self, n_sims):
        max_goals = 10
        for (h, a, neutral) in self.remaining:
            mat = self.predictor.score_matrix(h, a, neutral, max_goals).ravel()
            cum = np.cumsum(mat)
            draws = self.rng.random(n_sims)
            idx = np.searchsorted(cum, draws)
            idx = np.clip(idx, 0, len(mat) - 1)
            hg, ag = np.divmod(idx, max_goals + 1)
            self._presample[(h, a)] = (hg, ag)

    def _ko_prob(self, home, away):
        key = (home, away)
        if key not in self._ko_cache:
            self._ko_cache[key] = self.predictor.knockout_win_prob(home, away, neutral=True)
        return self._ko_cache[key]

    def _group_standings(self, sim_results):
        """Return dict group_letter -> ordered list of teams (best first), using
        played results + this sim's sampled results."""
        stats = {t: [0, 0, 0] for t in config.WC_TEAMS}  # pts, gd, gf

        def record(h, a, hg, ag):
            if hg > ag:
                stats[h][0] += 3
            elif hg < ag:
                stats[a][0] += 3
            else:
                stats[h][0] += 1; stats[a][0] += 1
            stats[h][1] += hg - ag; stats[a][1] += ag - hg
            stats[h][2] += hg; stats[a][2] += ag

        for (h, a, hg, ag) in self.played:
            record(h, a, hg, ag)
        for (h, a, hg, ag) in sim_results:
            record(h, a, hg, ag)

        # Tie-break: points, GD, GF, then Elo strength (proxy for FIFA's
        # head-to-head / drawing-of-lots), then a tiny random nudge.
        order = {}
        for letter, teams in config.GROUPS.items():
            ranked = sorted(teams, key=lambda t: (stats[t][0], stats[t][1], stats[t][2],
                                                   self.elo.get(t, 1500), self.rng.random()),
                            reverse=True)
            order[letter] = (ranked, {t: stats[t] for t in teams})
        return order

    def run(self, n_sims=20000):
        self._prepare_samples(n_sims)
        counters = {t: defaultdict(int) for t in config.WC_TEAMS}
        rem = self.remaining

        for s in range(n_sims):
            sim_results = [(h, a, self._presample[(h, a)][0][s], self._presample[(h, a)][1][s])
                           for (h, a, _) in rem]
            order = self._group_standings(sim_results)

            winners = {L: order[L][0][0] for L in config.GROUPS}
            runners = {L: order[L][0][1] for L in config.GROUPS}
            thirds = {L: order[L][0][2] for L in config.GROUPS}
            stats = {L: order[L][1] for L in config.GROUPS}

            # rank the 12 third-placed teams; top 8 qualify
            third_rank = sorted(config.GROUPS,
                                key=lambda L: (stats[L][thirds[L]][0], stats[L][thirds[L]][1],
                                               stats[L][thirds[L]][2],
                                               self.elo.get(thirds[L], 1500), self.rng.random()),
                                reverse=True)
            qual_groups = third_rank[:8]
            for L in config.GROUPS:
                for t in order[L][0][:2]:
                    counters[t]["R32"] += 1
            for L in qual_groups:
                counters[thirds[L]]["R32"] += 1

            slot_assign = _match_thirds(qual_groups, self.slots)
            if slot_assign is None:
                slot_assign = {m: g for m, g in zip([s[0] for s in self.slots], qual_groups)}

            def resolve(code):
                kind, letter = code[0], code[1:]
                if kind == "1":
                    return winners[letter]
                if kind == "2":
                    return runners[letter]
                return thirds[slot_assign[m]]  # third place -> group via this slot

            # Round of 32
            win = {}
            for m, (a, b) in config.R32.items():
                home, away = resolve(a), resolve(b)
                p = self._ko_prob(home, away)
                win[m] = home if self.rng.random() < p else away
                counters[win[m]]["R16"] += 1

            # R16 -> Final
            stage_label = {(89, 96): "QF", (97, 100): "SF", (101, 102): "Final"}
            for m in sorted(config.BRACKET_TREE):
                f1, f2 = config.BRACKET_TREE[m]
                home, away = win[f1], win[f2]
                p = self._ko_prob(home, away)
                w = home if self.rng.random() < p else away
                win[m] = w
                if 89 <= m <= 96:
                    counters[w]["QF"] += 1
                elif 97 <= m <= 100:
                    counters[w]["SF"] += 1
                elif 101 <= m <= 102:
                    counters[w]["Final"] += 1
                elif m == 104:
                    counters[w]["Champion"] += 1

        # normalize to probabilities; keep the raw champion count for a CI
        self.n_sims = n_sims
        out = {}
        for t in config.WC_TEAMS:
            out[t] = {k: counters[t][k] / n_sims for k in
                      ("R32", "R16", "QF", "SF", "Final", "Champion")}
            out[t]["champion_count"] = counters[t]["Champion"]
        return out
