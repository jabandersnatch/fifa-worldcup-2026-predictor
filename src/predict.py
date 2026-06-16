"""Turn the fitted models into per-match predictions.

The scoreline (and expected goals) come from a **lambda blend**: the Dixon-Coles
goal expectation has its home/away supremacy nudged toward Elo's implied
supremacy, which improves point predictions (cross-validated exact-score 14.8% →
16.0% over the 2010–2022 World Cups). Outcome probabilities are an **ensemble**
of that blended scoreline distribution and a probability-level Elo blend, which
keeps the calibration of the probability blend. Parameters were tuned on those
four World Cups; see experiments/cv.py.
"""
import numpy as np
from scipy.stats import poisson

from . import elo
from . import environment


class Predictor:
    HOME_ADV_ELO = 65.0
    SHOOTOUT_REGRESSION = 0.65   # Elo edge that survives extra time + penalties

    def __init__(self, dc_model, elo_ratings, elo_weight=0.70, beta=0.005,
                 draw_base=0.28, ensemble_mix=0.5, goal_scale=1.13):
        self.dc = dc_model
        self.elo = elo_ratings
        self.elo_weight = elo_weight       # weight on the Elo signal in both blends
        self.beta = beta                   # Elo rating-diff -> goal supremacy
        self.draw_base = draw_base         # max draw probability in the Elo 3-way
        self.mix = ensemble_mix            # weight on the lambda-blend in the ensemble
        # World Cup matches average ~2.67 goals vs ~2.35 in the broad training
        # mix; this calibrates the goal level to tournament scoring (corroborated
        # by the actual WC average and by Forebet's independent score engine).
        self.goal_scale = goal_scale

    # --- goal expectation -------------------------------------------------
    def blended_lambdas(self, home, away, neutral=True, city=None):
        """Dixon-Coles expected goals with the supremacy blended toward Elo, then
        an altitude supremacy shift if the venue (city) is at elevation."""
        lam, mu = self.dc.expected_goals(home, away, neutral)
        w = self.elo_weight
        if w > 0 and home in self.elo and away in self.elo:
            adv = 0.0 if neutral else self.HOME_ADV_ELO
            dr = (self.elo[home] + adv) - self.elo[away]
            sup_elo = self.beta * dr
            total = lam + mu
            sup = (1 - w) * (lam - mu) + w * sup_elo
            lam = max((total + sup) / 2, 0.05)
            mu = max((total - sup) / 2, 0.05)
        lam *= self.goal_scale
        mu *= self.goal_scale
        if city is not None:
            adj = environment.altitude_supremacy(home, away, city, neutral)
            if adj:
                lam = max(lam + adj / 2, 0.05)
                mu = max(mu - adj / 2, 0.05)
        return lam, mu

    # alias kept for callers that want expected goals
    def expected_goals(self, home, away, neutral=True, city=None):
        return self.blended_lambdas(home, away, neutral, city)

    def _matrix(self, lam, mu, max_goals=10):
        gh = np.arange(max_goals + 1)
        mat = np.outer(poisson.pmf(gh, lam), poisson.pmf(gh, mu))
        rho = self.dc.rho
        mat[0, 0] *= 1 - lam * mu * rho
        mat[0, 1] *= 1 + lam * rho
        mat[1, 0] *= 1 + mu * rho
        mat[1, 1] *= 1 - rho
        mat = np.clip(mat, 0, None)
        return mat / mat.sum()

    def score_matrix(self, home, away, neutral=True, max_goals=10, city=None):
        """Blended P(home=x, away=y) matrix used for predictions and sampling."""
        lam, mu = self.blended_lambdas(home, away, neutral, city)
        return self._matrix(lam, mu, max_goals)

    @staticmethod
    def _outcome(mat):
        return float(np.tril(mat, -1).sum()), float(np.trace(mat)), float(np.triu(mat, 1).sum())

    # --- full prediction --------------------------------------------------
    def predict(self, home, away, neutral=True, max_goals=10, city=None):
        lam, mu = self.blended_lambdas(home, away, neutral, city)
        mat = self._matrix(lam, mu, max_goals)                 # point-prediction matrix
        ph_l, pd_l, pa_l = self._outcome(mat)

        # probability-blend branch: raw DC outcome probs blended with Elo 3-way
        lam0, mu0 = self.dc.expected_goals(home, away, neutral)
        adj0 = environment.altitude_supremacy(home, away, city, neutral) if city else 0.0
        mat0 = self._matrix(max(lam0 * self.goal_scale + adj0 / 2, 0.05),
                            max(mu0 * self.goal_scale - adj0 / 2, 0.05), max_goals)
        ph_p, pd_p, pa_p = self._outcome(mat0)
        w = self.elo_weight
        if w > 0 and home in self.elo and away in self.elo:
            eh, ed, ea = elo.outcome_probs(self.elo[home], self.elo[away], neutral,
                                           draw_base=self.draw_base)
            ph_p, pd_p, pa_p = (1 - w) * ph_p + w * eh, (1 - w) * pd_p + w * ed, (1 - w) * pa_p + w * ea

        a = self.mix
        ph, pd_, pa = a * ph_l + (1 - a) * ph_p, a * pd_l + (1 - a) * pd_p, a * pa_l + (1 - a) * pa_p
        s = ph + pd_ + pa
        ph, pd_, pa = ph / s, pd_ / s, pa / s

        i, j = np.unravel_index(np.argmax(mat), mat.shape)
        # the N most likely exact scorelines and their probabilities
        flat = mat.ravel()
        order = np.argsort(flat)[::-1][:5]
        ncol = mat.shape[1]
        top_scores = [(int(k // ncol), int(k % ncol), float(flat[k])) for k in order]
        return {
            "home": home, "away": away, "neutral": neutral, "matrix": mat,
            "exp_home": lam, "exp_away": mu,
            "ml_home": int(i), "ml_away": int(j), "ml_prob": float(mat[i, j]),
            "top_scores": top_scores,
            "p_home": ph, "p_draw": pd_, "p_away": pa,
        }

    def knockout_win_prob(self, home, away, neutral=True, city=None):
        """P(home advances): regulation win plus the draw mass resolved by an
        Elo-tilted-but-regressed extra-time/penalty model."""
        pred = self.predict(home, away, neutral, city=city)
        if home in self.elo and away in self.elo:
            eh, _, ea = elo.outcome_probs(self.elo[home], self.elo[away], neutral)
            two_way = eh / (eh + ea) if (eh + ea) else 0.5
            shootout_edge = 0.5 + (two_way - 0.5) * self.SHOOTOUT_REGRESSION
        else:
            shootout_edge = 0.5
        return pred["p_home"] + pred["p_draw"] * shootout_edge
