"""Dixon-Coles bivariate-Poisson goal model with exponential time decay.

Estimates per-team attack/defence strengths, a global home advantage, and the
low-score dependence parameter rho. Recent matches (esp. the live 2026 results)
are up-weighted via time decay so current form drives predictions.

Reference: Dixon & Coles (1997), "Modelling Association Football Scores".
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


def _decay_weights(dates, ref_date, half_life_days):
    days = (ref_date - dates).dt.days.to_numpy(dtype=float)
    days = np.clip(days, 0, None)
    return 0.5 ** (days / half_life_days)


def _importance_weight(tournaments):
    w = np.ones(len(tournaments))
    lower = tournaments.str.lower()
    w[lower.str.contains("friendly")] = 0.5
    return w


def _tau(h, a, lam, mu, rho):
    """Dixon-Coles low-score correction for the (0/1)x(0/1) cells."""
    out = np.ones_like(lam)
    out = np.where((h == 0) & (a == 0), 1.0 - lam * mu * rho, out)
    out = np.where((h == 0) & (a == 1), 1.0 + lam * rho, out)
    out = np.where((h == 1) & (a == 0), 1.0 + mu * rho, out)
    out = np.where((h == 1) & (a == 1), 1.0 - rho, out)
    return out


class DixonColes:
    def __init__(self, half_life_days=2555, ridge=0.005):
        self.half_life_days = half_life_days
        self.ridge = ridge
        self.teams = []
        self.idx = {}
        self.attack = {}
        self.defence = {}
        self.home_adv = 0.0
        self.rho = 0.0

    def fit(self, matches):
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        self.teams = teams
        self.idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        hi = matches["home_team"].map(self.idx).to_numpy()
        ai = matches["away_team"].map(self.idx).to_numpy()
        hg = matches["home_score"].to_numpy(dtype=float)
        ag = matches["away_score"].to_numpy(dtype=float)
        neutral = matches["neutral"].to_numpy()

        ref = matches["date"].max()
        w = _decay_weights(matches["date"], ref, self.half_life_days)
        w = w * _importance_weight(matches["tournament"].astype(str))

        h00 = (hg == 0); h11 = (hg == 1)
        a00 = (ag == 0); a11 = (ag == 1)

        # Param vector: [attack(n), defence(n), home_adv, rho, intercept].
        # The intercept is the log baseline scoring rate; it is NOT regularised,
        # so the overall goal level fits freely (otherwise ridge shrinks
        # attack/defence to 0 and forces ~exp(0)=1 goal/side, ~24% too low).
        def unpack(p):
            return p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1], p[2 * n + 2]

        def neg_ll(p):
            atk, dfc, home_adv, rho, base = unpack(p)
            log_lam = base + atk[hi] - dfc[ai] + np.where(neutral, 0.0, home_adv)
            log_mu = base + atk[ai] - dfc[hi]
            lam = np.exp(log_lam)
            mu = np.exp(log_mu)
            # log Poisson pmf for home and away goals
            ll = (hg * log_lam - lam) + (ag * log_mu - mu)
            # tau correction (only the 4 low-score cells differ from 1)
            tau = np.ones_like(lam)
            tau = np.where(h00 & a00, 1.0 - lam * mu * rho, tau)
            tau = np.where(h00 & a11, 1.0 + lam * rho, tau)
            tau = np.where(h11 & a00, 1.0 + mu * rho, tau)
            tau = np.where(h11 & a11, 1.0 - rho, tau)
            tau = np.clip(tau, 1e-9, None)
            ll = ll + np.log(tau)
            total = np.sum(w * ll)
            # ridge keeps attack/defence identifiable and shrinks sparse teams
            # toward the mean (intercept excluded so the goal level is unbiased)
            penalty = self.ridge * (np.sum(atk ** 2) + np.sum(dfc ** 2))
            return -(total) + penalty

        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.05], [0.3]])
        bounds = [(-3, 3)] * (2 * n) + [(-1, 1), (-0.2, 0.2), (-1.0, 1.5)]
        res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500})
        atk, dfc, home_adv, rho, base = unpack(res.x)
        # center attack for interpretability (level shift absorbed by intercept)
        atk = atk - atk.mean()
        self.attack = {t: atk[i] for t, i in self.idx.items()}
        self.defence = {t: dfc[i] for t, i in self.idx.items()}
        self.home_adv = float(home_adv)
        self.rho = float(rho)
        self.intercept = float(base)
        self._fit_result = res
        return self

    def expected_goals(self, home, away, neutral=True):
        """Return (lambda_home, mu_away) expected-goal rates."""
        adv = 0.0 if neutral else self.home_adv
        lam = np.exp(self.intercept + self.attack[home] - self.defence[away] + adv)
        mu = np.exp(self.intercept + self.attack[away] - self.defence[home])
        return float(lam), float(mu)

    def score_matrix(self, home, away, neutral=True, max_goals=10):
        """Full P(home=x, away=y) matrix of shape (max_goals+1, max_goals+1)."""
        lam, mu = self.expected_goals(home, away, neutral)
        gh = np.arange(max_goals + 1)
        ph = poisson.pmf(gh, lam)
        pa = poisson.pmf(gh, mu)
        mat = np.outer(ph, pa)
        # apply tau to the four low-score cells
        mat[0, 0] *= 1.0 - lam * mu * self.rho
        mat[0, 1] *= 1.0 + lam * self.rho
        mat[1, 0] *= 1.0 + mu * self.rho
        mat[1, 1] *= 1.0 - self.rho
        mat = np.clip(mat, 0, None)
        mat /= mat.sum()
        return mat
