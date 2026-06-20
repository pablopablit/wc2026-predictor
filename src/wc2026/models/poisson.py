"""The primary model: a Bayesian Poisson goal model (PyMC).

Design (mirrors playmobil/worldcup-forecast, adapted to this feature-matrix
codebase). Each side's goals are Poisson with a log-rate that is **linear in the
leakage-safe features** — Elo encodes team strength (shrinking new teams toward
1500), and World Bank structural covariates (log-population, GDP as an inverted-U)
act as the priors for data-scarce teams:

    log λ_score = b0 + b_home·is_home
                     + b_att·own_elo  + b_def·opp_elo
                     + b_pop·own_logpop + b_gdp·own_gdp + b_gdp2·own_gdp²

Each match contributes two observations (home goals, away goals) with shared
attack/defence coefficients (a symmetry that halves the parameters). **Priors are
set by reasoning, not fit to the ~36 historical World Cups.** Inference is
variational (ADVI) by default — fast enough for the walk-forward backtest — with
full NUTS available via ``method="nuts"``.

Prediction unifies scoreline and outcome: posterior-mean rates give a **Poisson
score grid** (``score_grid``); summing its cells yields W/D/L probabilities and
the modal cell is the most-likely scoreline. A learned **temperature** calibrates
the probabilities (the brief's calibration step).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from wc2026 import config
from wc2026.models.base import Predictor

logger = logging.getLogger(__name__)

#: Structural feature columns the model standardises and consumes.
_STRUCT = ("log_pop_home", "log_pop_away", "gdp_home", "gdp_away")


def _poisson_grid(lam_home: np.ndarray, lam_away: np.ndarray, max_goals: int):
    """Return (P_home_win, P_draw, P_away_win, grids) for arrays of rates.

    ``grids`` has shape (n, G+1, G+1) with grids[n, i, j] = P(home i, away j).
    """
    goals = np.arange(max_goals + 1)
    pmf_h = poisson.pmf(goals[None, :], lam_home[:, None])  # (n, G+1)
    pmf_a = poisson.pmf(goals[None, :], lam_away[:, None])
    grids = pmf_h[:, :, None] * pmf_a[:, None, :]  # (n, G+1, G+1)
    # grids[n, i, j] = P(home=i, away=j); home wins when i > j (lower triangle).
    home_win_mask = np.tri(max_goals + 1, max_goals + 1, -1)  # 1 where row i > col j
    p_home = (grids * home_win_mask).sum(axis=(1, 2))
    p_away = (grids * home_win_mask.T).sum(axis=(1, 2))
    p_draw = np.einsum("nii->n", grids)
    total = p_home + p_draw + p_away  # ~1 minus the tail beyond max_goals
    return p_home / total, p_draw / total, p_away / total, grids


class BayesianPoissonPredictor(Predictor):
    """Bayesian Poisson regression on Elo + structural features (PyMC)."""

    model_type = "bayesian_poisson"
    requires_goals = True

    def __init__(self, max_goals: int = config.SCORE_GRID_MAX_GOALS) -> None:
        self.feature_names = ("home_advantage", "elo_home", "elo_away", *_STRUCT)
        self.max_goals = max_goals
        self._coef: dict[str, float] = {}
        self._struct_mean: dict[str, float] = {}
        self._struct_std: dict[str, float] = {}
        self._temperature = 1.0

    # -- feature prep ------------------------------------------------------- #
    def _scale_struct(self, X: pd.DataFrame, col: str) -> np.ndarray:
        v = X[col].to_numpy(dtype=float)
        mean, std = self._struct_mean[col], self._struct_std[col]
        v = np.where(np.isnan(v), mean, v)  # impute missing with the train mean
        return (v - mean) / std

    def _design(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Build per-side covariate arrays (own/opp) for home and away goals."""
        is_home = X["home_advantage"].to_numpy(dtype=float)
        elo_home = (X["elo_home"].to_numpy(dtype=float) - 1500.0) / 100.0
        elo_away = (X["elo_away"].to_numpy(dtype=float) - 1500.0) / 100.0
        lp_home = self._scale_struct(X, "log_pop_home")
        lp_away = self._scale_struct(X, "log_pop_away")
        gd_home = self._scale_struct(X, "gdp_home")
        gd_away = self._scale_struct(X, "gdp_away")
        return {
            "is_home": is_home,
            "elo_home": elo_home,
            "elo_away": elo_away,
            "lp_home": lp_home,
            "lp_away": lp_away,
            "gd_home": gd_home,
            "gd_away": gd_away,
        }

    # -- fit ---------------------------------------------------------------- #
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        goals: pd.DataFrame | None = None,
        method: str = "advi",
        draws: int = 400,
        seed: int = config.RANDOM_SEED,
    ) -> BayesianPoissonPredictor:
        if goals is None:
            raise ValueError("BayesianPoissonPredictor.fit requires goals=(home,away).")
        import pymc as pm

        # Fit structural scalers on the training fold (impute/scale reuse them).
        for col in _STRUCT:
            v = X[col].to_numpy(dtype=float)
            v = v[~np.isnan(v)]
            self._struct_mean[col] = float(v.mean()) if len(v) else 0.0
            self._struct_std[col] = float(v.std()) if len(v) and v.std() > 0 else 1.0

        d = self._design(X)
        gh = goals.iloc[:, 0].to_numpy(dtype=int)
        ga = goals.iloc[:, 1].to_numpy(dtype=int)

        # Stack the two per-match observations (home-goals, away-goals).
        is_home = np.concatenate([d["is_home"], np.zeros_like(d["is_home"])])
        own_elo = np.concatenate([d["elo_home"], d["elo_away"]])
        opp_elo = np.concatenate([d["elo_away"], d["elo_home"]])
        own_lp = np.concatenate([d["lp_home"], d["lp_away"]])
        own_gd = np.concatenate([d["gd_home"], d["gd_away"]])
        target = np.concatenate([gh, ga])

        with pm.Model():
            b0 = pm.Normal("b0", mu=np.log(max(target.mean(), 0.1)), sigma=0.5)
            b_home = pm.Normal("b_home", mu=0.25, sigma=0.2)
            b_att = pm.Normal("b_att", mu=0.3, sigma=0.3)
            b_def = pm.Normal("b_def", mu=-0.3, sigma=0.3)
            b_pop = pm.Normal("b_pop", mu=0.0, sigma=0.2)
            b_gd = pm.Normal("b_gd", mu=0.0, sigma=0.2)
            b_gd2 = pm.Normal("b_gd2", mu=0.0, sigma=0.2)
            eta = (
                b0
                + b_home * is_home
                + b_att * own_elo
                + b_def * opp_elo
                + b_pop * own_lp
                + b_gd * own_gd
                + b_gd2 * own_gd**2
            )
            pm.Poisson("goals", mu=pm.math.exp(eta), observed=target)

            if method == "nuts":
                idata = pm.sample(
                    draws=draws, tune=draws, chains=2, cores=1,
                    target_accept=0.9, random_seed=seed, progressbar=False,
                )
            elif method == "map":
                point = pm.find_MAP(progressbar=False)
                self._coef = {k: float(np.asarray(point[k])) for k in
                              ("b0", "b_home", "b_att", "b_def", "b_pop", "b_gd", "b_gd2")}
                self._fit_temperature(X, goals)
                return self
            else:  # advi (default)
                approx = pm.fit(20000, method="advi", random_seed=seed, progressbar=False)
                idata = approx.sample(draws)

        post = idata.posterior
        self._coef = {
            k: float(post[k].mean()) for k in
            ("b0", "b_home", "b_att", "b_def", "b_pop", "b_gd", "b_gd2")
        }
        logger.info("Fitted %s (%s) on %d matches: %s", self.model_type, method, len(X), self._coef)
        self._fit_temperature(X, goals)
        return self

    # -- rates / grid / probs ---------------------------------------------- #
    def _rates(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        d = self._design(X)
        c = self._coef
        eta_h = (
            c["b0"] + c["b_home"] * d["is_home"] + c["b_att"] * d["elo_home"]
            + c["b_def"] * d["elo_away"] + c["b_pop"] * d["lp_home"]
            + c["b_gd"] * d["gd_home"] + c["b_gd2"] * d["gd_home"] ** 2
        )
        eta_a = (
            c["b0"] + c["b_att"] * d["elo_away"] + c["b_def"] * d["elo_home"]
            + c["b_pop"] * d["lp_away"] + c["b_gd"] * d["gd_away"]
            + c["b_gd2"] * d["gd_away"] ** 2
        )
        return np.exp(eta_h), np.exp(eta_a)

    def score_grid(self, X: pd.DataFrame):
        """Return (grids, most_likely_scores) — the scoreline hook shared by the
        simulator and the CLI. ``grids[n, i, j]`` = P(home i, away j)."""
        lam_h, lam_a = self._rates(X)
        _, _, _, grids = _poisson_grid(lam_h, lam_a, self.max_goals)
        flat = grids.reshape(len(grids), -1).argmax(axis=1)
        gi, gj = np.unravel_index(flat, (self.max_goals + 1, self.max_goals + 1))
        scores = list(zip(gi.tolist(), gj.tolist(), strict=True))
        return grids, scores

    def _raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        lam_h, lam_a = self._rates(X)
        p_h, p_d, p_a, _ = _poisson_grid(lam_h, lam_a, self.max_goals)
        return np.column_stack([p_h, p_d, p_a])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return _apply_temperature(self._raw_proba(X), self._temperature)

    # -- calibration -------------------------------------------------------- #
    def _fit_temperature(self, X: pd.DataFrame, goals: pd.DataFrame) -> None:
        raw = self._raw_proba(X)
        gh = goals.iloc[:, 0].to_numpy()
        ga = goals.iloc[:, 1].to_numpy()
        y = np.where(gh > ga, 0, np.where(gh == ga, 1, 2))

        def neg_ll(t: float) -> float:
            p = _apply_temperature(raw, t)
            return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))

        res = minimize_scalar(neg_ll, bounds=(0.5, 3.0), method="bounded")
        self._temperature = float(res.x)
        logger.info("Calibrated temperature T=%.3f", self._temperature)


def _apply_temperature(proba: np.ndarray, t: float) -> np.ndarray:
    """Temperature-scale probabilities via their logits; T>1 softens, T<1 sharpens."""
    logits = np.log(np.clip(proba, 1e-12, None)) / t
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)
