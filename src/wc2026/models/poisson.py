"""The primary model: a hierarchical Bayesian Poisson goal model (PyMC).

Design (mirrors playmobil/worldcup-forecast, adapted to this codebase):

* Each team has partially-pooled latent **attack** and **defence** strengths;
  a global **home-advantage** term is added for the home side and suppressed on
  neutral ground.
* Expected goals are Poisson: ``log λ_home = μ + atk_home − def_away + home_adv``
  and symmetrically for the away side.
* **Priors are set by reasoning, not fit to the ~36 historical World Cups.**
  Data-scarce teams shrink toward an Elo-implied prior (and, in v1, toward
  structural socio-economic priors — log-population and GDP — from the World
  Bank); data-rich teams are driven by their match record via partial pooling.

Prediction unifies scoreline and outcome (replaces the brief's naive Poisson
placeholder): the posterior produces a **score grid** (a Poisson goal matrix);
summing its cells gives calibrated W/D/L probabilities, and the modal cell is the
most-likely scoreline. Probabilities are temperature-calibrated with a neutral-
site draw adjustment and judged on log-loss / Brier.

Performance note: full NUTS sampling is reserved for the final fit; the
walk-forward backtest uses a fast MAP / variational fit so ``make evaluate``
stays usable. All sampling is seeded from ``config.RANDOM_SEED``.
"""

from __future__ import annotations

import logging

from wc2026.models.base import Predictor

logger = logging.getLogger(__name__)


class BayesianPoissonPredictor(Predictor):
    """Hierarchical Bayesian Poisson goal model. Implemented in Phase 5.

    Implements the :class:`~wc2026.models.base.Predictor` interface so it is a
    drop-in for the baseline in the simulator and CLI. Exposes a ``score_grid``
    hook so the tournament simulator and the scoreline output share one source of
    truth for goal distributions.
    """

    model_type = "bayesian_poisson"
