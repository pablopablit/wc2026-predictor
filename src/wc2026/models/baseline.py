"""Rule-based / logistic baseline — the bar the real model must clear.

Predicts win/draw/away purely from **Elo difference + home advantage** via a
multinomial logistic regression. Cheap, interpretable, and a genuine benchmark:
if the Bayesian Poisson model cannot beat this on log-loss / Brier, it is not
earning its complexity.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wc2026 import config
from wc2026.models.base import Predictor

logger = logging.getLogger(__name__)

#: The only inputs the baseline is allowed to use.
BASELINE_FEATURES: tuple[str, ...] = ("elo_diff", "home_advantage")


class BaselinePredictor(Predictor):
    """Multinomial logistic regression on Elo difference + home advantage."""

    model_type = "baseline_elo"

    def __init__(self) -> None:
        self.feature_names = BASELINE_FEATURES
        self._pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        C=1.0,
                        random_state=config.RANDOM_SEED,
                    ),
                ),
            ]
        )

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        return X[list(BASELINE_FEATURES)].fillna(0.0).to_numpy(dtype=float)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> BaselinePredictor:
        self._pipe.fit(self._matrix(X), np.asarray(y))
        logger.info("Fitted %s on %d matches.", self.model_type, len(X))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = self._pipe.predict_proba(self._matrix(X))
        # LogisticRegression orders columns by sorted class labels (0,1,2), which
        # is exactly OUTCOME_CLASSES order (home_win, draw, away_win).
        return np.asarray(proba, dtype=float)
