"""The real model: a gradient-boosted 3-class match-outcome classifier.

Uses scikit-learn's ``HistGradientBoostingClassifier`` to avoid heavy extra deps
in v1, wrapped behind the :class:`~wc2026.models.base.Predictor` interface so
XGBoost/LightGBM can be swapped in later without touching callers.

Calibration matters more than raw accuracy for a probabilistic tournament sim, so
probabilities are calibrated (e.g. ``CalibratedClassifierCV``) and judged on
log-loss / Brier first, accuracy second.
"""

from __future__ import annotations

import logging

from wc2026.models.base import Predictor

logger = logging.getLogger(__name__)


class GradientBoostedPredictor(Predictor):
    """HistGradientBoostingClassifier + probability calibration. Phase 5."""

    model_type = "hist_gradient_boosting"
