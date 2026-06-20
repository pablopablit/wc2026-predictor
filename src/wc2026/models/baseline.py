"""Rule-based baseline — the bar the real model must clear.

Predicts purely from Elo difference plus a home-advantage term, mapped to
win/draw/away probabilities (a fixed logistic mapping, no learning required for
the simplest variant). Cheap, interpretable, and a genuine benchmark: if the
gradient-boosted classifier cannot beat this on log-loss/Brier, it is not adding
value.
"""

from __future__ import annotations

import logging

from wc2026.models.base import Predictor

logger = logging.getLogger(__name__)


class BaselinePredictor(Predictor):
    """Elo-difference + home-advantage baseline. Implemented in Phase 5."""

    model_type = "baseline_elo"
