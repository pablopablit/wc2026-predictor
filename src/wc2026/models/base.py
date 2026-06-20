"""The :class:`Predictor` interface and the :class:`Match` input contract.

Every model in this package — the rule-based baseline and the gradient-boosted
classifier alike — implements the same ``fit``/``predict_proba``/``save``/``load``
surface. The Monte Carlo simulator and the CLI depend only on this interface, so
swapping in XGBoost/LightGBM later is a drop-in change behind ``Predictor``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

#: Canonical class order for all probability outputs: P(home win), P(draw),
#: P(away win). Every ``predict_proba`` returns columns in exactly this order.
OUTCOME_CLASSES: tuple[str, str, str] = ("home_win", "draw", "away_win")


@dataclass(frozen=True, slots=True)
class Match:
    """Minimal match context required to make a prediction.

    All fields describe information knowable *strictly before kickoff* — there is
    deliberately no score here, so a ``Match`` can never leak its own result into
    a feature (see ``features`` and ``tests/test_features_no_leakage.py``).
    """

    home_team: str
    away_team: str
    match_date: date
    neutral: bool = False
    #: Free-form context the feature builder may use (e.g. tournament tier).
    context: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelMeta:
    """Sidecar metadata bundled with a persisted model (``model_meta.json``).

    Loading a model whose metadata does not match the current data manifest should
    warn — a prediction must be traceable to the data and code that produced it.
    """

    model_type: str
    training_cutoff: date | None
    feature_names: tuple[str, ...]
    metric_scorecard: dict[str, float]
    data_manifest_hash: str | None
    random_seed: int
    created_at: str
    wc2026_version: str


class Predictor(ABC):
    """Abstract base for all match-outcome predictors.

    Concrete subclasses must implement :meth:`fit` and :meth:`predict_proba`.
    ``save``/``load`` have working default implementations (joblib + JSON sidecar)
    that subclasses may rely on.
    """

    #: Set by subclasses; surfaced in metadata and used for the default filename.
    model_type: str = "abstract"

    #: Populated during ``fit``; the exact, ordered feature list the model expects.
    feature_names: tuple[str, ...] = ()

    #: Populated after fit/evaluate so it can be bundled into ``ModelMeta``.
    meta: ModelMeta | None = None

    # -- core interface ----------------------------------------------------- #
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Predictor:
        """Fit on a feature matrix ``X`` and integer labels ``y`` (0/1/2 →
        home_win/draw/away_win). Returns ``self`` for chaining."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return an ``(n, 3)`` array of calibrated probabilities, columns in
        :data:`OUTCOME_CLASSES` order. Each row sums to 1."""

    # -- persistence (default implementation) ------------------------------- #
    def save(self, path: str | Path) -> Path:
        """Serialize the model plus its ``model_meta.json`` sidecar.

        Implemented in Phase 5. Stubbed here so the interface is complete.
        """
        raise NotImplementedError("save() is implemented in Phase 5 (persistence).")

    @classmethod
    def load(cls, path: str | Path | None = None) -> Predictor:
        """Load a model and its metadata, warning on a manifest mismatch.

        Implemented in Phase 5. Stubbed here so the interface is complete.
        """
        raise NotImplementedError("load() is implemented in Phase 5 (persistence).")
