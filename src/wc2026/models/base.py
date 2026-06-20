"""The :class:`Predictor` interface and the :class:`Match` input contract.

Every model in this package — the Elo-logit baseline and the Bayesian Poisson
model alike — implements the same ``fit``/``predict_proba``/``save``/``load``
surface. The Monte Carlo simulator and the CLI depend only on this interface, so
swapping in another model later is a drop-in change behind ``Predictor``.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd

from wc2026 import __version__, config

if TYPE_CHECKING:
    from wc2026.features.build import FeatureContext

logger = logging.getLogger(__name__)

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

    #: True for models whose fit needs the raw goal targets (e.g. the Poisson
    #: model), not just the 3-class label. The backtest/trainer pass goals when so.
    requires_goals: bool = False

    #: Populated after fit/evaluate so it can be bundled into ``ModelMeta``.
    meta: ModelMeta | None = None

    # -- core interface ----------------------------------------------------- #
    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        goals: pd.DataFrame | None = None,
    ) -> Predictor:
        """Fit on a feature matrix ``X`` (columns from ``features.build``) and
        integer labels ``y`` (0/1/2 → home_win/draw/away_win). ``goals`` (home/away
        score) is supplied only to models with ``requires_goals``. Returns ``self``."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return an ``(n, 3)`` array of calibrated probabilities, columns in
        :data:`OUTCOME_CLASSES` order. Each row sums to 1."""

    # -- convenience -------------------------------------------------------- #
    def predict_match(
        self, match: Match, history: pd.DataFrame, ctx: FeatureContext
    ) -> np.ndarray:
        """Featurize a single :class:`Match` (leakage-safe) and predict its
        ``(1, 3)`` outcome probabilities."""
        from wc2026.features.build import build_features

        return self.predict_proba(build_features(match, history, ctx))

    # -- persistence (default implementation) ------------------------------- #
    def _default_path(self) -> Path:
        return config.MODELS_DIR / f"{self.model_type}.joblib"

    def save(self, path: str | Path | None = None) -> Path:
        """Serialize the model (joblib) plus a human-readable ``*.model_meta.json``
        sidecar carrying the cutoff, feature list, scorecard, and manifest hash."""
        dest = Path(path) if path is not None else self._default_path()
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dest)
        if self.meta is not None:
            sidecar = dest.with_suffix(".model_meta.json")
            meta_dict = asdict(self.meta)
            meta_dict["feature_names"] = list(self.meta.feature_names)
            if self.meta.training_cutoff is not None:
                meta_dict["training_cutoff"] = self.meta.training_cutoff.isoformat()
            sidecar.write_text(json.dumps(meta_dict, indent=2) + "\n")
        logger.info("Saved %s model to %s", self.model_type, dest)
        return dest

    @classmethod
    def load(cls, path: str | Path | None = None) -> Predictor:
        """Load a model, warning if its data-manifest hash no longer matches the
        current data state (a prediction should be traceable to its data)."""
        from wc2026.data import ingest

        src = Path(path) if path is not None else config.MODELS_DIR / f"{cls.model_type}.joblib"
        if not src.exists():
            raise FileNotFoundError(
                f"No model at {src}. Train one first with `wc2026 train` (or `make train`)."
            )
        model: Predictor = joblib.load(src)
        if model.meta is not None:
            current = ingest.manifest_hash()
            stored = model.meta.data_manifest_hash
            if current is not None and stored is not None and current != stored:
                logger.warning(
                    "Model was trained on data manifest %s but the current manifest is %s — "
                    "predictions may not be reproducible.",
                    stored,
                    current,
                )
        return model

    # -- metadata helper ---------------------------------------------------- #
    def build_meta(
        self,
        training_cutoff: date | None,
        scorecard: dict[str, float],
        manifest_hash: str | None,
    ) -> ModelMeta:
        """Assemble (and attach) the :class:`ModelMeta` sidecar for this model."""
        from datetime import UTC, datetime

        self.meta = ModelMeta(
            model_type=self.model_type,
            training_cutoff=training_cutoff,
            feature_names=tuple(self.feature_names),
            metric_scorecard=scorecard,
            data_manifest_hash=manifest_hash,
            random_seed=config.RANDOM_SEED,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            wc2026_version=__version__,
        )
        return self.meta
