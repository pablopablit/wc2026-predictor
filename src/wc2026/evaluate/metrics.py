"""Metrics, calibration, and the temporal (walk-forward) backtest.

Primary metrics are probabilistic — multiclass **log-loss** and **Brier score** —
with accuracy reported as a secondary, plus a reliability curve.

The backtest is strictly temporal: an expanding training window predicts the next
time-ordered block; nothing is shuffled across time, so reported numbers reflect
genuine out-of-sample, future-facing performance. The feature matrix is already
leakage-safe per row (Elo ``rating_before`` + history-appended-after), so we can
split a precomputed matrix by time for the model fit/eval step.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from wc2026.models.base import OUTCOME_CLASSES, Predictor

logger = logging.getLogger(__name__)

_LABELS = (0, 1, 2)


def multiclass_log_loss(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Lower is better. Uses the fixed 3-class label set so empty folds are safe."""
    return float(log_loss(y_true, proba, labels=_LABELS))


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean squared error between the probability vector and the one-hot truth."""
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true)), np.asarray(y_true, dtype=int)] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def accuracy(y_true: np.ndarray, proba: np.ndarray) -> float:
    return float(accuracy_score(y_true, proba.argmax(axis=1)))


def compute_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "log_loss": multiclass_log_loss(y_true, proba),
        "brier": multiclass_brier(y_true, proba),
        "accuracy": accuracy(y_true, proba),
    }


def reliability_curve(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Top-label reliability: bin by predicted-class confidence, compare to the
    empirical accuracy within each bin. Well-calibrated → confidence ≈ accuracy."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == np.asarray(y_true)).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            rows.append(
                {
                    "bin_lo": bins[b],
                    "bin_hi": bins[b + 1],
                    "count": int(mask.sum()),
                    "mean_confidence": float(conf[mask].mean()),
                    "accuracy": float(correct[mask].mean()),
                }
            )
    return pd.DataFrame(rows)


def walk_forward_splits(
    n: int, n_splits: int = 5, min_train_frac: float = 0.5
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) for an expanding-window temporal backtest.

    Assumes rows are already sorted by date. The first ``min_train_frac`` of the
    data seeds the initial training window; the remainder is split into
    ``n_splits`` consecutive test blocks.
    """
    start = int(n * min_train_frac)
    if start <= 0 or start >= n:
        raise ValueError(f"min_train_frac={min_train_frac} leaves no train/test data for n={n}.")
    fold = max((n - start) // n_splits, 1)
    for i in range(n_splits):
        tr_end = start + i * fold
        te_end = n if i == n_splits - 1 else min(tr_end + fold, n)
        if tr_end >= n or tr_end >= te_end:
            break
        yield np.arange(tr_end), np.arange(tr_end, te_end)


@dataclass
class Scorecard:
    """Aggregated backtest result for one model."""

    model_type: str
    folds: list[dict] = field(default_factory=list)

    def add_fold(self, fold: int, n_test: int, metrics: dict[str, float]) -> None:
        self.folds.append({"fold": fold, "n_test": n_test, **metrics})

    def mean(self) -> dict[str, float]:
        if not self.folds:
            return {}
        keys = ("log_loss", "brier", "accuracy")
        return {k: float(np.mean([f[k] for f in self.folds])) for k in keys}

    def to_dict(self) -> dict:
        return {"model_type": self.model_type, "folds": self.folds, "mean": self.mean()}

    def __str__(self) -> str:
        lines = [f"Scorecard — {self.model_type}  (classes: {', '.join(OUTCOME_CLASSES)})"]
        lines.append(f"{'fold':>4} {'n_test':>8} {'log_loss':>10} {'brier':>8} {'accuracy':>9}")
        for f in self.folds:
            lines.append(
                f"{f['fold']:>4} {f['n_test']:>8} {f['log_loss']:>10.4f} "
                f"{f['brier']:>8.4f} {f['accuracy']:>9.4f}"
            )
        m = self.mean()
        if m:
            lines.append(
                f"{'mean':>4} {'':>8} {m['log_loss']:>10.4f} "
                f"{m['brier']:>8.4f} {m['accuracy']:>9.4f}"
            )
        return "\n".join(lines)


def backtest(
    model_factory: Callable[[], Predictor],
    X: pd.DataFrame,
    y: pd.Series,
    goals: pd.DataFrame | None = None,
    n_splits: int = 5,
    min_train_frac: float = 0.5,
) -> Scorecard:
    """Walk-forward backtest: refit a fresh model per fold on the expanding window
    and score it on the next time-ordered block. ``goals`` (home/away score) is
    passed through to models that need it (``requires_goals``)."""
    y_arr = np.asarray(y)
    card = Scorecard(model_type=model_factory().model_type)
    for i, (tr, te) in enumerate(walk_forward_splits(len(X), n_splits, min_train_frac)):
        model = model_factory()
        if model.requires_goals:
            if goals is None:
                raise ValueError(f"{model.model_type} requires goal targets; pass goals=.")
            model.fit(X.iloc[tr], y_arr[tr], goals=goals.iloc[tr])
        else:
            model.fit(X.iloc[tr], y_arr[tr])
        proba = model.predict_proba(X.iloc[te])
        card.add_fold(i, len(te), compute_metrics(y_arr[te], proba))
    logger.info("Backtest of %s: %s", card.model_type, card.mean())
    return card
