"""Metric + walk-forward backtest tests (Phase 5)."""

import numpy as np

from wc2026.evaluate import metrics


def test_perfect_predictions_score_well() -> None:
    y = np.array([0, 1, 2, 0])
    proba = np.eye(3)[y] * 0.98 + 0.01  # near-one-hot, valid rows
    proba = proba / proba.sum(axis=1, keepdims=True)
    assert metrics.accuracy(y, proba) == 1.0
    assert metrics.multiclass_brier(y, proba) < 0.01
    assert metrics.multiclass_log_loss(y, proba) < 0.1


def test_uniform_predictions_have_expected_logloss() -> None:
    y = np.array([0, 1, 2])
    proba = np.full((3, 3), 1 / 3)
    # log-loss of a uniform 3-class predictor is ln(3).
    assert abs(metrics.multiclass_log_loss(y, proba) - np.log(3)) < 1e-9


def test_brier_worst_case() -> None:
    y = np.array([0])
    proba = np.array([[0.0, 0.0, 1.0]])  # all mass on the wrong class
    assert metrics.multiclass_brier(y, proba) == 2.0


def test_walk_forward_splits_are_temporal_and_disjoint() -> None:
    splits = list(metrics.walk_forward_splits(100, n_splits=5, min_train_frac=0.5))
    assert len(splits) == 5
    seen_test: set[int] = set()
    for tr, te in splits:
        # train is strictly before test
        assert tr.max() < te.min()
        # test blocks are disjoint
        assert not (seen_test & set(te.tolist()))
        seen_test |= set(te.tolist())
    # the union of test blocks covers the back half exactly
    assert min(seen_test) == 50 and max(seen_test) == 99


def test_reliability_curve_columns() -> None:
    rng = np.random.default_rng(0)
    proba = rng.dirichlet([1, 1, 1], size=50)
    y = rng.integers(0, 3, size=50)
    df = metrics.reliability_curve(y, proba, n_bins=5)
    assert set(df.columns) == {"bin_lo", "bin_hi", "count", "mean_confidence", "accuracy"}
    assert df["count"].sum() == 50
