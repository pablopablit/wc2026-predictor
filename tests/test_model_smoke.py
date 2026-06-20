"""Model smoke test (Phase 5) + scaffold-level interface checks (now).

The full smoke test trains on a tiny slice, predicts, and asserts probabilities
sum to 1. For now we verify the package imports and the Predictor interface is
shaped as the rest of the system expects.
"""

import pytest

from wc2026 import Match, Predictor, __version__
from wc2026.models.base import OUTCOME_CLASSES


def test_package_imports_and_version() -> None:
    assert isinstance(__version__, str)
    assert __version__.count(".") >= 2


def test_outcome_classes_order() -> None:
    assert OUTCOME_CLASSES == ("home_win", "draw", "away_win")


def test_predictor_is_abstract() -> None:
    with pytest.raises(TypeError):
        Predictor()  # type: ignore[abstract]


def test_match_carries_no_score() -> None:
    from datetime import date

    m = Match(home_team="Ecuador", away_team="Argentina", match_date=date(2026, 6, 20))
    # A Match must not expose any goal/score field (no-leakage by construction).
    assert not any("score" in f or "goal" in f for f in m.__dataclass_fields__)


def _toy_matrix():
    import pandas as pd

    from wc2026.features import build
    from wc2026.features.elo import EloModel

    results = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [f"2000-{m:02d}-01" for m in range(1, 13)]
                + [f"2001-{m:02d}-01" for m in range(1, 13)]
            ),
            "home_team": (["A", "B", "C", "D"] * 6),
            "away_team": (["B", "C", "D", "A"] * 6),
            "home_score": ([2, 0, 1, 3] * 6),
            "away_score": ([0, 1, 1, 0] * 6),
            "tournament": (["Friendly"] * 24),
            "neutral": ([False] * 24),
        }
    )
    ctx = build.FeatureContext(
        elo=EloModel().fit(results),
        confederations={t: "UEFA" for t in "ABCD"},
        structural={},
    )
    return build.build_training_matrix(results, ctx)


def test_train_predict_probs_sum_to_one() -> None:
    import numpy as np

    from wc2026.models.baseline import BaselinePredictor

    X, y = _toy_matrix()
    model = BaselinePredictor().fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_poisson_grid_orientation_and_probs() -> None:
    """A strong home side (high λ_home, low λ_away) must get the highest win prob,
    and probabilities must sum to 1 — guards the win/loss triangle orientation."""
    import numpy as np

    from wc2026.models.poisson import _poisson_grid

    p_h, p_d, p_a, grids = _poisson_grid(
        np.array([2.5, 0.4]), np.array([0.4, 2.5]), max_goals=10
    )
    np.testing.assert_allclose(p_h + p_d + p_a, 1.0, rtol=1e-6)
    assert p_h[0] > p_a[0]  # strong home favourite
    assert p_a[1] > p_h[1]  # strong away favourite


def test_poisson_fit_predict_smoke() -> None:
    """End-to-end Bayesian Poisson fit (fast MAP) + predict on the toy matrix."""
    import numpy as np
    import pandas as pd

    from wc2026.models.poisson import BayesianPoissonPredictor

    X, y = _toy_matrix()
    # toy results reused: build aligned goals from the same synthetic frame.
    goals = pd.DataFrame({"home_score": [2, 0, 1, 3] * 6, "away_score": [0, 1, 1, 0] * 6})
    model = BayesianPoissonPredictor().fit(X, y, goals=goals, method="map")
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)
    grids, scores = model.score_grid(X.iloc[:2])
    assert len(scores) == 2 and all(len(s) == 2 for s in scores)


def test_save_load_roundtrip(tmp_path) -> None:
    import datetime as dt

    import numpy as np

    from wc2026.models.baseline import BaselinePredictor

    X, y = _toy_matrix()
    model = BaselinePredictor().fit(X, y)
    model.build_meta(
        training_cutoff=dt.date(2001, 12, 1),
        scorecard={"log_loss": 1.0},
        manifest_hash="abc123",
    )
    path = model.save(tmp_path / "m.joblib")
    assert path.exists()
    assert (tmp_path / "m.model_meta.json").exists()

    loaded = BaselinePredictor.load(path)
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))
    assert loaded.meta is not None
    assert loaded.meta.data_manifest_hash == "abc123"
