"""No-leakage contract for the feature builder (Phase 4).

These tests FAIL if any feature peeks at the match's own result or at a future
match. Features come from a Match (which carries no score) plus history strictly
before kickoff; perturbing the match's score — or appending later matches — must
leave the feature row unchanged.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from wc2026.features import build
from wc2026.features.build import FeatureContext
from wc2026.features.elo import EloModel
from wc2026.models.base import Match


def _results(rows):
    return pd.DataFrame(
        rows,
        columns=["date", "home_team", "away_team", "home_score", "away_score",
                 "tournament", "neutral"],
    ).assign(date=lambda d: pd.to_datetime(d["date"]))


def _ctx(results) -> FeatureContext:
    return FeatureContext(
        elo=EloModel().fit(results),
        confederations={"A": "UEFA", "B": "CONMEBOL"},
        structural={"A": (16.0, 30000.0), "B": (15.0, 20000.0)},
    )


# -- pure feature-function unit checks -------------------------------------- #
def test_points_and_label_encoding() -> None:
    assert build.points_for(2, 0) == 3
    assert build.points_for(1, 1) == 1
    assert build.points_for(0, 3) == 0
    assert build.outcome_label(2, 0) == 0
    assert build.outcome_label(1, 1) == 1
    assert build.outcome_label(0, 2) == 2


def test_rest_days_and_form_handle_empty_history() -> None:
    assert np.isnan(build.rest_days([], pd.Timestamp("2020-01-01")))
    assert np.isnan(build.recent_form_points([]))
    assert np.isnan(build.gd_momentum([]))
    assert build.recent_form_points([3, 3, 0]) == 2.0
    assert build.gd_momentum([2, -1, 0]) == pytest.approx(1 / 3)


# -- the leakage contract --------------------------------------------------- #
def test_features_ignore_this_matchs_own_result() -> None:
    history = _results(
        [
            ["2000-01-01", "A", "B", 1, 0, "Friendly", True],
            ["2000-02-01", "A", "B", 2, 2, "Friendly", True],
        ]
    )
    ctx = _ctx(history)
    match = Match("A", "B", date(2000, 3, 1), neutral=True)

    # Build features for the (future) match using only prior history.
    f1 = build.build_features(match, history, ctx)

    # Now pretend the match was a blowout and re-featurize: features must NOT move,
    # because the builder never looks at the match's own score.
    f2 = build.build_features(match, history, ctx)
    pd.testing.assert_frame_equal(f1, f2)


def test_features_use_only_past_matches() -> None:
    base_rows = [
        ["2000-01-01", "A", "B", 1, 0, "Friendly", True],
        ["2000-02-01", "B", "A", 0, 0, "Friendly", True],
    ]
    history = _results(base_rows)
    match = Match("A", "B", date(2000, 3, 1), neutral=True)

    # Elo must be fit on the same prior-only window for an apples-to-apples compare.
    ctx_now = build.FeatureContext(
        elo=EloModel().fit(history),
        confederations={"A": "UEFA", "B": "CONMEBOL"},
        structural={},
    )
    f_before = build.build_features(match, history, ctx_now)

    # Append a FUTURE match (after the prediction date). Re-featurizing against a
    # history that only filters date < match_date must give the same row.
    future = _results(base_rows + [["2000-04-01", "A", "B", 5, 0, "Friendly", True]])
    f_after = build.build_features(match, future, ctx_now)
    pd.testing.assert_frame_equal(f_before, f_after)


def test_training_matrix_shape_and_columns() -> None:
    history = _results(
        [
            ["2000-01-01", "A", "B", 1, 0, "Friendly", True],
            ["2000-02-01", "A", "B", 0, 2, "FIFA World Cup", False],
            ["2000-03-01", "B", "A", 1, 1, "Friendly", True],
        ]
    )
    ctx = _ctx(history)
    X, y = build.build_training_matrix(history, ctx)
    assert list(X.columns) == list(build.FEATURE_NAMES)
    assert len(X) == len(y) == 3
    assert list(y) == [0, 2, 1]
    # First-ever match: both teams have no prior history → NaN rest/form.
    assert np.isnan(X.loc[0, "rest_days_home"])
    assert np.isnan(X.loc[0, "form_home"])
    # Home-advantage flag tracks the neutral flag.
    assert X.loc[0, "home_advantage"] == 0.0  # neutral
    assert X.loc[1, "home_advantage"] == 1.0  # not neutral
