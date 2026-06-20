"""Elo sanity + leakage-safety tests (Phase 3)."""

import pandas as pd
import pytest

from wc2026.features import elo
from wc2026.features.elo import BASE_RATING, EloModel


def _match(date, home, away, hs, as_, tournament="Friendly", neutral=True):
    return {
        "date": pd.Timestamp(date),
        "home_team": home,
        "away_team": away,
        "home_score": hs,
        "away_score": as_,
        "tournament": tournament,
        "neutral": neutral,
    }


def test_winner_gains_rating_loser_loses() -> None:
    df = pd.DataFrame([_match("2000-01-01", "A", "B", 2, 0)])
    model = EloModel().fit(df)
    assert model.current_rating("A") > BASE_RATING
    assert model.current_rating("B") < BASE_RATING


def test_rating_exchange_is_zero_sum() -> None:
    # Two equal teams; whatever the winner gains, the loser loses (sum conserved).
    df = pd.DataFrame([_match("2000-01-01", "A", "B", 3, 1)])
    model = EloModel().fit(df)
    total = model.current_rating("A") + model.current_rating("B")
    assert total == pytest.approx(2 * BASE_RATING, abs=1e-9)


def test_draw_between_equal_teams_is_no_op() -> None:
    df = pd.DataFrame([_match("2000-01-01", "A", "B", 1, 1, neutral=True)])
    model = EloModel().fit(df)
    assert model.current_rating("A") == pytest.approx(BASE_RATING)
    assert model.current_rating("B") == pytest.approx(BASE_RATING)


def test_home_advantage_shifts_expectation() -> None:
    # On home soil a draw should still nudge the (favoured) home team down a touch.
    no_ha = elo.expected_score(BASE_RATING, BASE_RATING, home_advantage=0.0)
    with_ha = elo.expected_score(BASE_RATING, BASE_RATING, home_advantage=65.0)
    assert no_ha == pytest.approx(0.5)
    assert with_ha > 0.5


def test_goal_margin_multiplier_monotonic() -> None:
    assert elo.goal_margin_multiplier(1) == 1.0
    assert elo.goal_margin_multiplier(2) == 1.5
    assert elo.goal_margin_multiplier(5) > elo.goal_margin_multiplier(3)


def test_importance_tiers() -> None:
    assert elo.importance_tier("Friendly") == 1
    assert elo.importance_tier("FIFA World Cup") == 5
    assert elo.importance_tier("FIFA World Cup qualification") == 3
    assert elo.importance_tier("UEFA Euro") == 4
    assert elo.match_importance("FIFA World Cup") > elo.match_importance("Friendly")


def test_rating_before_is_leakage_safe() -> None:
    # Three sequential matches for team A; rating_before must never see the
    # current match's result or any future match.
    df = pd.DataFrame(
        [
            _match("2000-01-01", "A", "B", 2, 0),
            _match("2000-02-01", "A", "C", 0, 1),
            _match("2000-03-01", "A", "D", 3, 3),
        ]
    )
    model = EloModel().fit(df)

    # Before its first match ever, A is at base.
    assert model.rating_before("A", "2000-01-01") == pytest.approx(BASE_RATING)
    # Strictly-before semantics: on the 2nd match date, A's rating reflects only
    # the 1st match (a win → above base), not the 2nd.
    r_before_2nd = model.rating_before("A", "2000-02-01")
    assert r_before_2nd > BASE_RATING
    # A team never seen is at base.
    assert model.rating_before("Z", "2000-01-01") == pytest.approx(BASE_RATING)


def test_rating_before_excludes_same_day_match() -> None:
    df = pd.DataFrame([_match("2000-01-01", "A", "B", 5, 0)])
    model = EloModel().fit(df)
    # Querying on the match date itself must NOT include that match's outcome.
    assert model.rating_before("A", "2000-01-01") == pytest.approx(BASE_RATING)
    # After the match, the rating has moved.
    assert model.rating_before("A", "2000-01-02") > BASE_RATING
