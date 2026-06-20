"""Elo sanity tests (Phase 3).

When implemented these assert that a team which wins gains rating (and its
opponent loses it), and that total rating is approximately conserved across a
match (a zero-sum exchange).
"""

import pytest


@pytest.mark.skip(reason="EloModel is implemented in Phase 3.")
def test_winner_gains_rating_loser_loses() -> None:
    ...


@pytest.mark.skip(reason="EloModel is implemented in Phase 3.")
def test_rating_approximately_conserved() -> None:
    ...


@pytest.mark.skip(reason="EloModel is implemented in Phase 3.")
def test_rating_before_is_leakage_safe() -> None:
    ...
