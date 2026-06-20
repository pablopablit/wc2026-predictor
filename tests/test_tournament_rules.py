"""Tournament-rule tests (Phase 6).

Standings + the exact 2026 tiebreaker chain + best-third-placed selection,
checked against hand-constructed group tables. Two scaffold-level checks below run
now to lock the fixed format constants in place.
"""

import pytest

from wc2026 import config


def test_format_constants_are_fixed() -> None:
    assert config.NUM_TEAMS == 48
    assert config.NUM_GROUPS == 12
    assert config.TEAMS_PER_GROUP == 4
    assert len(config.GROUP_NAMES) == 12
    assert config.GROUP_NAMES[0] == "A" and config.GROUP_NAMES[-1] == "L"
    assert config.R32_TEAMS == 32
    assert config.NUM_TEAMS == config.NUM_GROUPS * config.TEAMS_PER_GROUP


def test_tiebreakers_in_official_order() -> None:
    assert config.TIEBREAKERS == (
        "points",
        "goal_difference",
        "goals_scored",
        "head_to_head",
        "fair_play",
        "drawing_of_lots",
    )


def test_host_group_assignments() -> None:
    assert config.HOST_GROUPS["United States"] == "D"
    assert config.HOST_GROUPS["Mexico"] == "A"
    assert config.HOST_GROUPS["Canada"] == "B"


@pytest.mark.skip(reason="Standings/tiebreakers implemented in Phase 6.")
def test_standings_break_ties_in_order() -> None:
    ...


@pytest.mark.skip(reason="Best-third-placed ranking implemented in Phase 6.")
def test_best_eight_third_placed_selected() -> None:
    ...
