"""Tournament-rule tests (Phase 6).

Standings + the exact 2026 tiebreaker chain + best-third-placed selection,
checked against hand-constructed group tables. Two scaffold-level checks below run
now to lock the fixed format constants in place.
"""

import pandas as pd

from wc2026 import config
from wc2026.tournament import groups
from wc2026.tournament.groups import TeamStanding


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


def _round_robin(scores: dict[tuple[str, str], tuple[int, int]], fair_play=None) -> pd.DataFrame:
    rows = []
    for (h, a), (hs, as_) in scores.items():
        row = {"home_team": h, "away_team": a, "home_score": hs, "away_score": as_}
        if fair_play:
            row["home_fair_play"] = fair_play.get(h, 0)
            row["away_fair_play"] = fair_play.get(a, 0)
        rows.append(row)
    return pd.DataFrame(rows)


def test_standings_points_and_gd() -> None:
    # A beats B and C; B beats C. A=6, B=3, C=0.
    m = _round_robin({("A", "B"): (2, 0), ("A", "C"): (3, 1), ("B", "C"): (1, 0)})
    st = groups.compute_standings(m)
    assert (st["A"].points, st["A"].goal_difference) == (6, 4)
    assert st["B"].points == 3
    assert st["C"].points == 0
    assert groups.rank_group(m) == ["A", "B", "C"]


def test_tiebreak_by_goal_difference_then_goals() -> None:
    # A and B both 4 pts; A has better GD; C, D weaker.
    m = _round_robin(
        {
            ("A", "B"): (1, 1), ("A", "C"): (4, 0), ("A", "D"): (1, 0),
            ("B", "C"): (1, 0), ("B", "D"): (2, 1), ("C", "D"): (0, 0),
        }
    )
    order = groups.rank_group(m)
    assert order[:2] == ["A", "B"]  # A ahead on GD (+5 vs +1)


def test_tiebreak_head_to_head() -> None:
    # A and B level on points (6), GD (+1) and goals (2) overall; B won A-B 1-0.
    m = _round_robin(
        {
            ("A", "B"): (0, 1), ("A", "C"): (1, 0), ("A", "D"): (1, 0),
            ("B", "C"): (0, 1), ("B", "D"): (1, 0), ("C", "D"): (0, 1),
        }
    )
    st = groups.compute_standings(m)
    triple = lambda t: (st[t].points, st[t].goal_difference, st[t].goals_for)  # noqa: E731
    assert triple("A") == triple("B") == (6, 1, 2)  # genuinely tied before H2H
    order = groups.rank_group(m)
    assert order.index("B") < order.index("A")  # B wins the head-to-head


def test_tiebreak_fair_play_then_lots() -> None:
    # Two teams identical on everything incl. H2H draw; fewer cards wins.
    m = _round_robin(
        {("A", "B"): (1, 1)},
        fair_play={"A": -2, "B": -5},  # B has more cards (more negative) → worse
    )
    order = groups.rank_group(m, teams=["A", "B"])
    assert order == ["A", "B"]


def test_lots_is_deterministic() -> None:
    m = _round_robin({("A", "B"): (0, 0)})
    assert groups.rank_group(m, teams=["A", "B"]) == groups.rank_group(m, teams=["A", "B"])


def test_best_third_placed_selects_top_eight() -> None:
    # 12 third-placed teams with descending points; top 8 by points should advance.
    thirds = [
        TeamStanding(team=f"T{i}", played=3, points=p, goals_for=p, goals_against=0)
        for i, p in enumerate([7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 1, 0])
    ]
    qualifiers = groups.best_third_placed(thirds)
    assert len(qualifiers) == 8
    assert [q.points for q in qualifiers] == [7, 6, 6, 5, 5, 4, 4, 3]


def test_best_third_placed_uses_goal_difference() -> None:
    a = TeamStanding(team="A", points=3, goals_for=5, goals_against=1)  # GD +4
    b = TeamStanding(team="B", points=3, goals_for=2, goals_against=1)  # GD +1
    ranked = groups.rank_third_placed([b, a])
    assert [t.team for t in ranked] == ["A", "B"]
