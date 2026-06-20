"""Simulator helper + 2026 bracket/data integrity tests (no model needed)."""

from wc2026 import config
from wc2026.data import loaders
from wc2026.tournament import simulate


def test_groups_are_48_teams_in_12_groups_of_4() -> None:
    groups = loaders.load_wc2026_groups()
    assert len(groups) == config.NUM_GROUPS
    assert sorted(groups) == list(config.GROUP_NAMES)
    all_teams = [t for ts in groups.values() for t in ts]
    assert len(all_teams) == config.NUM_TEAMS
    assert len(set(all_teams)) == config.NUM_TEAMS  # no team in two groups
    assert all(len(ts) == config.TEAMS_PER_GROUP for ts in groups.values())


def test_hosts_in_their_assigned_groups() -> None:
    groups = loaders.load_wc2026_groups()
    for host, g in config.HOST_GROUPS.items():
        assert host in groups[g]


def test_fixtures_cover_full_round_robin_with_host_advantage() -> None:
    fx = loaders.load_wc2026_fixtures()
    assert len(fx) == config.NUM_GROUPS * 6  # 6 matches per group
    # Each host is the home side in exactly 3 group matches.
    for host in config.HOST_TEAMS:
        assert int((fx["home_team"] == host).sum()) == 3
    # Non-host group games are neutral.
    assert bool((~fx["neutral"]).sum() == 9)  # 3 hosts x 3 matches


def test_bracket_map_is_complete_and_consistent() -> None:
    b = loaders.load_bracket_map()
    r32 = b["r32"]
    assert len(r32) == 16
    third_slots = [e for e in r32 if e["away"].startswith("3:")]
    assert len(third_slots) == config.BEST_THIRD_PLACED_ADVANCING  # 8
    # Knockout feeders must reference matches that exist.
    known = {e["match"] for e in r32} | {e["match"] for e in b["knockout"]}
    for e in b["knockout"]:
        for side in ("home", "away"):
            assert int(e[side][1:]) in known
    # The tree ends in a single final.
    assert b["knockout"][-1]["round"] == "F"


def test_assign_thirds_respects_group_constraints() -> None:
    # Two slots; team from group A may only go to the slot that allows A.
    slots = [("3:CDF", "CDF"), ("3:ABF", "ABF")]
    ranked = [("TeamA", "A"), ("TeamC", "C")]
    assign = simulate._assign_thirds(ranked, slots)
    assert assign["3:ABF"] == "TeamA"  # A only fits the ABF slot
    assert assign["3:CDF"] == "TeamC"


def test_elo_penalty_favours_stronger_team() -> None:
    import numpy as np

    p = simulate._elo_penalty(np.array([2000.0]), np.array([1600.0]))[0]
    assert p > 0.5
    assert simulate._elo_penalty(np.array([1500.0]), np.array([1500.0]))[0] == 0.5
