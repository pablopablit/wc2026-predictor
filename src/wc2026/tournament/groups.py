"""Group standings, the exact 2026 tiebreakers, and best-third-placed ranking.

Standings are computed from a set of played group matches and ties are broken in
the order fixed in :data:`wc2026.config.TIEBREAKERS`:

    points → goal difference → goals scored → head-to-head → fair-play → lots

Head-to-head is applied *only among the teams still tied* (a mini-table of their
matches against each other). Drawing of lots is made deterministic from
``RANDOM_SEED`` so results are reproducible.

The twelve third-placed teams are ranked (points → GD → goals → fair-play → lots,
no head-to-head since they are in different groups) and the top eight advance to
the Round of 32. This logic is unit-tested against hand-built tables in
``tests/test_tournament_rules.py``.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import pandas as pd

from wc2026 import config

logger = logging.getLogger(__name__)

_REQUIRED_COLS = ("home_team", "away_team", "home_score", "away_score")


@dataclass
class TeamStanding:
    """One team's aggregated group record."""

    team: str
    played: int = 0
    won: int = 0
    drawn: int = 0
    lost: int = 0
    goals_for: int = 0
    goals_against: int = 0
    points: int = 0
    fair_play: int = 0  # FIFA fair-play points (<= 0; higher = fewer cards = better)

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against

    def as_dict(self) -> dict:
        return {
            "team": self.team,
            "played": self.played,
            "won": self.won,
            "drawn": self.drawn,
            "lost": self.lost,
            "goals_for": self.goals_for,
            "goals_against": self.goals_against,
            "goal_difference": self.goal_difference,
            "points": self.points,
            "fair_play": self.fair_play,
        }


def _lots_key(team: str, seed: int = config.RANDOM_SEED) -> str:
    """Deterministic 'drawing of lots' key — reproducible from the seed."""
    return hashlib.sha256(f"{seed}:{team}".encode()).hexdigest()


def compute_standings(
    matches: pd.DataFrame, teams: list[str] | None = None
) -> dict[str, TeamStanding]:
    """Aggregate played ``matches`` into per-team standings.

    Required columns: ``home_team, away_team, home_score, away_score``.
    Optional: ``home_fair_play, away_fair_play`` (defaults 0).
    """
    missing = [c for c in _REQUIRED_COLS if c not in matches.columns]
    if missing:
        raise ValueError(f"group matches missing columns: {missing}")

    if teams is None:
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    standings = {t: TeamStanding(team=t) for t in teams}
    has_fp = "home_fair_play" in matches.columns and "away_fair_play" in matches.columns

    for m in matches.itertuples(index=False):
        h, a = standings[m.home_team], standings[m.away_team]
        hs, as_ = int(m.home_score), int(m.away_score)
        h.played += 1
        a.played += 1
        h.goals_for += hs
        h.goals_against += as_
        a.goals_for += as_
        a.goals_against += hs
        if has_fp:
            h.fair_play += int(m.home_fair_play)
            a.fair_play += int(m.away_fair_play)
        if hs > as_:
            h.won += 1
            a.lost += 1
            h.points += 3
        elif hs < as_:
            a.won += 1
            h.lost += 1
            a.points += 3
        else:
            h.drawn += 1
            a.drawn += 1
            h.points += 1
            a.points += 1
    return standings


def _h2h_standings(
    teams: list[str], matches: pd.DataFrame
) -> dict[str, TeamStanding]:
    """Mini-table restricted to matches *between* the given (tied) teams."""
    sub = matches[matches["home_team"].isin(teams) & matches["away_team"].isin(teams)]
    return compute_standings(sub, teams)


def _cluster(items: list[str], key) -> list[list[str]]:
    """Group a pre-sorted list into runs sharing the same ``key`` value."""
    clusters: list[list[str]] = []
    for it in items:
        if clusters and key(it) == key(clusters[-1][0]):
            clusters[-1].append(it)
        else:
            clusters.append([it])
    return clusters


def _order_by_overall(teams: list[str], st: dict[str, TeamStanding]) -> list[str]:
    return sorted(
        teams,
        key=lambda t: (st[t].points, st[t].goal_difference, st[t].goals_for),
        reverse=True,
    )


def _resolve_tied(
    tied: list[str], matches: pd.DataFrame, st: dict[str, TeamStanding], seed: int
) -> list[str]:
    """Break a cluster tied on (points, GD, goals) via head-to-head → fair-play → lots."""
    h2h = _h2h_standings(tied, matches)
    ordered = sorted(
        tied,
        key=lambda t: (h2h[t].points, h2h[t].goal_difference, h2h[t].goals_for),
        reverse=True,
    )
    out: list[str] = []
    for sub in _cluster(
        ordered, key=lambda t: (h2h[t].points, h2h[t].goal_difference, h2h[t].goals_for)
    ):
        if len(sub) == 1:
            out.extend(sub)
        else:
            # fair-play (higher better), then deterministic lots.
            out.extend(sorted(sub, key=lambda t: (-st[t].fair_play, _lots_key(t, seed))))
    return out


def rank_group(
    matches: pd.DataFrame,
    teams: list[str] | None = None,
    seed: int = config.RANDOM_SEED,
) -> list[str]:
    """Return group teams ordered 1st→last, applying the full 2026 tiebreakers."""
    st = compute_standings(matches, teams)
    all_teams = list(st)
    ordered = _order_by_overall(all_teams, st)
    result: list[str] = []
    for cluster in _cluster(
        ordered, key=lambda t: (st[t].points, st[t].goal_difference, st[t].goals_for)
    ):
        result.extend(
            cluster if len(cluster) == 1 else _resolve_tied(cluster, matches, st, seed)
        )
    return result


def group_table(
    matches: pd.DataFrame,
    teams: list[str] | None = None,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Ranked standings table (position 1..N) for one group."""
    st = compute_standings(matches, teams)
    order = rank_group(matches, teams, seed)
    rows = [{"position": i + 1, **st[t].as_dict()} for i, t in enumerate(order)]
    return pd.DataFrame(rows)


def rank_third_placed(
    thirds: list[TeamStanding], seed: int = config.RANDOM_SEED
) -> list[TeamStanding]:
    """Rank the third-placed teams (points → GD → goals → fair-play → lots).

    No head-to-head — these teams are in different groups. The first
    :data:`config.BEST_THIRD_PLACED_ADVANCING` entries are the qualifiers.
    """
    return sorted(
        thirds,
        key=lambda s: (
            -s.points,
            -s.goal_difference,
            -s.goals_for,
            -s.fair_play,
            _lots_key(s.team, seed),
        ),
    )


def best_third_placed(
    thirds: list[TeamStanding], seed: int = config.RANDOM_SEED
) -> list[TeamStanding]:
    """The top eight third-placed teams that advance to the Round of 32."""
    return rank_third_placed(thirds, seed)[: config.BEST_THIRD_PLACED_ADVANCING]
