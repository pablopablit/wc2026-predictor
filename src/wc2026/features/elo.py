"""Leakage-safe football Elo rating timeline.

Standard football Elo computed from the martj42 results themselves (no external
Elo file in v1):

* Expected score from the rating difference (with a home-advantage term, in
  rating points, suppressed on neutral ground).
* K-factor scaled by **match importance** (tournament tier) and **goal margin**.
* Optional decay of ratings toward the mean for inactive teams (off by default).

The central guarantee is :meth:`EloModel.rating_before` — a team's rating as it
stood *strictly before* a given date — so any historical match can be featurized
without leaking its own (or any future) result. The per-match rating exchange is
**zero-sum** (the home team gains exactly what the away team loses), which the
tests assert as a conservation property.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from wc2026 import config

logger = logging.getLogger(__name__)

BASE_RATING = 1500.0

# --------------------------------------------------------------------------- #
# Match importance (tournament tier) — reused by the feature builder.
# --------------------------------------------------------------------------- #
#: K-factor multiplier per importance tier (1 = friendly … 5 = World Cup finals).
TIER_K_MULTIPLIER: dict[int, float] = {1: 0.5, 2: 0.75, 3: 1.0, 4: 1.25, 5: 1.5}


def importance_tier(tournament: str) -> int:
    """Map a tournament label to an importance tier in ``[1, 5]``.

    5 World Cup finals · 4 continental finals / Confederations Cup ·
    3 qualifiers + Nations League · 2 other tournaments · 1 friendlies.
    """
    t = (tournament or "").lower()
    if "friendly" in t:
        return 1
    if "qualif" in t:  # WC or continental qualification
        return 3
    if "fifa world cup" in t:
        return 5
    if "nations league" in t or "confederations cup" in t:
        return 3
    continental = (
        "uefa euro",
        "copa américa",
        "copa america",
        "african cup of nations",
        "afc asian cup",
        "gold cup",
        "concacaf championship",
        "oceania nations cup",
    )
    if any(c in t for c in continental):
        return 4
    return 2


def match_importance(tournament: str) -> float:
    """K-factor multiplier for a tournament (see :func:`importance_tier`)."""
    return TIER_K_MULTIPLIER[importance_tier(tournament)]


def goal_margin_multiplier(margin: int) -> float:
    """World-Football-Elo goal-margin weight: 1 goal →1.0, 2 →1.5, 3+ grows."""
    m = abs(int(margin))
    if m <= 1:
        return 1.0
    if m == 2:
        return 1.5
    return (11 + m) / 8.0


def expected_score(rating_home: float, rating_away: float, home_advantage: float) -> float:
    """Logistic expected score for the home side (includes home advantage)."""
    return 1.0 / (1.0 + 10.0 ** (-((rating_home + home_advantage) - rating_away) / 400.0))


class EloModel:
    """A fitted, leakage-safe Elo rating timeline over a results DataFrame."""

    def __init__(
        self,
        base_rating: float = BASE_RATING,
        k_factor: float = config.ELO_K_FACTOR,
        home_advantage: float = config.ELO_HOME_ADVANTAGE,
        decay_per_year: float = 0.0,
    ) -> None:
        self.base_rating = base_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.decay_per_year = decay_per_year

        self._ratings: dict[str, float] = {}
        self._last_played: dict[str, np.datetime64] = {}
        # Per-team post-match history for leakage-safe lookups.
        self._hist_dates: dict[str, list[np.datetime64]] = {}
        self._hist_after: dict[str, list[float]] = {}
        self._hist_dates_arr: dict[str, np.ndarray] = {}
        self._hist_after_arr: dict[str, np.ndarray] = {}
        self._timeline_rows: list[dict] = []
        self._fitted = False

    # -- internals ---------------------------------------------------------- #
    def _decayed(self, team: str, when: np.datetime64) -> float:
        rating = self._ratings.get(team, self.base_rating)
        if self.decay_per_year <= 0 or team not in self._last_played:
            return rating
        years = (when - self._last_played[team]) / np.timedelta64(365, "D")
        return self.base_rating + (rating - self.base_rating) * float(
            np.exp(-self.decay_per_year * max(years, 0.0))
        )

    def _record(self, team: str, when: np.datetime64, rating_after: float) -> None:
        self._ratings[team] = rating_after
        self._last_played[team] = when
        self._hist_dates.setdefault(team, []).append(when)
        self._hist_after.setdefault(team, []).append(rating_after)

    # -- fit ---------------------------------------------------------------- #
    def fit(self, results: pd.DataFrame) -> EloModel:
        """Walk matches in chronological order, updating ratings and the timeline."""
        df = results.sort_values("date", kind="stable")
        for row in df.itertuples(index=False):
            when = np.datetime64(pd.Timestamp(row.date), "D")
            home, away = row.home_team, row.away_team
            ha = 0.0 if bool(row.neutral) else self.home_advantage

            r_home = self._decayed(home, when)
            r_away = self._decayed(away, when)
            exp_home = expected_score(r_home, r_away, ha)

            if row.home_score > row.away_score:
                s_home = 1.0
            elif row.home_score == row.away_score:
                s_home = 0.5
            else:
                s_home = 0.0

            k = self.k_factor * match_importance(row.tournament)
            k *= goal_margin_multiplier(row.home_score - row.away_score)
            delta = k * (s_home - exp_home)

            new_home = r_home + delta
            new_away = r_away - delta  # zero-sum exchange
            self._record(home, when, new_home)
            self._record(away, when, new_away)
            self._timeline_rows.append(
                {
                    "date": row.date,
                    "home_team": home,
                    "away_team": away,
                    "rating_home_before": r_home,
                    "rating_away_before": r_away,
                    "rating_home_after": new_home,
                    "rating_away_after": new_away,
                    "delta": delta,
                }
            )

        # Freeze per-team history into sorted arrays for fast leakage-safe lookups.
        for team, dates in self._hist_dates.items():
            self._hist_dates_arr[team] = np.array(dates, dtype="datetime64[D]")
            self._hist_after_arr[team] = np.array(self._hist_after[team], dtype=float)
        self._fitted = True
        logger.info(
            "Fitted Elo on %d matches across %d teams.",
            len(self._timeline_rows),
            len(self._ratings),
        )
        return self

    # -- queries ------------------------------------------------------------ #
    def rating_before(self, team: str, when: date | pd.Timestamp | str) -> float:
        """Rating of ``team`` using only matches *strictly before* ``when``.

        Returns the base rating if the team has no prior matches. Same-day matches
        are excluded — the no-leakage guarantee.
        """
        arr = self._hist_dates_arr.get(team)
        if arr is None or len(arr) == 0:
            return self.base_rating
        d = np.datetime64(pd.Timestamp(when), "D")
        idx = int(np.searchsorted(arr, d, side="left"))
        if idx == 0:
            return self.base_rating
        return float(self._hist_after_arr[team][idx - 1])

    def current_rating(self, team: str) -> float:
        """Latest rating for ``team`` (base rating if never seen)."""
        return self._ratings.get(team, self.base_rating)

    def ratings(self) -> dict[str, float]:
        """A snapshot copy of all current ratings."""
        return dict(self._ratings)

    @property
    def timeline(self) -> pd.DataFrame:
        """Per-match rating timeline (for inspection / EDA)."""
        return pd.DataFrame(self._timeline_rows)
