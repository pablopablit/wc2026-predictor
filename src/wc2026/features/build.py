"""Assemble inputs for the models — one small, tested function each.

Two consumers share this module:
* the **Elo-logit baseline**, which uses the match-context features below;
* the **Bayesian Poisson** model, which additionally uses per-team **structural
  priors** (log-population, GDP per capita) so data-scarce teams shrink toward a
  sensible prior rather than noise.

Match-context features (each computable strictly pre-kickoff):

* ``elo_diff``            home Elo − away Elo (before kickoff)
* ``elo_home`` / ``elo_away``  individual ratings
* ``home_advantage``      1 for a genuine home side, 0 if neutral
* ``host_home`` / ``host_away``  is the team a 2026 host
* ``rest_days_home/away`` days since each team's previous international
* ``form_home/away``      points in the last N internationals
* ``gd_momentum_home/away``  mean goal difference over the last N
* ``importance_tier``     match-importance tier from the tournament label
* ``log_pop_home/away``   log population (World Bank, structural prior)
* ``gdp_home/away``       GDP per capita (World Bank, structural prior)

Categorical: ``confed_home`` / ``confed_away`` (FIFA confederation).

The no-leakage contract (``tests/test_features_no_leakage.py``): every feature is
a function of a :class:`~wc2026.models.base.Match` plus history *before* the match
date; none reads the match's own score or any future match. Market/betting odds
are never inputs (``config.MARKET_ODDS_AS_INPUT``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from wc2026 import config
from wc2026.data import loaders
from wc2026.features import elo as elo_mod
from wc2026.features.elo import EloModel
from wc2026.models.base import Match

logger = logging.getLogger(__name__)

#: Ordered numeric feature columns (the model consumes them in this order).
NUMERIC_FEATURES: tuple[str, ...] = (
    "elo_diff",
    "elo_home",
    "elo_away",
    "home_advantage",
    "host_home",
    "host_away",
    "rest_days_home",
    "rest_days_away",
    "form_home",
    "form_away",
    "gd_momentum_home",
    "gd_momentum_away",
    "importance_tier",
    "log_pop_home",
    "log_pop_away",
    "gdp_home",
    "gdp_away",
)
#: Categorical feature columns.
CATEGORICAL_FEATURES: tuple[str, ...] = ("confed_home", "confed_away")
FEATURE_NAMES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DEFAULT_FORM_WINDOW = 5

#: Team name -> World Bank country name, for the renamed/abbreviated entries.
WB_ALIASES: dict[str, str] = {
    "Egypt": "Egypt, Arab Rep.",
    "Iran": "Iran, Islamic Rep.",
    "South Korea": "Korea, Rep.",
    "North Korea": "Korea, Dem. People's Rep.",
    "Slovakia": "Slovak Republic",
    "Syria": "Syrian Arab Republic",
    "Venezuela": "Venezuela, RB",
    "DR Congo": "Congo, Dem. Rep.",
    "Congo": "Congo, Rep.",
    "Kyrgyzstan": "Kyrgyz Republic",
    "Hong Kong": "Hong Kong SAR, China",
    "Macau": "Macao SAR, China",
    "Bahamas": "Bahamas, The",
    "Gambia": "Gambia, The",
    "Yemen": "Yemen, Rep.",
    "Somalia": "Somalia, Fed. Rep.",
    "Micronesia": "Micronesia, Fed. Sts.",
    "Saint Kitts and Nevis": "St. Kitts and Nevis",
    "Saint Lucia": "St. Lucia",
    "Saint Vincent and the Grenadines": "St. Vincent and the Grenadines",
    "Saint Martin": "St. Martin (French part)",
    "United States Virgin Islands": "Virgin Islands (U.S.)",
    "Turkey": "Turkiye",
    "Czech Republic": "Czechia",
    "Cape Verde": "Cabo Verde",
    "Ivory Coast": "Cote d'Ivoire",
    "Curaçao": "Curacao",
    "Laos": "Lao PDR",
    "Brunei": "Brunei Darussalam",
    # The UK home nations are one World Bank entity; use it as a structural proxy.
    "England": "United Kingdom",
    "Scotland": "United Kingdom",
    "Wales": "United Kingdom",
    "Northern Ireland": "United Kingdom",
}


# --------------------------------------------------------------------------- #
# Small, individually-tested feature functions.
# --------------------------------------------------------------------------- #
def points_for(goals_for: int, goals_against: int) -> int:
    """Football points from one team's perspective (3 win / 1 draw / 0 loss)."""
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def rest_days(prior_dates: list[pd.Timestamp], match_date: pd.Timestamp) -> float:
    """Days since the team's most recent prior match; NaN if it has no history."""
    if not prior_dates:
        return float("nan")
    return float((pd.Timestamp(match_date) - max(prior_dates)).days)


def recent_form_points(prior_points: list[int], n: int = DEFAULT_FORM_WINDOW) -> float:
    """Mean points per game over the last ``n`` prior matches; NaN if none."""
    if not prior_points:
        return float("nan")
    window = prior_points[-n:]
    return float(np.mean(window))


def gd_momentum(prior_gd: list[int], n: int = DEFAULT_FORM_WINDOW) -> float:
    """Mean goal difference over the last ``n`` prior matches; NaN if none."""
    if not prior_gd:
        return float("nan")
    return float(np.mean(prior_gd[-n:]))


def host_flag(team: str) -> float:
    """1.0 if ``team`` is a 2026 host, else 0.0."""
    return 1.0 if team in config.HOST_TEAMS else 0.0


# --------------------------------------------------------------------------- #
# Feature context (shared lookups) + matrix builders.
# --------------------------------------------------------------------------- #
@dataclass
class FeatureContext:
    """Fitted lookups needed to featurize matches."""

    elo: EloModel
    confederations: dict[str, str]
    structural: dict[str, tuple[float, float]]  # team -> (log_population, gdp_per_capita)
    form_window: int = DEFAULT_FORM_WINDOW

    def confed(self, team: str) -> str:
        return self.confederations.get(team, config.UNKNOWN_CONFEDERATION)

    def struct(self, team: str) -> tuple[float, float]:
        return self.structural.get(team, (float("nan"), float("nan")))


def build_structural_map(worldbank: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Map team name -> (log population, GDP per capita) via exact name + aliases."""
    by_country = worldbank.set_index("country")
    out: dict[str, tuple[float, float]] = {}
    all_teams = set(loaders.load_confederations()) | config.HOST_TEAMS
    for team in all_teams:
        wb_name = WB_ALIASES.get(team, team)
        if wb_name in by_country.index:
            row = by_country.loc[wb_name]
            pop = float(row["population"]) if pd.notna(row["population"]) else float("nan")
            gdp = float(row["gdp_per_capita"]) if pd.notna(row["gdp_per_capita"]) else float("nan")
            out[team] = (float(np.log(pop)) if pop > 0 else float("nan"), gdp)
    return out


def make_context(
    results: pd.DataFrame,
    worldbank: pd.DataFrame | None = None,
    form_window: int = DEFAULT_FORM_WINDOW,
) -> FeatureContext:
    """Fit Elo on ``results`` and assemble the shared feature lookups."""
    elo = EloModel().fit(results)
    confeds = loaders.load_confederations()
    wb = worldbank if worldbank is not None else loaders.load_worldbank()
    structural = build_structural_map(wb)
    return FeatureContext(
        elo=elo, confederations=confeds, structural=structural, form_window=form_window
    )


def _row_features(
    home: str,
    away: str,
    match_date: pd.Timestamp,
    neutral: bool,
    tournament: str,
    ctx: FeatureContext,
    hist_dates: dict[str, list],
    hist_points: dict[str, list],
    hist_gd: dict[str, list],
) -> dict[str, float | str]:
    """Build one feature row from already-accumulated prior-only history."""
    elo_home = ctx.elo.rating_before(home, match_date)
    elo_away = ctx.elo.rating_before(away, match_date)
    log_pop_h, gdp_h = ctx.struct(home)
    log_pop_a, gdp_a = ctx.struct(away)
    return {
        "elo_diff": elo_home - elo_away,
        "elo_home": elo_home,
        "elo_away": elo_away,
        "home_advantage": 0.0 if neutral else 1.0,
        "host_home": host_flag(home),
        "host_away": host_flag(away),
        "rest_days_home": rest_days(hist_dates.get(home, []), match_date),
        "rest_days_away": rest_days(hist_dates.get(away, []), match_date),
        "form_home": recent_form_points(hist_points.get(home, []), ctx.form_window),
        "form_away": recent_form_points(hist_points.get(away, []), ctx.form_window),
        "gd_momentum_home": gd_momentum(hist_gd.get(home, []), ctx.form_window),
        "gd_momentum_away": gd_momentum(hist_gd.get(away, []), ctx.form_window),
        "importance_tier": float(elo_mod.importance_tier(tournament)),
        "log_pop_home": log_pop_h,
        "log_pop_away": log_pop_a,
        "gdp_home": gdp_h,
        "gdp_away": gdp_a,
        "confed_home": ctx.confed(home),
        "confed_away": ctx.confed(away),
    }


def outcome_label(home_score: int, away_score: int) -> int:
    """0 = home win, 1 = draw, 2 = away win (matches OUTCOME_CLASSES order)."""
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


def build_training_matrix(
    results: pd.DataFrame, ctx: FeatureContext
) -> tuple[pd.DataFrame, pd.Series]:
    """Featurize every match in ``results`` via a single chronological, leakage-safe
    pass; returns ``(X, y)`` with ``X`` columns == :data:`FEATURE_NAMES`."""
    df = results.sort_values("date", kind="stable")
    hist_dates: dict[str, list] = {}
    hist_points: dict[str, list] = {}
    hist_gd: dict[str, list] = {}

    rows: list[dict] = []
    labels: list[int] = []
    for r in df.itertuples(index=False):
        rows.append(
            _row_features(
                r.home_team, r.away_team, pd.Timestamp(r.date), bool(r.neutral),
                r.tournament, ctx, hist_dates, hist_points, hist_gd,
            )
        )
        labels.append(outcome_label(r.home_score, r.away_score))
        # Append *after* featurizing so the current match never leaks into itself.
        for team, gf, ga in (
            (r.home_team, r.home_score, r.away_score),
            (r.away_team, r.away_score, r.home_score),
        ):
            hist_dates.setdefault(team, []).append(pd.Timestamp(r.date))
            hist_points.setdefault(team, []).append(points_for(gf, ga))
            hist_gd.setdefault(team, []).append(gf - ga)

    X = pd.DataFrame(rows, columns=list(FEATURE_NAMES))
    y = pd.Series(labels, name="outcome")
    return X, y


def goal_targets(results: pd.DataFrame) -> pd.DataFrame:
    """Aligned (home_score, away_score) goal targets in the same row order as
    :func:`build_training_matrix` (date-sorted) — used by the Poisson model."""
    df = results.sort_values("date", kind="stable")
    return df[["home_score", "away_score"]].reset_index(drop=True).astype(int)


def build_features(match: Match, history: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Build a single-row feature frame for an unplayed :class:`Match`.

    ``history`` is the full results frame; only rows strictly before the match
    date are used, so the result is leakage-safe by construction.
    """
    md = pd.Timestamp(match.match_date)
    prior = history[history["date"] < md]
    hist_dates: dict[str, list] = {}
    hist_points: dict[str, list] = {}
    hist_gd: dict[str, list] = {}
    for team in (match.home_team, match.away_team):
        sub = prior[(prior.home_team == team) | (prior.away_team == team)].sort_values("date")
        for r in sub.itertuples(index=False):
            is_home = r.home_team == team
            gf = r.home_score if is_home else r.away_score
            ga = r.away_score if is_home else r.home_score
            hist_dates.setdefault(team, []).append(pd.Timestamp(r.date))
            hist_points.setdefault(team, []).append(points_for(gf, ga))
            hist_gd.setdefault(team, []).append(gf - ga)

    tournament = str(match.context.get("tournament", "FIFA World Cup"))
    row = _row_features(
        match.home_team, match.away_team, md, match.neutral,
        tournament, ctx, hist_dates, hist_points, hist_gd,
    )
    return pd.DataFrame([row], columns=list(FEATURE_NAMES))
