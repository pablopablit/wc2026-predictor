"""Read raw files into tidy, validated DataFrames.

Loaders are the only place raw files are parsed. They return canonical-schema
DataFrames (validated by :mod:`wc2026.data.schema`) so that everything downstream
— Elo, features, evaluation — can assume clean, deduplicated, well-typed data.
A missing cache fails with a clear pointer to ``wc2026 data``.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

from wc2026 import config
from wc2026.data import schema

logger = logging.getLogger(__name__)


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing data file: {path}\n"
            "Run `wc2026 data` (or `make data`) to download/build the sources first."
        )
    return path


def drop_unplayed(df: pd.DataFrame) -> pd.DataFrame:
    """Drop scheduled-but-unplayed fixtures (missing scores) — training needs
    only matches with a known result. Returns a fresh, re-indexed frame."""
    played = df.dropna(subset=["home_score", "away_score"])
    if (dropped := len(df) - len(played)) > 0:
        logger.info("results.csv: dropped %d scheduled/unplayed fixture(s).", dropped)
    return played.reset_index(drop=True)


def load_results() -> pd.DataFrame:
    """Load martj42 ``results.csv`` as a validated, played-only, date-sorted frame."""
    path = _require(config.RAW_DIR / "results.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df = drop_unplayed(df)
    df = schema.validate_results(df)
    dup_keys = int(df.duplicated(["date", "home_team", "away_team"]).sum())
    if dup_keys:
        logger.info("results.csv: %d same-day repeat fixture(s) retained.", dup_keys)
    return df.sort_values("date").reset_index(drop=True)


def load_shootouts() -> pd.DataFrame:
    """Load martj42 ``shootouts.csv`` as a validated, date-sorted DataFrame."""
    path = _require(config.RAW_DIR / "shootouts.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df = schema.validate_shootouts(df)
    return df.sort_values("date").reset_index(drop=True)


def load_worldbank() -> pd.DataFrame:
    """Load the cached World Bank structural snapshot, validated."""
    path = _require(config.WORLDBANK_PATH)
    df = pd.read_csv(path)
    return schema.validate_worldbank(df)


@lru_cache(maxsize=1)
def load_confederations() -> dict[str, str]:
    """Return a ``team -> confederation`` map (committed reference data).

    Teams absent from the file map to ``config.UNKNOWN_CONFEDERATION``; callers
    look up with ``.get(team, config.UNKNOWN_CONFEDERATION)``.
    """
    path = _require(config.CONFEDERATIONS_PATH)
    raw = json.loads(path.read_text())
    return {
        team: confed
        for confed, teams in raw.items()
        if not confed.startswith("_")
        for team in teams
    }
