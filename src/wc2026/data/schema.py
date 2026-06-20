"""Column contracts and validation for incoming data (pandera 0.32, pandas API).

Validation fails *loudly* with a helpful message: expected column names and
dtypes, no impossible scores (negative goals), and exact-duplicate rows removed
before validation. Each loader runs the matching ``validate_*`` helper so that
everything downstream can assume clean, typed, deduplicated data.
"""

from __future__ import annotations

import logging

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
#: martj42 ``results.csv`` — the backbone training data.
RESULTS_SCHEMA = pa.DataFrameSchema(
    {
        "date": pa.Column("datetime64[ns]"),
        "home_team": pa.Column(str, nullable=False),
        "away_team": pa.Column(str, nullable=False),
        "home_score": pa.Column(int, pa.Check.ge(0), nullable=False),
        "away_score": pa.Column(int, pa.Check.ge(0), nullable=False),
        "tournament": pa.Column(str, nullable=False),
        "city": pa.Column(str, nullable=True),
        "country": pa.Column(str, nullable=True),
        "neutral": pa.Column(bool, nullable=False),
    },
    strict="filter",  # keep only the declared columns, in this contract
    coerce=True,
    # NB: no hard uniqueness on (date, home, away) — two teams genuinely can play
    # twice on the same date in the historical record. Exact-duplicate *rows* are
    # still dropped in `_validate`; remaining key-duplicates are logged by loaders.
)

#: martj42 ``shootouts.csv`` — used later to resolve draw-but-progressed logic.
SHOOTOUTS_SCHEMA = pa.DataFrameSchema(
    {
        "date": pa.Column("datetime64[ns]"),
        "home_team": pa.Column(str, nullable=False),
        "away_team": pa.Column(str, nullable=False),
        "winner": pa.Column(str, nullable=False),
    },
    strict="filter",
    coerce=True,
)

#: World Bank structural indicators (tidy, one row per country snapshot).
WORLDBANK_SCHEMA = pa.DataFrameSchema(
    {
        "country": pa.Column(str, nullable=False),
        "iso3": pa.Column(str, nullable=True),
        "year": pa.Column(int, pa.Check.in_range(1960, 2026)),
        "population": pa.Column(float, pa.Check.gt(0), nullable=True),
        "gdp_per_capita": pa.Column(float, pa.Check.ge(0), nullable=True),
    },
    strict="filter",
    coerce=True,
)


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _validate(df: pd.DataFrame, schema: pa.DataFrameSchema, name: str) -> pd.DataFrame:
    """Drop exact-duplicate rows, then validate; re-raise with a helpful message."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if (dropped := before - len(df)) > 0:
        logger.info("%s: dropped %d exact-duplicate row(s).", name, dropped)
    try:
        return schema.validate(df, lazy=True)
    except SchemaError as exc:  # pragma: no cover - exercised via lazy errors
        raise ValueError(f"{name} failed schema validation: {exc}") from exc
    except pa.errors.SchemaErrors as exc:
        raise ValueError(
            f"{name} failed schema validation:\n{exc.failure_cases}"
        ) from exc


def validate_results(df: pd.DataFrame) -> pd.DataFrame:
    return _validate(df, RESULTS_SCHEMA, "results.csv")


def validate_shootouts(df: pd.DataFrame) -> pd.DataFrame:
    return _validate(df, SHOOTOUTS_SCHEMA, "shootouts.csv")


def validate_worldbank(df: pd.DataFrame) -> pd.DataFrame:
    return _validate(df, WORLDBANK_SCHEMA, "worldbank_indicators.csv")
