"""Central configuration: paths, the deterministic seed, and the fixed 2026 format.

Everything here is intentionally constant and import-safe (no side effects beyond
directory path computation). The 2026 World Cup format is hard-coded per the build
brief — it is a rule of the tournament, not something to infer at runtime.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
#: Single source of truth for randomness, threaded through numpy, the model, and
#: the Monte Carlo simulator so that the same inputs always yield the same output.
RANDOM_SEED = 20260611  # World Cup 2026 opening day.

# --------------------------------------------------------------------------- #
# Paths (no hard-coded absolutes; everything is relative to the project root
# unless overridden by an environment variable).
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("WC2026_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = Path(os.environ.get("WC2026_MODELS_DIR", PROJECT_ROOT / "models"))

#: Records the source file hashes + retrieval dates so a prediction can be traced
#: back to the exact data state it came from.
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"

# Hard-coded 2026 fixture/group files (small, committed to git).
WC2026_GROUPS_PATH = RAW_DIR / "wc2026_groups.json"
WC2026_FIXTURES_PATH = RAW_DIR / "wc2026_fixtures.csv"
#: Bracket mapping (which group position meets which) lives in data, not code.
WC2026_BRACKET_MAP_PATH = RAW_DIR / "wc2026_bracket_map.json"

# --------------------------------------------------------------------------- #
# 2026 tournament format (FIXED — do not infer at runtime).
# --------------------------------------------------------------------------- #
NUM_TEAMS = 48
NUM_GROUPS = 12
TEAMS_PER_GROUP = 4
GROUP_NAMES: tuple[str, ...] = tuple("ABCDEFGHIJKL")  # Groups A–L.
MATCHES_PER_TEAM_GROUP_STAGE = 3

#: Top 2 of each group (24) + 8 best third-placed teams = 32 → Round of 32.
TEAMS_ADVANCING_TOP_TWO = NUM_GROUPS * 2  # 24
BEST_THIRD_PLACED_ADVANCING = 8
R32_TEAMS = TEAMS_ADVANCING_TOP_TWO + BEST_THIRD_PLACED_ADVANCING  # 32

#: Knockout stages, in order.
KNOCKOUT_STAGES: tuple[str, ...] = (
    "R32",  # Round of 32
    "R16",  # Round of 16
    "QF",  # Quarter-finals
    "SF",  # Semi-finals
    "F",  # Final
)

#: Group tiebreakers, applied in this exact order (brief §1).
TIEBREAKERS: tuple[str, ...] = (
    "points",
    "goal_difference",
    "goals_scored",
    "head_to_head",
    "fair_play",  # fewest disciplinary points (cards)
    "drawing_of_lots",
)

#: Hosts and their pre-assigned groups (brief §1).
HOST_GROUPS: dict[str, str] = {
    "United States": "D",
    "Mexico": "A",
    "Canada": "B",
}
HOST_TEAMS: frozenset[str] = frozenset(HOST_GROUPS)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_LEVEL = os.environ.get("WC2026_LOG_LEVEL", "INFO").upper()


def configure_logging(level: str | None = None) -> None:
    """Configure root logging for CLI entry points (idempotent-ish).

    Library modules use ``logging.getLogger(__name__)`` and never call this; only
    the CLI configures handlers, so importing the package stays side-effect free.
    """
    logging.basicConfig(
        level=level or LOG_LEVEL,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
