"""Read raw files into tidy, validated DataFrames.

Loaders are the only place raw files are parsed. They return canonical-schema
DataFrames (validated by :mod:`wc2026.data.schema`) so that everything downstream
— Elo, features, evaluation — can assume clean, deduplicated, well-typed data.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Phase 2 implements: load_results(), load_shootouts(), load_worldbank(),
# load_wc2026_groups(), load_wc2026_fixtures(), load_bracket_map().
