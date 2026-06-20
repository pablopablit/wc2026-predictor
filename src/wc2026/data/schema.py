"""Column contracts and validation for incoming data.

Validation must fail *loudly* with a helpful message: assert expected column
names and dtypes, reject impossible scores (negative goals), and deduplicate
exact-duplicate rows. Implemented with ``pandera`` in Phase 2.

The canonical results schema (from martj42):
    date         datetime64[ns]
    home_team    str
    away_team    str
    home_score   int (>= 0)
    away_score   int (>= 0)
    tournament   str
    city         str (nullable)
    country      str (nullable)
    neutral      bool
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Phase 2 implements: RESULTS_SCHEMA, SHOOTOUTS_SCHEMA, validate_results().
