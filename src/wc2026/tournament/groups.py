"""Group standings, the exact 2026 tiebreakers, and best-third-placed ranking.

Computes standings from a set of played group matches and breaks ties in the
order fixed in :data:`wc2026.config.TIEBREAKERS`:
points → goal difference → goals scored → head-to-head → fair-play → lots.

Also ranks the twelve third-placed teams to select the eight that advance to the
Round of 32. This logic is fiddly and a common bug source, so it is unit-tested
against hand-constructed group tables (``tests/test_tournament_rules.py``).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Phase 6 implements: compute_standings(), rank_group(), best_third_placed().
