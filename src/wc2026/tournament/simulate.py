"""Monte Carlo tournament simulation — the main expansion seam.

For each of N simulations (default 10k, configurable and seeded):

1. sample every group match from the model's predicted W/D/L probabilities;
2. resolve groups via :mod:`wc2026.tournament.groups` (full tiebreakers +
   best-third-placed selection);
3. build the Round of 32 from the official seeding map (loaded from data, not
   hard-coded in code, so it stays editable);
4. play the knockouts, resolving draws with a penalty coin-flip weighted by Elo
   (a deliberately crude v1 placeholder);
5. tally how often each team reaches each stage / lifts the cup.

Returns a sorted DataFrame of advancement probabilities. Kept modular so richer
match models or draw-resolution can slot in without rewriting the driver.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Phase 7 implements: simulate_tournament(n, model, ...), _play_match(),
# _play_knockouts(), advancement_table().
