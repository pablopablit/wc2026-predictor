"""Leakage-safe football Elo rating timeline.

Standard football Elo computed from the martj42 results themselves (no external
Elo file in v1 — deriving it is more robust and keeps the pipeline self-contained):

* K-factor scaled by match importance (tournament tier) and goal margin.
* Home-advantage term (suppressed on neutral grounds).
* Optional decay of ratings toward the mean between matches.

The central guarantee is :meth:`EloModel.rating_before` — the rating of a team as
it stood *strictly before* a given date — so any historical match can be
featurized without leaking its own (or any future) result.

Sanity properties asserted in ``tests/test_elo.py``:
* a team that wins gains rating, its opponent loses it;
* total rating is approximately conserved across a match (zero-sum exchange).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Phase 3 implements: EloModel(fit/rating_before/rating_timeline),
# K-factor and expected-score helpers.
