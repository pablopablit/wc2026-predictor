"""Assemble the model feature matrix for a match — one small, tested function each.

Features (v1, deliberately small; each computable strictly pre-kickoff):

* ``elo_diff``            home Elo − away Elo (before kickoff)
* ``elo_home`` / ``elo_away``  individual ratings
* ``home_advantage``      1 for a genuine home side, 0 if neutral
* ``host_flag_home/away`` is the team a 2026 host
* ``rest_days_home/away`` days since each team's previous international
* ``form_home/away``      points in the last N internationals
* ``gd_momentum_home/away``  recent goal-difference trend
* ``confed_home/away``    confederation (categorical)
* ``importance_tier``     match-importance tier from the tournament label

The no-leakage contract (``tests/test_features_no_leakage.py``): every feature
function takes a :class:`~wc2026.models.base.Match` plus history *before* the
match date and never reads the match's own score or any future match.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Phase 4 implements: FEATURE_NAMES, build_features(match, history, elo),
# and the individual feature_* helpers.
