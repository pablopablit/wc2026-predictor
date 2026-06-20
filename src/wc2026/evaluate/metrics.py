"""Metrics, calibration, and the temporal (walk-forward) backtest.

Primary metrics are probabilistic — multiclass **log-loss** and **Brier score** —
with accuracy reported as a secondary, plus a reliability/calibration curve.

The backtest is strictly temporal: train up to a cutoff, validate on a later
window, test on the most recent window; never shuffle across time. A
``TimeSeriesSplit``-style generator walks the cutoff forward so reported numbers
reflect genuine out-of-sample, future-facing performance.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Phase 5 implements: log_loss/brier/accuracy wrappers, reliability_curve(),
# walk_forward_splits(), backtest(), Scorecard.
