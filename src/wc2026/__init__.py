"""wc2026-predictor — a 2026 FIFA World Cup match-prediction framework.

Public surface (kept tiny in v1):

    >>> from wc2026 import Predictor, Match
    >>> model = Predictor.load()            # once a model is trained
    >>> model.predict_proba(match)          # win / draw / loss probabilities

Submodules:
    config      paths, seed, fixed 2026 format constants
    data        ingestion, loaders, schema validation
    features    leakage-safe Elo timeline + feature matrix builder
    models      Predictor interface, baseline, gradient-boosted classifier
    evaluate    metrics + temporal backtest
    tournament  group standings/tiebreakers + Monte Carlo simulation
    cli         the ``wc2026`` command-line entry point
"""

from __future__ import annotations

__version__ = "0.1.0"

from wc2026.models.base import Match, Predictor

__all__ = ["Match", "Predictor", "__version__"]
