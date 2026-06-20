"""Download, cache, and version-stamp the external data sources.

Sources (see ``sources.md`` for URLs, licenses, columns, retrieval dates):

* martj42/international_results — ``results.csv`` (backbone training data),
  plus ``shootouts.csv`` and ``goalscorers.csv`` companion files.
* World Bank Open Data API — population (``SP.POP.TOTL``) and GDP per capita
  (``NY.GDP.PCAP.CD``), used as structural priors for the Poisson model. Fetched
  via plain ``requests`` against the public JSON API and cached as CSV; values
  are frozen to a pre-2026 snapshot to avoid leakage.

Design notes for later phases:
* All downloads land in ``config.RAW_DIR`` and are treated as immutable.
* Each fetch records the file hash + retrieval date into
  ``config.MANIFEST_PATH`` so any prediction is traceable to its data state.
* Network calls are gated behind an explicit confirmation in the CLI.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Phase 2 implements: refresh_sources(), _download(), _stamp_manifest().
