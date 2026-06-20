# wc2026-predictor

A small but production-shaped machine-learning framework that predicts **2026 FIFA
World Cup** match outcomes and simulates the tournament forward.

> **Status:** v1, built in phases. Phase 1 (scaffold + toolchain) is complete; the
> data, Elo, feature, model, and simulation layers are filled in across later
> phases. Module docstrings note which phase implements each piece.

## What it does (v1 scope)

Given two national teams and minimal context (date, neutral/host), it produces:

1. **Win / Draw / Loss** probabilities for the home/first team (a calibrated
   3-class classifier).
2. A derived **most-likely scoreline** (naive Poisson helper in v1 — a documented
   placeholder for a future bivariate-Poisson / Dixon-Coles model).
3. A **confidence / uncertainty** indicator.

It also predicts the full 2026 group-stage fixture table and **Monte-Carlo
simulates** the tournament (group stage → knockouts) to report each team's
probability of advancing and of winning the cup.

The 2026 format is hard-coded as a rule, not inferred: 48 teams, 12 groups (A–L)
of 4, top 2 per group plus the 8 best third-placed teams into a Round of 32, then
R16 → QF → SF → Final. Hosts: USA (Group D), Mexico (Group A), Canada (Group B).

## Quickstart (5 commands)

```bash
make setup       # sync the venv to the lock file + install the package
make data        # download/refresh sources, build processed datasets (asks first)
make train       # train, evaluate, persist; prints the scorecard
make predict ARGS="--home Ecuador --away Argentina --date 2026-06-20"
make simulate ARGS="--n 10000"
```

The CLI is also directly available as `wc2026` inside the venv
(`.venv/bin/wc2026 --help`), and the Python API is `from wc2026 import Predictor`.

## Toolchain

- **Python 3.13** managed via **pyenv** (`.python-version` pins `3.13.14`).
- Project-local **`.venv`**; dependencies declared in `pyproject.toml` and pinned
  in `requirements.lock` via **pip-tools** (`make lock`).
- **ruff** (lint + format), **mypy** (types), **pytest** (tests).

## Data sources & licenses

See [`sources.md`](sources.md) for full details. Backbone training data is
[martj42/international_results](https://github.com/martj42/international_results)
(`results.csv`, `shootouts.csv`). Team strength is an **Elo rating derived from
those results** (no external Elo file in v1). The 2026 groups, fixtures, and
bracket map are committed as small versioned files under `data/raw/`.

## Model

- **Baseline** (`models/baseline.py`): Elo difference + home advantage — the bar
  to beat.
- **Classifier** (`models/classifier.py`): scikit-learn
  `HistGradientBoostingClassifier` with probability calibration, behind the same
  `Predictor` interface so XGBoost/LightGBM can be swapped in later.
- Evaluated primarily on **log-loss** and **Brier score** (calibration matters
  most for a probabilistic sim), accuracy secondary, with a reliability curve.
- Trained with a **temporal** (walk-forward) split — never shuffled across time.

## Data discipline

- **No leakage:** every feature is computable strictly before kickoff; the Elo
  module exposes `rating_before(team, date)`, and a dedicated test fails if any
  feature peeks at the result or the future.
- **Deterministic & seeded:** one `RANDOM_SEED` in `config.py` threaded through
  numpy, the model, and the simulator.
- **Reproducible:** source hashes + retrieval dates recorded in
  `data/processed/manifest.json`; each model bundles a `model_meta.json` sidecar.

## Known limitations (v1)

- Small, deliberately simple feature set.
- Naive Poisson scorelines (placeholder; hook left for a richer model).
- Crude knockout draw-resolution (Elo-weighted penalty coin-flip).
- No player-level data, no live updates, no betting-odds ingestion, no web UI.

## How to extend (the hooks)

- **Swap the model:** implement the `Predictor` interface
  (`fit`/`predict_proba`/`save`/`load`) in a new class — nothing else changes.
- **Add features:** add a small, tested `feature_*` function in
  `features/build.py` and append it to `FEATURE_NAMES`.
- **Richer scorelines:** replace the naive Poisson helper with a
  bivariate-Poisson / Dixon-Coles model behind the same scoreline hook.
- **Richer simulation:** the bracket mapping lives in
  `data/raw/wc2026_bracket_map.json` (editable, not code); draw-resolution and
  match sampling in `tournament/simulate.py` are isolated functions.
- **Add club/player data:** new loaders in `data/` + new features; the schema and
  manifest machinery already support versioned sources.

## Project layout

```
src/wc2026/
  config.py        paths, seed, fixed 2026 format
  data/            ingest · loaders · schema
  features/        elo · build
  models/          base (Predictor) · baseline · classifier
  evaluate/        metrics + temporal backtest
  tournament/      groups (standings/tiebreakers) · simulate (Monte Carlo)
  cli.py           the wc2026 entry point
tests/             elo · no-leakage · tournament-rules · model-smoke
```

## License

MIT (project code). Data sources retain their own licenses — see `sources.md`.
