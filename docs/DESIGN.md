# Design & methodology

This document records *why* the model is shaped the way it is. It is influenced by
[playmobil/worldcup-forecast](https://github.com/playmobil/worldcup-forecast),
adapted to this codebase and the build brief.

## Objective

Calibrated win/draw/loss probabilities (and a derived scoreline) for 2026 World
Cup matches, plus a Monte-Carlo simulation of the whole tournament. **Calibration
is the priority over raw accuracy** — the probabilities feed a simulator, so being
well-calibrated matters more than topping an accuracy leaderboard.

## Model

A **hierarchical Bayesian Poisson** goal model (PyMC):

- Latent per-team **attack** and **defence** strengths, **partially pooled** so
  data-rich teams are driven by results and data-scarce teams shrink toward
  priors.
- Global **home-advantage** term, suppressed on neutral ground.
- `log λ_home = μ + atk_home − def_away + home_adv`, symmetrically for away.
- **Priors are set by reasoning, not fit to the ~36 historical World Cups.** This
  is the central anti-overfitting decision. Priors are informed by:
  - leakage-safe **Elo** (K=40, scaled by importance & goal margin), and
  - **structural** indicators (World Bank log-population and GDP per capita; GDP
    enters as an inverted-U).

### From goals to outcomes

The posterior produces a **score grid** — a matrix of `P(home goals = i, away
goals = j)`. Summing the lower triangle / diagonal / upper triangle gives
`P(home win) / P(draw) / P(away win)`. The modal cell is the most-likely
scoreline. A **temperature calibration** and a **neutral-site draw adjustment**
refine the probabilities. v1 treats the two goal counts as independent (single
Poisson per side); bivariate-Poisson / Dixon-Coles correlation is a documented,
to-be-validated extension.

### Baseline

An **Elo-difference + home-advantage logit** (`models/baseline.py`) is the bar the
Bayesian model must clear on log-loss / Brier. If the richer model can't beat it,
it isn't earning its complexity.

## Data discipline

- **No leakage:** every feature is computable strictly before kickoff;
  `EloModel.rating_before(team, date)` enforces this and a dedicated test fails if
  any feature peeks at the result or the future. Structural data is frozen to a
  pre-2026 snapshot.
- **Independence principle:** betting/market odds are **never** a model input
  (`config.MARKET_ODDS_AS_INPUT = False`) — only ever an optional benchmark.
- **Temporal evaluation:** walk-forward splits, never shuffled across time. Full
  NUTS sampling is reserved for the final fit; the backtest uses a fast
  MAP/variational fit so it stays runnable.
- **Reproducible:** one `RANDOM_SEED`; source hashes + retrieval dates recorded in
  `data/processed/manifest.json`; each model bundles a `model_meta.json` sidecar.

## Open questions / future work

- Does the structural prior actually help out-of-sample, or does Elo subsume it?
  (Record the answer in `FINDINGS.md`.)
- Bivariate-Poisson / Dixon-Coles: the reference found it over-fit sparse data —
  re-test on our larger international history before adopting.
- Model-averaging the Bayesian and Elo-logit outputs to reduce variance.
