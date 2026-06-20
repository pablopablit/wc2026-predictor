# Findings ledger

A running record of what we *tested* and what the evidence said — not just final
accuracy numbers. Inspired by the reference project's `FINDINGS.md`. Each entry:
hypothesis → how it was tested → verdict (✅ kept / ❌ rejected / 🔬 open).

Add an entry whenever a modeling choice is validated against the temporal
backtest (log-loss / Brier on a held-out, future-facing window).

| # | Hypothesis | Test | Verdict | Notes |
|---|------------|------|---------|-------|
| — | _(none yet — populated from Phase 5 onward)_ | | 🔬 | Baseline and Bayesian Poisson not yet trained. |

## Candidate hypotheses to test (from DESIGN.md)

- **Bayesian Poisson beats the Elo-logit baseline** on log-loss / Brier (the bar).
- **Structural priors (GDP/population) add signal** beyond Elo, out-of-sample.
- **Bivariate-Poisson / Dixon-Coles** improves over independent Poisson — the
  reference found it *over-fit*; re-test on our larger international history.
- **Temperature calibration** improves reliability without hurting accuracy.
- **Model-averaging** Bayesian + Elo-logit reduces variance.
