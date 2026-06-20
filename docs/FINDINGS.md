# Findings ledger

A running record of what we *tested* and what the evidence said — not just final
accuracy numbers. Inspired by the reference project's `FINDINGS.md`. Each entry:
hypothesis → how it was tested → verdict (✅ kept / ❌ rejected / 🔬 open).

Add an entry whenever a modeling choice is validated against the temporal
backtest (log-loss / Brier on a held-out, future-facing window).

| # | Hypothesis | Test | Verdict | Notes |
|---|------------|------|---------|-------|
| 1 | Elo-logit baseline beats an uninformed (uniform) predictor | 5-fold walk-forward backtest on 49,433 internationals (1872–2026) | ✅ | mean log-loss **0.893** vs uniform ln 3 = 1.099; Brier **0.526**; accuracy **58.8%**. This is the bar the Bayesian Poisson model must clear. |
| 2 | Bayesian Poisson beats the Elo-logit baseline | (pending Phase 5b) | 🔬 | — |

## Candidate hypotheses to test (from DESIGN.md)

- **Bayesian Poisson beats the Elo-logit baseline** on log-loss / Brier (the bar).
- **Structural priors (GDP/population) add signal** beyond Elo, out-of-sample.
- **Bivariate-Poisson / Dixon-Coles** improves over independent Poisson — the
  reference found it *over-fit*; re-test on our larger international history.
- **Temperature calibration** improves reliability without hurting accuracy.
- **Model-averaging** Bayesian + Elo-logit reduces variance.
