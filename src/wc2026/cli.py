"""The ``wc2026`` command-line entry point (stdlib argparse — no extra deps in v1).

Subcommands:
    data             download/refresh sources + build processed datasets
    train            train, evaluate, persist; print the scorecard
    evaluate         run the temporal backtest and print metrics
    predict          predict a single match
    predict-fixtures table of predictions for all 2026 group games
    simulate         Monte Carlo advancement probabilities

Every command supports ``--help`` and a ``--json`` flag for machine-readable
output. Command bodies are filled in across Phases 2–8; this scaffold wires the
parser and a clean dispatch table.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence

from wc2026 import __version__
from wc2026.config import configure_logging

logger = logging.getLogger("wc2026")

_NOT_YET = "This command is implemented in a later build phase."


def _cmd_data(args: argparse.Namespace) -> int:
    """Download/refresh sources, build the manifest, and print a summary."""
    from wc2026.data import ingest

    if not args.yes:
        print(
            "About to fetch data over the network:\n"
            f"  - martj42 results.csv / shootouts.csv\n"
            f"  - World Bank indicators ({', '.join(ingest.config.WORLDBANK_INDICATORS)})\n"
            "into ./data/raw/."
        )
        reply = input("Proceed? [y/N] ").strip().lower()
        if reply not in {"y", "yes"}:
            print("Aborted; no network calls made.")
            return 1

    manifest = ingest.refresh_sources(force=args.force)

    if args.json:
        print(json.dumps(manifest, indent=2))
        return 0

    print(f"\nData manifest ({manifest['generated_at']}):")
    for name, e in manifest["sources"].items():
        rows = f"{e['rows']:,} rows" if e["rows"] is not None else "n/a"
        print(f"  {name:10s} {rows:>14s}  {e['bytes'] / 1024:8.1f} KiB  sha256:{e['sha256'][:12]}…")
    print(f"\nManifest hash: {ingest.manifest_hash()}")
    return 0


def _model_factory(name: str):
    """Return a zero-arg constructor for the requested model type."""
    if name == "baseline":
        from wc2026.models.baseline import BaselinePredictor

        return BaselinePredictor
    if name == "bayesian":
        from wc2026.models.poisson import BayesianPoissonPredictor

        return BayesianPoissonPredictor
    raise ValueError(f"Unknown model '{name}'.")


def _prepare_matrix():
    """Load data, fit Elo + structural lookups, build (X, y, goals, cutoff)."""
    from wc2026.data import loaders
    from wc2026.features import build

    results = loaders.load_results()
    ctx = build.make_context(results)
    X, y = build.build_training_matrix(results, ctx)
    goals = build.goal_targets(results)
    cutoff = results["date"].max().date()
    return X, y, goals, cutoff


def _fit_model(factory, X, y, goals):
    """Fit a fresh model on all data, passing goals when the model needs them."""
    model = factory()
    if model.requires_goals:
        return model.fit(X, y, goals=goals)
    return model.fit(X, y)


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Walk-forward temporal backtest; print the scorecard."""
    import json as _json

    from wc2026.evaluate.metrics import backtest

    X, y, goals, _ = _prepare_matrix()
    card = backtest(_model_factory(args.model), X, y, goals=goals, n_splits=args.splits)
    if args.json:
        print(_json.dumps(card.to_dict(), indent=2))
    else:
        print(card)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Backtest, fit the final model on all data, persist with metadata."""
    import json as _json

    from wc2026.data import ingest
    from wc2026.evaluate.metrics import backtest

    X, y, goals, cutoff = _prepare_matrix()
    factory = _model_factory(args.model)

    card = backtest(factory, X, y, goals=goals, n_splits=args.splits)
    model = _fit_model(factory, X, y, goals)
    model.build_meta(
        training_cutoff=cutoff,
        scorecard=card.mean(),
        manifest_hash=ingest.manifest_hash(),
    )
    path = model.save()

    if args.json:
        print(_json.dumps({"saved": str(path), **card.to_dict()}, indent=2))
    else:
        print(card)
        print(f"\nTrained on {len(X):,} matches up to {cutoff}. Saved model → {path}")
    return 0


def _confidence(proba) -> str:
    """A simple confidence label from the top outcome probability."""
    top = float(max(proba))
    if top >= 0.55:
        return "high"
    if top >= 0.40:
        return "medium"
    return "low"


def _predictor_and_context(model_name: str):
    """Load a model and assemble the feature context + history (shared helper)."""
    from wc2026.data import loaders
    from wc2026.features import build

    model = _model_factory(model_name).load()
    results = loaders.load_results()
    ctx = build.make_context(results)
    return model, results, ctx


def _predict_rows(model, X) -> tuple:
    """Return (proba (n,3), scorelines or None) for a feature frame."""
    proba = model.predict_proba(X)
    scores = model.score_grid(X)[1] if hasattr(model, "score_grid") else None
    return proba, scores


def _cmd_predict(args: argparse.Namespace) -> int:
    """Predict a single match: W/D/L, most-likely scoreline, confidence."""
    import json as _json
    from datetime import date as _date

    from wc2026.features import build
    from wc2026.models.base import Match

    model, results, ctx = _predictor_and_context(args.model)
    md = _date.fromisoformat(args.date) if args.date else _date(2026, 6, 15)
    match = Match(
        args.home, args.away, md, neutral=args.neutral,
        context={"tournament": "FIFA World Cup"},
    )
    X = build.build_features(match, results, ctx)
    proba, scores = _predict_rows(model, X)
    p = proba[0]
    scoreline = f"{scores[0][0]}-{scores[0][1]}" if scores else None

    if args.json:
        print(_json.dumps({
            "home": args.home, "away": args.away, "date": md.isoformat(),
            "neutral": args.neutral, "p_home_win": float(p[0]), "p_draw": float(p[1]),
            "p_away_win": float(p[2]), "most_likely_score": scoreline,
            "confidence": _confidence(p),
        }, indent=2))
        return 0

    venue = "neutral" if args.neutral else f"{args.home} home"
    print(f"\n{args.home} vs {args.away}  ({md}, {venue})\n")
    print(f"  {args.home} win : {p[0] * 100:5.1f}%")
    print(f"  draw          : {p[1] * 100:5.1f}%")
    print(f"  {args.away} win : {p[2] * 100:5.1f}%")
    if scoreline:
        print(f"\n  most-likely scoreline : {scoreline} ({args.home}-{args.away})")
    print(f"  confidence            : {_confidence(p)}")
    return 0


def _cmd_predict_fixtures(args: argparse.Namespace) -> int:
    """Predict all 72 group-stage fixtures as a table."""
    import pandas as pd

    from wc2026.data import loaders
    from wc2026.features import build
    from wc2026.tournament.simulate import ASOF

    model, results, ctx = _predictor_and_context(args.model)
    fixtures = loaders.load_wc2026_fixtures()
    teams = [t for ts in loaders.load_wc2026_groups().values() for t in ts]
    hist = build.asof_history(results, teams, ASOF)
    pairs = [
        (r.home_team, r.away_team, bool(r.neutral)) for r in fixtures.itertuples(index=False)
    ]
    X = build.feature_rows(pairs, ASOF, ctx, hist)
    proba, scores = _predict_rows(model, X)

    rows = []
    for k, r in enumerate(fixtures.itertuples(index=False)):
        rows.append({
            "group": r.group, "home": r.home_team, "away": r.away_team,
            "P(H)%": round(float(proba[k][0]) * 100, 1),
            "P(D)%": round(float(proba[k][1]) * 100, 1),
            "P(A)%": round(float(proba[k][2]) * 100, 1),
            "score": f"{scores[k][0]}-{scores[k][1]}" if scores else "",
        })
    out = pd.DataFrame(rows)
    if args.json:
        print(out.to_json(orient="records"))
    else:
        print(f"\n2026 group-stage predictions ({args.model})\n")
        print(out.to_string(index=False))
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    """Monte Carlo the tournament; print advancement probabilities."""
    from wc2026.tournament.simulate import simulate_tournament

    df = simulate_tournament(n=args.n)
    if args.json:
        print(df.to_json(orient="records"))
        return 0

    print(f"\n2026 World Cup — advancement probabilities ({args.n:,} simulations)\n")
    pct = df.copy()
    for c in ("reach_R32", "reach_R16", "reach_QF", "reach_SF", "reach_final", "win"):
        pct[c] = (df[c] * 100).map(lambda v: f"{v:5.1f}")
    pct = pct.rename(
        columns={
            "reach_R32": "R32", "reach_R16": "R16", "reach_QF": "QF",
            "reach_SF": "SF", "reach_final": "Final", "win": "Win%",
        }
    )
    print(pct.head(24).to_string(index=False))
    return 0


def _add_json_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wc2026",
        description="Predict and simulate the 2026 FIFA World Cup.",
    )
    parser.add_argument("--version", action="version", version=f"wc2026 {__version__}")
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level (DEBUG/INFO/WARNING/ERROR). Defaults to WC2026_LOG_LEVEL.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("data", help="Download/refresh sources and build datasets.")
    p_data.add_argument(
        "--yes", action="store_true", help="Skip the confirmation before network calls."
    )
    p_data.add_argument(
        "--force", action="store_true", help="Re-download even if files are already cached."
    )
    _add_json_flag(p_data)

    p_train = sub.add_parser("train", help="Train, evaluate, and persist the model.")
    p_train.add_argument(
        "--model", choices=("baseline", "bayesian"), default="bayesian", help="Model to train."
    )
    p_train.add_argument("--splits", type=int, default=5, help="Backtest folds.")
    _add_json_flag(p_train)

    p_eval = sub.add_parser("evaluate", help="Run the temporal backtest.")
    p_eval.add_argument(
        "--model", choices=("baseline", "bayesian"), default="bayesian", help="Model to evaluate."
    )
    p_eval.add_argument("--splits", type=int, default=5, help="Backtest folds.")
    _add_json_flag(p_eval)

    p_pred = sub.add_parser("predict", help="Predict a single match.")
    p_pred.add_argument("--home", required=True, help="Home / first team.")
    p_pred.add_argument("--away", required=True, help="Away / second team.")
    p_pred.add_argument("--date", default=None, help="Match date (YYYY-MM-DD).")
    p_pred.add_argument(
        "--neutral", action="store_true", help="Played on neutral ground."
    )
    p_pred.add_argument(
        "--model", choices=("bayesian", "baseline"), default="bayesian", help="Model to use."
    )
    _add_json_flag(p_pred)

    p_fix = sub.add_parser(
        "predict-fixtures", help="Predict all 2026 group-stage fixtures."
    )
    p_fix.add_argument(
        "--model", choices=("bayesian", "baseline"), default="bayesian", help="Model to use."
    )
    _add_json_flag(p_fix)

    p_sim = sub.add_parser("simulate", help="Monte Carlo advancement probabilities.")
    p_sim.add_argument(
        "--n", type=int, default=10_000, help="Number of simulations (default 10000)."
    )
    _add_json_flag(p_sim)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    # Dispatch table; command bodies are filled in across Phases 2–8.
    handlers = {
        "data": _cmd_data,
        "train": _cmd_train,
        "evaluate": _cmd_evaluate,
        "simulate": _cmd_simulate,
        "predict": _cmd_predict,
        "predict-fixtures": _cmd_predict_fixtures,
    }
    handler = handlers.get(args.command)
    if handler is not None:
        return handler(args)

    logger.info("Command '%s' selected.", args.command)
    print(f"[wc2026] '{args.command}' — {_NOT_YET}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
