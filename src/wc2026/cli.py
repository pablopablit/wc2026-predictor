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
    _add_json_flag(p_train)

    p_eval = sub.add_parser("evaluate", help="Run the temporal backtest.")
    _add_json_flag(p_eval)

    p_pred = sub.add_parser("predict", help="Predict a single match.")
    p_pred.add_argument("--home", required=True, help="Home / first team.")
    p_pred.add_argument("--away", required=True, help="Away / second team.")
    p_pred.add_argument("--date", default=None, help="Match date (YYYY-MM-DD).")
    p_pred.add_argument(
        "--neutral", action="store_true", help="Played on neutral ground."
    )
    _add_json_flag(p_pred)

    p_fix = sub.add_parser(
        "predict-fixtures", help="Predict all 2026 group-stage fixtures."
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
    }
    handler = handlers.get(args.command)
    if handler is not None:
        return handler(args)

    logger.info("Command '%s' selected.", args.command)
    print(f"[wc2026] '{args.command}' — {_NOT_YET}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
