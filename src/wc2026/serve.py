"""A tiny local HTTP API for the desktop app (stdlib only — no web deps in v1).

Loads the model + feature context once at startup, then answers JSON requests:

    GET /health                      -> {"status": "ok"}
    GET /teams                       -> {"all": [...], "wc2026": [...]}
    GET /predict?home=&away=&neutral=0|1[&date=YYYY-MM-DD]
    GET /simulate?n=2000             -> [{team, group, win, ...}, ...]

Run with ``python -m wc2026.serve [--port 8765]``. The Electron app spawns this.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from wc2026.config import configure_logging
from wc2026.data import loaders
from wc2026.features import build
from wc2026.models.base import Match

logger = logging.getLogger("wc2026.serve")


class _State:
    """Loaded-once model and lookups."""

    def __init__(self) -> None:
        from wc2026.models.poisson import BayesianPoissonPredictor

        logger.info("Loading data, Elo, and model …")
        self.results = loaders.load_results()
        self.ctx = build.make_context(self.results)
        self.model = BayesianPoissonPredictor.load()
        self.wc_teams = [t for ts in loaders.load_wc2026_groups().values() for t in ts]
        self.all_teams = sorted(self.ctx.elo.ratings())
        logger.info("Ready: %d teams, %d WC teams.", len(self.all_teams), len(self.wc_teams))

    def predict(self, home: str, away: str, neutral: bool, when: date) -> dict:
        match = Match(home, away, when, neutral=neutral, context={"tournament": "FIFA World Cup"})
        X = build.build_features(match, self.results, self.ctx)
        p = self.model.predict_proba(X)[0]
        gi, gj = self.model.score_grid(X)[1][0]
        top = float(max(p))
        conf = "high" if top >= 0.55 else "medium" if top >= 0.40 else "low"
        return {
            "home": home, "away": away, "neutral": neutral, "date": when.isoformat(),
            "p_home_win": float(p[0]), "p_draw": float(p[1]), "p_away_win": float(p[2]),
            "score": f"{gi}-{gj}", "confidence": conf,
        }

    def simulate(self, n: int) -> list[dict]:
        from wc2026.tournament.simulate import simulate_tournament

        df = simulate_tournament(model=self.model, n=n)
        return json.loads(df.to_json(orient="records"))


_STATE: _State | None = None


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # quiet default logging
        pass

    def _send(self, code: int, payload) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        assert _STATE is not None
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/health":
                self._send(200, {"status": "ok"})
            elif url.path == "/teams":
                self._send(200, {"all": _STATE.all_teams, "wc2026": _STATE.wc_teams})
            elif url.path == "/predict":
                when = date.fromisoformat(q["date"]) if q.get("date") else date(2026, 6, 15)
                self._send(200, _STATE.predict(
                    q["home"], q["away"], q.get("neutral") in ("1", "true", "on"), when
                ))
            elif url.path == "/simulate":
                n = max(100, min(int(q.get("n", 2000)), 50000))
                self._send(200, _STATE.simulate(n))
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:  # surface errors as JSON to the UI
            logger.exception("request failed")
            self._send(400, {"error": str(exc)})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="wc2026-serve")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    configure_logging()

    global _STATE
    _STATE = _State()
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    logger.info("Serving on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
