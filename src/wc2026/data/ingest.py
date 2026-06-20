"""Download, cache, and version-stamp the external data sources.

Sources (see ``sources.md`` for URLs, licenses, columns, retrieval dates):

* martj42/international_results — ``results.csv`` (backbone training data),
  plus ``shootouts.csv`` and ``goalscorers.csv`` companion files.
* World Bank Open Data API — population (``SP.POP.TOTL``) and GDP per capita
  (``NY.GDP.PCAP.CD``), used as structural priors for the Poisson model. Fetched
  via plain ``requests`` against the public JSON API and cached as CSV; values
  are frozen to a pre-2026 snapshot to avoid leakage.

Design:
* All downloads land in ``config.RAW_DIR`` and are treated as immutable.
* Each fetch records the file hash + size + retrieval date + URL into
  ``config.MANIFEST_PATH`` so any prediction is traceable to its data state.
* Network calls are gated behind an explicit confirmation in the CLI; the
  functions here perform the work and return the manifest.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from wc2026 import config

logger = logging.getLogger(__name__)

_MARTJ42_BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
RESULTS_URL = f"{_MARTJ42_BASE}/results.csv"
SHOOTOUTS_URL = f"{_MARTJ42_BASE}/shootouts.csv"
GOALSCORERS_URL = f"{_MARTJ42_BASE}/goalscorers.csv"  # ignored in v1, fetched for completeness

WORLDBANK_BASE = "https://api.worldbank.org/v2"
#: Freeze the structural snapshot to pre-2026 (no leakage).
WORLDBANK_DATE_RANGE = "1960:2024"

_HTTP_TIMEOUT = 60
_CHUNK = 1 << 16


@dataclass(frozen=True)
class ManifestEntry:
    """A single source's provenance record."""

    name: str
    url: str
    path: str  # relative to the project root
    sha256: str
    bytes: int
    rows: int | None
    retrieved_at: str  # ISO-8601 UTC


def file_sha256(path: Path) -> str:
    """Stream a SHA-256 of a file (works for large files without loading them)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` (atomic via a temp file)."""
    logger.info("Downloading %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=_HTTP_TIMEOUT) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                fh.write(chunk)
    tmp.replace(dest)


def _count_rows(path: Path) -> int | None:
    try:
        return int(pd.read_csv(path).shape[0])
    except Exception:  # pragma: no cover - row count is best-effort metadata
        return None


def _entry(name: str, url: str, path: Path) -> ManifestEntry:
    return ManifestEntry(
        name=name,
        url=url,
        path=str(path.relative_to(config.PROJECT_ROOT)),
        sha256=file_sha256(path),
        bytes=path.stat().st_size,
        rows=_count_rows(path),
        retrieved_at=_now(),
    )


def _get_json(url: str, retries: int = 3, backoff: float = 1.5) -> object:
    """GET JSON with retries — the World Bank API returns spurious 400s under load."""
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=_HTTP_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last = exc
            logger.warning("GET %s failed (attempt %d/%d): %s", url, attempt, retries, exc)
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise RuntimeError(f"GET {url} failed after {retries} attempts") from last


def fetch_worldbank(dest: Path) -> None:
    """Fetch most-recent population & GDP-per-capita per country and cache as CSV."""
    frames = []
    for indicator in config.WORLDBANK_INDICATORS:
        url = (
            f"{WORLDBANK_BASE}/country/all/indicator/{indicator}"
            f"?format=json&per_page=20000&mrnev=1&date={WORLDBANK_DATE_RANGE}"
        )
        logger.info("Fetching World Bank indicator %s", indicator)
        payload = _get_json(url)
        if not (isinstance(payload, list) and len(payload) >= 2 and payload[1]):
            raise RuntimeError(f"Unexpected World Bank response for {indicator}: {payload!r:.200}")
        records: list[dict] = payload[1]
        rows = [
            {
                "country": rec["country"]["value"],
                "iso3": rec.get("countryiso3code") or None,
                "year": int(rec["date"]) if rec.get("date") else None,
                indicator: rec["value"],
            }
            for rec in records
            if rec.get("value") is not None
        ]
        frames.append(pd.DataFrame(rows))

    pop, gdp = frames
    merged = pop.merge(gdp, on=["country", "iso3"], how="outer", suffixes=("_pop", "_gdp"))
    merged["year"] = merged[["year_pop", "year_gdp"]].max(axis=1)
    out = merged.rename(
        columns={
            config.WORLDBANK_INDICATORS[0]: "population",
            config.WORLDBANK_INDICATORS[1]: "gdp_per_capita",
        }
    )[["country", "iso3", "year", "population", "gdp_per_capita"]]
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)


def refresh_sources(force: bool = False) -> dict:
    """Download/refresh all sources, write the manifest, and return it.

    ``force=False`` skips files that already exist (idempotent re-runs); the
    manifest is always rewritten so hashes/timestamps stay current.
    """
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    plan: list[tuple[str, str, Path, str]] = [
        ("results", RESULTS_URL, config.RAW_DIR / "results.csv", "download"),
        ("shootouts", SHOOTOUTS_URL, config.RAW_DIR / "shootouts.csv", "download"),
        ("worldbank", "", config.WORLDBANK_PATH, "worldbank"),
    ]

    entries: list[ManifestEntry] = []
    for name, url, path, kind in plan:
        if path.exists() and not force:
            logger.info("%s already cached at %s (use force=True to refresh).", name, path)
        elif kind == "download":
            _download(url, path)
        elif kind == "worldbank":
            fetch_worldbank(path)
        url_for_entry = url or f"{WORLDBANK_BASE} ({', '.join(config.WORLDBANK_INDICATORS)})"
        entries.append(_entry(name, url_for_entry, path))

    manifest = {
        "generated_at": _now(),
        "sources": {e.name: asdict(e) for e in entries},
    }
    config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Wrote manifest with %d source(s) to %s", len(entries), config.MANIFEST_PATH)
    return manifest


def load_manifest() -> dict | None:
    """Return the data manifest if it exists, else ``None``."""
    if config.MANIFEST_PATH.exists():
        return json.loads(config.MANIFEST_PATH.read_text())
    return None


def manifest_hash() -> str | None:
    """A stable short hash over all source hashes — bundled into model metadata."""
    manifest = load_manifest()
    if not manifest:
        return None
    digest = hashlib.sha256()
    for name in sorted(manifest["sources"]):
        digest.update(manifest["sources"][name]["sha256"].encode())
    return digest.hexdigest()[:16]
