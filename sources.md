# Data sources

Each source is downloaded, cached under `data/raw/`, and version-stamped (file
hash + retrieval date) in `data/processed/manifest.json`. Retrieval dates are
filled in when `wc2026 data` is first run (Phase 2).

---

## 1. International results (backbone training data)

- **Repository:** https://github.com/martj42/international_results
- **Primary file:** `results.csv`
- **Companion files:** `shootouts.csv` (used later to resolve
  draw-but-progressed knockout logic), `goalscorers.csv` (ignored in v1)
- **Coverage:** ~49,000 men's full international matches, 1872–present
- **License:** CC0 1.0 (public domain dedication) — see the repository
- **Retrieved:** _pending first `wc2026 data` run_

### `results.csv` columns

| column      | type   | notes                              |
|-------------|--------|------------------------------------|
| date        | date   | match date (YYYY-MM-DD)            |
| home_team   | str    | home / first team                  |
| away_team   | str    | away / second team                 |
| home_score  | int    | full-time goals, ≥ 0               |
| away_score  | int    | full-time goals, ≥ 0               |
| tournament  | str    | competition label                  |
| city        | str    | host city (nullable)               |
| country     | str    | host country (nullable)            |
| neutral     | bool   | True if played on neutral ground   |

### `shootouts.csv` columns

| column    | type | notes                          |
|-----------|------|--------------------------------|
| date      | date | match date                     |
| home_team | str  |                                |
| away_team | str  |                                |
| winner    | str  | shootout winner                |

---

## 2. Team strength — Elo (derived, not downloaded)

In v1 the per-team Elo rating is **computed from source #1** rather than fetched
from an external Elo provider — it is more robust and keeps the pipeline
self-contained. Standard football Elo: K-factor scaled by match importance and
goal margin, a home-advantage term, and decay between matches. The full rating
timeline is stored so any historical match can be featurized with ratings as they
stood *before* kickoff (leakage-safe). See `src/wc2026/features/elo.py`.

---

## 3. 2026 fixtures, groups, and bracket map (hard-coded, versioned)

- `data/raw/wc2026_groups.json` — the 12 groups (A–L) and their four teams each
- `data/raw/wc2026_fixtures.csv` — the group-stage fixture list
- `data/raw/wc2026_bracket_map.json` — which group positions meet in the R32

These small files are committed to git. The loaders are structured so that
swapping in the official list (when finalized) is a trivial file replacement. A
clearly-marked **placeholder** is provided until the official data is available.

- **License:** factual tournament structure (not copyrightable as data); team
  lists derive from FIFA's published draw.
- **Retrieved / authored:** _pending Phase 2_
