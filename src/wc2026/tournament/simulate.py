"""Monte Carlo tournament simulation — the main expansion seam.

For each of N seeded simulations:

1. sample every group match's **scoreline** from the model's Poisson score grid;
2. resolve the 12 groups via :mod:`wc2026.tournament.groups` (full tiebreakers +
   best-third-placed selection);
3. fill the Round of 32 from the official slot template (loaded from
   ``wc2026_bracket_map.json`` — data, not code), assigning the eight third-placed
   teams to their constrained slots;
4. play the knockouts through the bracket tree, resolving each tie with an
   Elo-weighted penalty coin-flip (a deliberately crude v1 placeholder);
5. tally how often each team reaches each stage / lifts the cup.

The model must expose ``score_grid`` (the Bayesian Poisson model), since group
standings need scorelines, not just W/D/L. Knockout matchups vary per simulation,
so a 48×48 pairwise "advance" matrix is precomputed once and looked up in the hot
loop, keeping 10k simulations fast.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from wc2026 import config
from wc2026.data import loaders
from wc2026.features import build
from wc2026.tournament import groups as G

logger = logging.getLogger(__name__)

#: As-of date for featurization (pre-tournament; ratings are "current").
ASOF = pd.Timestamp("2026-06-11")
_STAGES = ("reach_R32", "reach_R16", "reach_QF", "reach_SF", "reach_final", "win")


def _elo_penalty(elo_home: np.ndarray, elo_away: np.ndarray) -> np.ndarray:
    """Elo-weighted probability the first team wins a penalty shootout (neutral)."""
    return 1.0 / (1.0 + 10.0 ** (-(elo_home - elo_away) / 400.0))


def _assign_thirds(
    ranked_thirds: list[tuple[str, str]], slots: list[tuple[str, str]]
) -> dict[str, str]:
    """Assign qualifying third-placed teams (team, group) to constrained R32 slots.

    Each slot allows thirds from a fixed set of groups; backtracking finds a valid
    perfect matching (FIFA guarantees one exists for any real combination).
    """
    assign: dict[str, str] = {}
    used = [False] * len(ranked_thirds)

    def bt(si: int) -> bool:
        if si == len(slots):
            return True
        sid, allowed = slots[si]
        for qi, (team, grp) in enumerate(ranked_thirds):
            if not used[qi] and grp in allowed:
                used[qi] = True
                assign[sid] = team
                if bt(si + 1):
                    return True
                used[qi] = False
        return False

    if not bt(0):  # pragma: no cover - real combinations always match
        for (sid, _), (team, _) in zip(slots, ranked_thirds, strict=False):
            assign[sid] = team
    return assign


class TournamentSimulator:
    """Precomputes match probabilities once, then runs N seeded simulations."""

    def __init__(self, model, asof: pd.Timestamp = ASOF) -> None:
        if not hasattr(model, "score_grid"):
            raise TypeError(
                "The simulator needs a model with score_grid() (the Bayesian Poisson "
                "model). Train it with `wc2026 train --model bayesian`."
            )
        self.model = model
        self.asof = asof
        self.groups = loaders.load_wc2026_groups()
        self.fixtures = loaders.load_wc2026_fixtures()
        self.bracket = loaders.load_bracket_map()

        self.teams = [t for g in self.groups.values() for t in g]
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.group_of = {t: g for g, ts in self.groups.items() for t in ts}

        results = loaders.load_results()
        self.ctx = build.make_context(results)
        self.hist = build.asof_history(results, self.teams, asof)
        self._elo = np.array([self.ctx.elo.rating_before(t, asof) for t in self.teams])

        self._prep_group_grids()
        self._prep_pairwise()

    # -- precomputation ----------------------------------------------------- #
    def _prep_group_grids(self) -> None:
        pairs = [
            (r.home_team, r.away_team, bool(r.neutral))
            for r in self.fixtures.itertuples(index=False)
        ]
        X = build.feature_rows(pairs, self.asof, self.ctx, self.hist)
        grids, _ = self.model.score_grid(X)
        n_fix = len(pairs)
        side = grids.shape[1]
        flat = grids.reshape(n_fix, -1)
        flat = flat / flat.sum(axis=1, keepdims=True)
        self._fix_cum = flat.cumsum(axis=1)
        self._fix_side = side
        self._fix_meta = [
            (r.group, r.home_team, r.away_team) for r in self.fixtures.itertuples(index=False)
        ]

    def _prep_pairwise(self) -> None:
        n = len(self.teams)
        pairs, ij = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    pairs.append((self.teams[i], self.teams[j], True))  # neutral
                    ij.append((i, j))
        X = build.feature_rows(pairs, self.asof, self.ctx, self.hist)
        proba = self.model.predict_proba(X)  # (m, 3): home/draw/away
        A = np.zeros((n, n))
        for (i, j), p in zip(ij, proba, strict=True):
            pen = _elo_penalty(self._elo[i : i + 1], self._elo[j : j + 1])[0]
            A[i, j] = p[0] + p[1] * pen  # P(i advances over j)
        self._advance = A

    # -- per-simulation helpers -------------------------------------------- #
    def _sample_group_scores(self, rng: np.random.Generator) -> np.ndarray:
        r = rng.random(self._fix_cum.shape[0])
        idx = (self._fix_cum < r[:, None]).sum(axis=1)
        idx = np.clip(idx, 0, self._fix_side * self._fix_side - 1)
        return np.column_stack([idx // self._fix_side, idx % self._fix_side])

    def _resolve_groups(self, scores: np.ndarray):
        """Return (winners, runners, ranked_thirds) for one sampled group stage."""
        by_group: dict[str, list[dict]] = {g: [] for g in self.groups}
        for k, (grp, h, a) in enumerate(self._fix_meta):
            by_group[grp].append(
                {"home_team": h, "away_team": a,
                 "home_score": int(scores[k, 0]), "away_score": int(scores[k, 1])}
            )
        winners, runners, thirds = {}, {}, []
        for g, matches in by_group.items():
            df = pd.DataFrame(matches)
            order = G.rank_group(df, teams=self.groups[g])
            st = G.compute_standings(df, teams=self.groups[g])
            winners[g], runners[g] = order[0], order[1]
            thirds.append(st[order[2]])
        ranked = G.rank_third_placed(thirds)[: config.BEST_THIRD_PLACED_ADVANCING]
        return winners, runners, ranked

    def _play(self, home: str, away: str, rng: np.random.Generator) -> str:
        return home if rng.random() < self._advance[self.idx[home], self.idx[away]] else away

    def _simulate_once(self, rng: np.random.Generator, counts: np.ndarray) -> None:
        scores = self._sample_group_scores(rng)
        winners, runners, ranked_thirds = self._resolve_groups(scores)

        slot_map: dict[str, str] = {}
        for g in self.groups:
            slot_map[f"1{g}"] = winners[g]
            slot_map[f"2{g}"] = runners[g]
        third_slots = [
            (e["away"], e["away"].split(":")[1])
            for e in self.bracket["r32"] if e["away"].startswith("3:")
        ]
        ranked_pairs = [(s.team, self.group_of[s.team]) for s in ranked_thirds]
        slot_map.update(_assign_thirds(ranked_pairs, third_slots))

        # Everyone in the R32 reached that stage.
        qualifiers = [winners[g] for g in self.groups] + [runners[g] for g in self.groups]
        qualifiers += [s.team for s in ranked_thirds]
        for t in qualifiers:
            counts[self.idx[t], 0] += 1

        winner_by_match: dict[int, str] = {}
        for e in self.bracket["r32"]:
            h, a = slot_map[e["home"]], slot_map[e["away"]]
            w = self._play(h, a, rng)
            winner_by_match[e["match"]] = w
            counts[self.idx[w], 1] += 1  # reached R16

        # round -> counts column for the *winner* of that round.
        round_col = {"R16": 2, "QF": 3, "SF": 4, "F": 5}
        for e in self.bracket["knockout"]:
            h = winner_by_match[int(e["home"][1:])]
            a = winner_by_match[int(e["away"][1:])]
            w = self._play(h, a, rng)
            winner_by_match[e["match"]] = w
            counts[self.idx[w], round_col[e["round"]]] += 1

    # -- public ------------------------------------------------------------- #
    def run(self, n: int = 10_000, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        counts = np.zeros((len(self.teams), len(_STAGES)), dtype=np.int64)
        for _ in range(n):
            self._simulate_once(rng, counts)
        probs = counts / n
        df = pd.DataFrame(probs, columns=list(_STAGES))
        df.insert(0, "team", self.teams)
        df.insert(1, "group", [self.group_of[t] for t in self.teams])
        return df.sort_values(["win", "reach_final", "reach_SF"], ascending=False).reset_index(
            drop=True
        )


def simulate_tournament(
    model=None, n: int = 10_000, seed: int = config.RANDOM_SEED
) -> pd.DataFrame:
    """Run the Monte Carlo and return a sorted advancement-probability table.

    ``model`` defaults to the persisted Bayesian Poisson model.
    """
    if model is None:
        from wc2026.models.poisson import BayesianPoissonPredictor

        model = BayesianPoissonPredictor.load()
    logger.info("Simulating the 2026 World Cup: n=%d, seed=%d", n, seed)
    return TournamentSimulator(model).run(n=n, seed=seed)
