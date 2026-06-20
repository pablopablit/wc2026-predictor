"""Schema-validation tests for the data layer (no network required)."""

import pandas as pd
import pytest

from wc2026.data import ingest, loaders, schema


def _good_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-11-20", "2022-11-21"]),
            "home_team": ["Qatar", "England"],
            "away_team": ["Ecuador", "Iran"],
            "home_score": [0, 6],
            "away_score": [2, 2],
            "tournament": ["FIFA World Cup", "FIFA World Cup"],
            "city": ["Al Khor", "Doha"],
            "country": ["Qatar", "Qatar"],
            "neutral": [False, True],
        }
    )


def test_validate_results_passes_clean_data() -> None:
    out = schema.validate_results(_good_results())
    assert list(out.columns) == [
        "date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "city", "country", "neutral",
    ]
    assert out["home_score"].dtype.kind == "i"
    assert out["neutral"].dtype == bool


def test_validate_results_drops_exact_duplicates() -> None:
    df = pd.concat([_good_results(), _good_results().iloc[[0]]], ignore_index=True)
    out = schema.validate_results(df)
    assert len(out) == 2  # the duplicated first row is removed


def test_validate_results_rejects_negative_scores() -> None:
    bad = _good_results()
    bad.loc[0, "home_score"] = -1
    with pytest.raises(ValueError, match="schema validation"):
        schema.validate_results(bad)


def test_validate_results_rejects_missing_column() -> None:
    bad = _good_results().drop(columns=["neutral"])
    with pytest.raises(ValueError, match="schema validation"):
        schema.validate_results(bad)


def test_drop_unplayed_removes_missing_scores() -> None:
    df = _good_results()
    df.loc[1, ["home_score", "away_score"]] = [pd.NA, pd.NA]
    out = loaders.drop_unplayed(df)
    assert len(out) == 1
    assert out.loc[0, "home_team"] == "Qatar"


def test_file_sha256_is_stable(tmp_path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"world cup 2026")
    assert ingest.file_sha256(p) == ingest.file_sha256(p)
    assert len(ingest.file_sha256(p)) == 64
