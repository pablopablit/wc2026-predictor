"""Model smoke test (Phase 5) + scaffold-level interface checks (now).

The full smoke test trains on a tiny slice, predicts, and asserts probabilities
sum to 1. For now we verify the package imports and the Predictor interface is
shaped as the rest of the system expects.
"""

import pytest

from wc2026 import Match, Predictor, __version__
from wc2026.models.base import OUTCOME_CLASSES


def test_package_imports_and_version() -> None:
    assert isinstance(__version__, str)
    assert __version__.count(".") >= 2


def test_outcome_classes_order() -> None:
    assert OUTCOME_CLASSES == ("home_win", "draw", "away_win")


def test_predictor_is_abstract() -> None:
    with pytest.raises(TypeError):
        Predictor()  # type: ignore[abstract]


def test_match_carries_no_score() -> None:
    from datetime import date

    m = Match(home_team="Ecuador", away_team="Argentina", match_date=date(2026, 6, 20))
    # A Match must not expose any goal/score field (no-leakage by construction).
    assert not any("score" in f or "goal" in f for f in m.__dataclass_fields__)


@pytest.mark.skip(reason="End-to-end model smoke test implemented in Phase 5.")
def test_train_predict_probs_sum_to_one() -> None:
    ...
