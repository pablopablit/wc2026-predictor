"""No-leakage contract for the feature builder (Phase 4).

This test must FAIL if any feature function peeks at the match's own result or at
any future match: features are computed from a Match (which carries no score) plus
history strictly before kickoff. We also verify that perturbing the match's score
leaves the feature vector unchanged.
"""

import pytest


@pytest.mark.skip(reason="Feature builder is implemented in Phase 4.")
def test_features_ignore_match_result() -> None:
    ...


@pytest.mark.skip(reason="Feature builder is implemented in Phase 4.")
def test_features_use_only_past_matches() -> None:
    ...
