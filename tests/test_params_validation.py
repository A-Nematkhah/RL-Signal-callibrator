"""Typos in estimator / holdout target keys must fail loudly with close matches, not be silently ignored."""

import pytest

from rlsignal.params import EstimatorParams
from rlsignal.validation.holdout import DEFAULT_TARGETS, validate_holdout_targets


def test_unknown_estimator_key_lists_close_matches():
    with pytest.raises(ValueError, match="uncertian_reference") as e:
        EstimatorParams.from_cfg({"uncertian_reference": "block", "prob": 0.8})
    assert "uncertain_reference" in str(e.value)


def test_unknown_holdout_target_lists_close_matches():
    with pytest.raises(ValueError, match="prob_taw") as e:
        validate_holdout_targets({**DEFAULT_TARGETS, "prob_taw": 0.8})
    assert "prob_tau" in str(e.value)


def test_known_keys_still_load():
    P = EstimatorParams.from_cfg({"prob": 0.75, "uncertain_reference": "exclude_pairs"})
    assert P.prob == 0.75 and P.uncertain_reference == "exclude_pairs"
    assert validate_holdout_targets(DEFAULT_TARGETS) == DEFAULT_TARGETS
