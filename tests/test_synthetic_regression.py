"""The synthetic estimator fixtures must stay bit-identical for a given seed: other tests are calibrated on them."""
import hashlib
import json

import numpy as np

from rlsignal.validation.synthetic import make_estimator_runs

SPECS = {"a": dict(p_final=0.8, onset=2000, ramp=4000, seed_sd=0.05, coll_final=0.2, ret_final=1.0),
         "b": dict(p_final=0.4, onset=1000, ramp=8000, beh_onset=0, beh_ramp=3000, fail_frac=0.3),
         "c": dict(p_final=0.3, drift=0.4),
         "null_control": dict(p_final=0.0)}
STEPS = [2048, 4096, 8192, 16384]


def _round(o):
    if isinstance(o, float):
        return float(f"{o:.12g}")
    if isinstance(o, dict):
        return {str(k): _round(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_round(v) for v in o]
    return o


def digest(runs):
    h = hashlib.sha256()
    for rid in sorted(runs):
        for s in sorted(runs[rid]):
            r = runs[rid][s]
            h.update(json.dumps(_round({"E": r.eval_E, "null": r.null_sets, "R": r.eval_R}), sort_keys=True).encode())
            for t in sorted(r.probe):
                h.update(np.round(r.probe[t].astype(np.float64), 6).tobytes())
    return h.hexdigest()


def test_make_estimator_runs_is_bit_identical_for_a_fixed_seed():
    runs = make_estimator_runs(SPECS, STEPS, n_seeds=3, n_eps=6, n_ref_eps=10, seed=123)
    assert digest(runs) == EXPECTED


# Digest of the implementation before the B023 clean-up (closures bound explicitly); must never change.
EXPECTED = "e20c147353395e790f0b99d7c5e4bbfedd9a2063f2f5df01f3a5049a703706dc"
