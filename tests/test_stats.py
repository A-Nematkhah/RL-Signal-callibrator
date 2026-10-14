import numpy as np
from rlsignal.stats import bh_adjust, best_split, js_divergence, permutation_energy_test, total_variation


def test_permutation_null_is_calibrated():
    rng = np.random.default_rng(1)
    ps = [permutation_energy_test(rng.normal(size=(30, 3)), rng.normal(size=(30, 3)), 199, rng)[1] for _ in range(150)]
    assert np.mean(np.array(ps) < 0.05) < 0.12


def test_permutation_detects_shift():
    rng = np.random.default_rng(2)
    _, p = permutation_energy_test(rng.normal(size=(30, 3)), rng.normal(size=(30, 3)) + 1.0, 199, rng)
    assert p < 0.01


def test_bh_monotone_and_bounded():
    p = np.array([0.001, 0.02, 0.03, 0.5])
    a = bh_adjust(p)
    assert (a >= p - 1e-12).all() and (a <= 1).all() and a[0] <= a[1] <= a[2] <= a[3]


def test_divergences():
    assert js_divergence([1, 0], [1, 0]) == 0
    assert abs(total_variation([1, 0], [0, 1]) - 1) < 1e-12


def test_best_split():
    k, d = best_split([0, 0, 0, 1, 1, 1])
    assert k == 3 and abs(d - 1) < 1e-12


def test_paired_permutation_null_is_calibrated_and_detects_shift():
    from rlsignal.stats import paired_permutation_energy_test
    rng = np.random.default_rng(3)
    ps = []
    for _ in range(120):
        scen = rng.normal(size=(30, 3)) * 3                  # strong shared scenario effect
        ps.append(paired_permutation_energy_test(scen + rng.normal(size=(30, 3)) * .3,
                                                 scen + rng.normal(size=(30, 3)) * .3, 199, rng)[1])
    assert np.mean(np.array(ps) < 0.05) < 0.12
    scen = rng.normal(size=(30, 3)) * 3
    _, p = paired_permutation_energy_test(scen + 0.8 + rng.normal(size=(30, 3)) * .3, scen, 199, rng)
    assert p < 0.01      # paired design sees a shift that the shared scenario variance would hide from the unpaired test
