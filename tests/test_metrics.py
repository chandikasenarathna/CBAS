# tests/test_metrics.py — unit tests for FAR/FRR/EER

import numpy as np
from utils.metrics import far_frr, eer, sweep


def test_far_frr_perfect_separation():
    genuine  = [0.9, 0.95, 0.85, 0.92]
    impostor = [0.1, 0.2, 0.15, 0.05]
    far, frr = far_frr(genuine, impostor, threshold=0.5)
    assert far == 0.0
    assert frr == 0.0


def test_far_frr_all_wrong():
    genuine  = [0.1, 0.2]      # all below threshold -> all rejected
    impostor = [0.8, 0.9]      # all above threshold -> all accepted
    far, frr = far_frr(genuine, impostor, threshold=0.5)
    assert far == 1.0
    assert frr == 1.0


def test_eer_perfect_separation():
    rng = np.random.default_rng(0)
    genuine  = rng.normal(0.8, 0.05, 500)
    impostor = rng.normal(0.2, 0.05, 500)
    e, t = eer(genuine, impostor)
    assert e < 0.02
    assert 0.2 < t < 0.8


def test_eer_overlapping():
    # Identical distributions -> EER should be ~0.5
    rng = np.random.default_rng(1)
    genuine  = rng.normal(0.5, 0.1, 1000)
    impostor = rng.normal(0.5, 0.1, 1000)
    e, t = eer(genuine, impostor)
    assert 0.4 < e < 0.6


def test_sweep_monotonic():
    genuine  = [0.6, 0.7, 0.8]
    impostor = [0.2, 0.3, 0.4]
    th, fars, frrs = sweep(genuine, impostor, n_points=50)
    # FAR decreases as threshold rises; FRR increases
    assert all(fars[i] >= fars[i + 1] for i in range(len(fars) - 1))
    assert all(frrs[i] <= frrs[i + 1] for i in range(len(frrs) - 1))
