# utils/metrics.py — Biometric performance metrics: FAR, FRR, EER
#
# Conventions:
#   - Scores are in [0, 1]; HIGHER = more likely legitimate.
#   - A window is ACCEPTED when score >= threshold, REJECTED otherwise.
#   - FAR(threshold) = fraction of impostor windows accepted
#   - FRR(threshold) = fraction of genuine windows rejected
#   - EER = the point where FAR ~= FRR (threshold minimising |FAR - FRR|)

import numpy as np


def far_frr(genuine_scores, impostor_scores, threshold: float):
    """Compute (FAR, FRR) at a single threshold."""
    genuine  = np.asarray(genuine_scores,  dtype=float)
    impostor = np.asarray(impostor_scores, dtype=float)
    if len(genuine) == 0 or len(impostor) == 0:
        raise ValueError("Need both genuine and impostor scores.")
    far = float(np.mean(impostor >= threshold))
    frr = float(np.mean(genuine  <  threshold))
    return far, frr


def sweep(genuine_scores, impostor_scores, n_points: int = 200):
    """
    Sweep thresholds across the observed score range.
    Returns (thresholds, fars, frrs) numpy arrays, sorted ascending.
    """
    genuine  = np.asarray(genuine_scores,  dtype=float)
    impostor = np.asarray(impostor_scores, dtype=float)
    lo = float(min(genuine.min(),  impostor.min()))
    hi = float(max(genuine.max(),  impostor.max()))
    thresholds = np.linspace(max(0.0, lo - 1e-6), min(1.0, hi + 1e-6), n_points)
    fars = np.array([np.mean(impostor >= t) for t in thresholds])
    frrs = np.array([np.mean(genuine  <  t) for t in thresholds])
    return thresholds, fars, frrs


def eer(genuine_scores, impostor_scores):
    """
    Equal Error Rate and its threshold.
    Returns (eer_value, eer_threshold).
    """
    thresholds, fars, frrs = sweep(genuine_scores, impostor_scores, n_points=1000)
    i = int(np.argmin(np.abs(fars - frrs)))
    return float((fars[i] + frrs[i]) / 2.0), float(thresholds[i])


def det_points(genuine_scores, impostor_scores, n_points: int = 100):
    """(FAR, FRR) pairs for a DET curve."""
    thresholds, fars, frrs = sweep(genuine_scores, impostor_scores, n_points)
    return fars, frrs
