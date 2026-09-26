# utils/features.py — Feature extraction from browser-captured keystroke events.
#
# The browser JS sends arrays of {key, press_time, release_time} events.
# This module converts them into statistical feature vectors.

import numpy as np


def extract_raw_features(events: list) -> list:
    """
    Convert raw keystroke events into [dwell_time, flight_time] pairs.

    events: list of dicts with 'press_time' and 'release_time' (ms from JS).
    Returns: list of [dwell, flight] pairs.
    """
    pairs = []
    for i, ev in enumerate(events):
        pt = ev.get("press_time", 0)
        rt = ev.get("release_time", 0)
        if rt <= pt:
            continue

        dwell = (rt - pt) / 1000.0  # ms → seconds

        # Flight time: gap from previous key release to this key press
        if i > 0:
            prev_rt = events[i - 1].get("release_time", 0)
            flight = (pt - prev_rt) / 1000.0
        else:
            flight = 0.0

        # Sanity filters
        if dwell > 2.0 or dwell < 0.005:
            continue
        flight = max(0.0, flight)
        if flight > 3.0:
            flight = 0.0

        pairs.append([round(dwell, 6), round(flight, 6)])

    return pairs


def pairs_to_feature_vector(pairs: list) -> list:
    """
    Convert a list of [dwell, flight] pairs into an 8-element statistical
    feature vector:
      [mean_dwell, std_dwell, min_dwell, max_dwell,
       mean_flight, std_flight, min_flight, max_flight]
    """
    if len(pairs) < 3:
        return [0.0] * 8

    arr = np.array(pairs, dtype=float)
    d = arr[:, 0]
    f = arr[:, 1]

    return [
        float(np.mean(d)), float(np.std(d)),
        float(np.min(d)),  float(np.max(d)),
        float(np.mean(f)), float(np.std(f)),
        float(np.min(f)),  float(np.max(f)),
    ]


def events_to_windows(events: list, window_size: int) -> list:
    """
    Convert raw events into multiple feature vectors, one per window.
    Returns list of 8-element feature vectors.
    """
    pairs = extract_raw_features(events)
    windows = []
    for i in range(0, len(pairs) - window_size + 1, window_size):
        chunk = pairs[i : i + window_size]
        fv = pairs_to_feature_vector(chunk)
        if any(v != 0.0 for v in fv):
            windows.append(fv)
    return windows


def compute_digraph_features(events: list) -> list:
    """
    Compute digraph (two-key combination) timing features.
    Returns additional features:
      [mean_dd, std_dd, mean_ud, std_ud]
    where:
      dd = key-down to key-down latency
      ud = key-up to key-down latency (= flight time)
    """
    if len(events) < 2:
        return [0.0] * 4

    dds, uds = [], []
    for i in range(1, len(events)):
        pt_curr = events[i].get("press_time", 0)
        pt_prev = events[i - 1].get("press_time", 0)
        rt_prev = events[i - 1].get("release_time", 0)

        dd = (pt_curr - pt_prev) / 1000.0
        ud = (pt_curr - rt_prev) / 1000.0

        if 0.01 < dd < 3.0:
            dds.append(dd)
        if -1.0 < ud < 3.0:
            uds.append(ud)

    dds = np.array(dds) if dds else np.array([0.0])
    uds = np.array(uds) if uds else np.array([0.0])

    return [
        float(np.mean(dds)), float(np.std(dds)),
        float(np.mean(uds)), float(np.std(uds)),
    ]


def events_to_full_feature_vector(events: list) -> list:
    """
    Extract a comprehensive 12-element feature vector from raw events.
    Combines per-window stats + digraph features for richer representation.
    """
    pairs = extract_raw_features(events)
    base  = pairs_to_feature_vector(pairs)         # 8 features
    dgf   = compute_digraph_features(events)        # 4 features
    return base + dgf                               # 12 features


def windows_to_full_feature_vectors(events: list, window_size: int) -> list:
    """
    Split raw events into non-overlapping windows of `window_size` keystrokes
    and compute the full 12-element feature vector for each window.

    This is THE canonical way to produce training/scoring samples. It is used
    identically for enrolment, live monitoring and KeyRecs benchmark data, so
    that the feature distribution seen at training time matches the one seen
    at runtime (same window size, same filters, same statistics).
    """
    vectors = []
    for i in range(0, len(events) - window_size + 1, window_size):
        chunk = events[i : i + window_size]
        fv = events_to_full_feature_vector(chunk)
        if any(v != 0.0 for v in fv):
            vectors.append(fv)
    return vectors
