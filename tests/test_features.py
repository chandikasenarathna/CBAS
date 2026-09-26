# tests/test_features.py — unit tests for keystroke feature extraction

from utils.features import (extract_raw_features, pairs_to_feature_vector,
                            compute_digraph_features, events_to_full_feature_vector,
                            windows_to_full_feature_vectors)


def _mk_events(n, dwell_ms=100, flight_ms=120):
    """Build n synthetic events with constant dwell/flight (ms)."""
    events, t = [], 0.0
    for _ in range(n):
        events.append({"press_time": t, "release_time": t + dwell_ms})
        t += dwell_ms + flight_ms
    return events


def test_extract_raw_features_basic():
    events = _mk_events(5, dwell_ms=100, flight_ms=200)
    pairs = extract_raw_features(events)
    assert len(pairs) == 5
    # dwell ~0.1s for all
    assert all(abs(p[0] - 0.1) < 1e-6 for p in pairs)
    # first flight is 0, rest ~0.2s
    assert pairs[0][1] == 0.0
    assert all(abs(p[1] - 0.2) < 1e-6 for p in pairs[1:])


def test_extract_raw_features_filters_invalid():
    events = [
        {"press_time": 0, "release_time": 0},        # rt <= pt -> dropped
        {"press_time": 100, "release_time": 3000},   # dwell > 2s -> dropped
        {"press_time": 3100, "release_time": 3102},  # dwell < 5ms -> dropped
        {"press_time": 3300, "release_time": 3400},  # valid
    ]
    pairs = extract_raw_features(events)
    assert len(pairs) == 1
    assert abs(pairs[0][0] - 0.1) < 1e-6


def test_pairs_to_feature_vector_shape():
    pairs = [[0.1, 0.2]] * 10
    fv = pairs_to_feature_vector(pairs)
    assert len(fv) == 8
    assert abs(fv[0] - 0.1) < 1e-9   # mean dwell
    assert abs(fv[4] - 0.2) < 1e-9   # mean flight
    assert fv[1] == 0.0              # std of constant = 0


def test_pairs_to_feature_vector_too_few():
    assert pairs_to_feature_vector([[0.1, 0.2]]) == [0.0] * 8


def test_full_feature_vector_has_12_elements():
    events = _mk_events(40)
    fv = events_to_full_feature_vector(events)
    assert len(fv) == 12
    assert all(isinstance(v, float) for v in fv)


def test_windows_to_full_feature_vectors():
    events = _mk_events(75)  # 2.5 windows of 30
    vectors = windows_to_full_feature_vectors(events, 30)
    assert len(vectors) == 2
    assert all(len(v) == 12 for v in vectors)


def test_window_feature_distribution_matches_single_window():
    """A 30-key window must give the same vector whether scored directly
    (runtime path) or via the windowing path (training path)."""
    events = _mk_events(60)
    direct = events_to_full_feature_vector(events[:30])
    windowed = windows_to_full_feature_vectors(events, 30)[0]
    assert direct == windowed


def test_digraph_features():
    events = _mk_events(10, dwell_ms=100, flight_ms=100)
    dg = compute_digraph_features(events)
    assert len(dg) == 4
    # dd = dwell + flight = 0.2s
    assert abs(dg[0] - 0.2) < 1e-6
    # ud = flight = 0.1s
    assert abs(dg[2] - 0.1) < 1e-6
