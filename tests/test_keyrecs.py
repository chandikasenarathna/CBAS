# tests/test_keyrecs.py — unit tests for the KeyRecs loader (small fixtures)

import os
from utils.keyrecs import load_fixed_text, load_free_text, _synthesize_events

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def test_synthesize_events():
    dwells = [0.1, 0.12, 0.11]
    dds    = [0.2, 0.22]          # n-1 down-down latencies
    events = _synthesize_events(dwells, dds)
    assert len(events) == 3
    # press times: 0, 0.2, 0.42 (seconds -> ms)
    assert abs(events[1]["press_time"] - 200.0) < 1e-6
    assert abs(events[2]["press_time"] - 420.0) < 1e-6
    # dwell of key 1 = 0.1s = 100ms
    assert abs(events[0]["release_time"] - events[0]["press_time"] - 100.0) < 1e-6


def test_load_free_text_fixture():
    out = load_free_text(FIX, window_size=4)
    assert "p999" in out
    # fixture has 12 digraph rows -> 3 windows of 4
    assert len(out["p999"]) == 3
    assert all(len(v) == 12 for v in out["p999"])


def test_load_fixed_text_fixture():
    out = load_fixed_text(FIX, window_size=10)
    assert "p999" in out
    # fixture has 2 repetitions of 10 keys -> 2 windows of 10
    assert len(out["p999"]) == 2
    assert all(len(v) == 12 for v in out["p999"])
