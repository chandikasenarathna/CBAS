# tests/test_trust.py — unit tests for the EWMA trust engine

from utils.trust import TrustEngine, LEVEL_OK, LEVEL_WARN, LEVEL_MFA, LEVEL_WARMUP
from config import TRUST_DECAY, TRUST_INIT, THRESHOLD_GOOD, THRESHOLD_LOCK


def test_ewma_update_math():
    e = TrustEngine()
    e.update(0.0)
    expected = TRUST_DECAY * TRUST_INIT
    assert abs(e.score - expected) < 1e-9


def test_warmup_delays_security_decision_and_uses_profile_baseline():
    e = TrustEngine({"good": 0.60, "warn": 0.55, "lock": 0.50, "baseline": 0.68})
    first = e.update(0.10)
    second = e.update(0.10)
    assert first["level"] == LEVEL_WARMUP
    assert second["level"] == LEVEL_WARMUP
    assert second["warmup_remaining"] == 1
    third = e.update(0.10)
    assert third["windows_seen"] == 3


def test_levels_with_default_thresholds():
    e = TrustEngine()
    # Feed high scores -> stays OK
    for _ in range(10):
        state = e.update(0.95)
    assert state["level"] == LEVEL_OK
    # Feed very low scores -> drops through WARN to MFA/locked
    seen_warn = False
    for _ in range(60):
        state = e.update(0.0)
        if state["level"] == LEVEL_WARN:
            seen_warn = True
    assert state["level"] == LEVEL_MFA
    assert state["locked"] is True
    assert seen_warn  # passed through the amber zone


def test_locked_engine_ignores_further_input():
    e = TrustEngine()
    for _ in range(100):
        e.update(0.0)
    assert e.locked is True
    score_at_lock = e.score
    # Even perfect scores do not restore a locked session
    state = e.update(1.0)
    assert state["locked"] is True
    assert state["score"] == round(score_at_lock, 4)


def test_mfa_pass_unlocks():
    e = TrustEngine()
    for _ in range(100):
        e.update(0.0)
    assert e.locked is True
    e.mfa_passed()
    assert e.locked is False
    assert e.level == LEVEL_OK
    assert e.score > THRESHOLD_GOOD


def test_per_user_thresholds_override():
    custom = {"good": 0.9, "warn": 0.7, "lock": 0.5}
    e = TrustEngine(thresholds=custom)
    # Drive score to 0.8 -> between warn and good for default, but below
    # custom 'good' (0.9) -> should be WARN with custom thresholds
    for _ in range(50):
        state = e.update(0.8)
    assert abs(e.score - 0.8) < 1e-6
    assert state["level"] == LEVEL_WARN
