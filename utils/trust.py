# utils/trust.py — Trust score engine v2
#
# EWMA-smoothed trust score with per-user thresholds and a hard session-lock
# state. Once the score falls below the LOCK threshold the engine locks and
# stays locked until a successful MFA challenge — further typing does NOT
# restore trust by itself (that is what makes it a real session lock).

import time
from config import (TRUST_DECAY, TRUST_INIT, THRESHOLD_GOOD, THRESHOLD_WARN,
                    THRESHOLD_LOCK, WARMUP_WINDOWS)

LEVEL_OK   = "OK"
LEVEL_WARN = "WARN"
LEVEL_MFA  = "MFA_REQUIRED"
LEVEL_WARMUP = "WARMUP"


class TrustEngine:
    def __init__(self, thresholds: dict = None):
        t = thresholds or {}
        self.th_good = float(t.get("good", THRESHOLD_GOOD))
        self.th_warn = float(t.get("warn", THRESHOLD_WARN))
        self.th_lock = float(t.get("lock", THRESHOLD_LOCK))
        # Starting at a trained user's genuine baseline avoids a misleading
        # apparent fall from 1.00 while normal behaviour is being observed.
        self.score      = float(t.get("baseline", TRUST_INIT))
        self.level      = LEVEL_WARMUP
        self.windows_seen = 0
        self.warmup_windows = WARMUP_WINDOWS
        self.history    = []
        self.mfa_active = False
        self.locked     = False

    def update(self, new_score: float) -> dict:
        # A locked session stays locked regardless of further input
        if self.locked:
            return self.state()

        self.score = TRUST_DECAY * self.score + (1 - TRUST_DECAY) * new_score
        self.score = max(0.0, min(1.0, self.score))
        self.windows_seen += 1

        # Score and display early windows, but do not warn or lock until a
        # short baseline-confirmation period has completed.
        if self.windows_seen < self.warmup_windows:
            self.level = LEVEL_WARMUP
            self.history.append((time.time(), round(self.score, 4), self.level))
            return self.state()

        if self.score >= self.th_good:
            self.level = LEVEL_OK
        elif self.score >= self.th_lock:
            self.level = LEVEL_WARN
        else:
            self.level = LEVEL_MFA
            self.mfa_active = True
            self.locked = True

        self.history.append((time.time(), round(self.score, 4), self.level))
        return self.state()

    def mfa_passed(self):
        self.score = self.th_good + 0.10
        self.level = LEVEL_OK
        self.windows_seen = self.warmup_windows
        self.mfa_active = False
        self.locked = False

    def mfa_failed(self):
        self.mfa_active = True
        self.level = LEVEL_MFA
        self.locked = True

    def reset(self):
        thresholds = {"good": self.th_good, "warn": self.th_warn, "lock": self.th_lock}
        thresholds["baseline"] = self.score
        self.__init__(thresholds)

    def state(self) -> dict:
        return {
            "score": round(self.score, 4),
            "level": self.level,
            "mfa_active": self.mfa_active,
            "locked": self.locked,
            "windows_seen": self.windows_seen,
            "warmup_remaining": max(0, self.warmup_windows - self.windows_seen),
            "thresholds": {"good": self.th_good, "warn": self.th_warn, "lock": self.th_lock},
        }

    def recent_history(self, n=50):
        return [{"t": t, "score": s, "level": l} for t, s, l in self.history[-n:]]
