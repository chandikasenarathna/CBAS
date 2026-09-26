# config.py — Central configuration for CBAS v2

import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
DB_PATH     = os.path.join(DATA_DIR, "cbas.db")

# ── Keystroke capture ──────────────────────────────────────────────────────────
# Use the same window length for enrolment, live monitoring and evaluation.
# A 60-key window is less noisy than the earlier 30-key setting, though it takes
# longer to make a decision. Existing profiles must be reset and retrained.
WINDOW_SIZE         = 60
ENROL_PASSAGES      = 10
ENROL_MIN_WINDOWS   = 40
ENROL_MIN_FREE_WINDOWS = 12
ENROL_REQUIRE_FREE  = True
PROFILE_VERSION     = 2         # invalidates profiles trained with older window/calibration rules

# ── Enrolment passage (user types this multiple times) ─────────────────────────
ENROL_TEXT = (
    "The quick brown fox jumps over the lazy dog near the riverbank. "
    "Security systems must balance usability with strong protection. "
    "Typing patterns reveal unique behavioural characteristics of each person."
)

# ── Trust score engine ─────────────────────────────────────────────────────────
TRUST_DECAY     = 0.75      # EWMA decay — balances responsiveness with stability
TRUST_INIT      = 0.65      # fallback only; trained users use their learned baseline
WARMUP_WINDOWS  = 3         # observe a few windows before making a security decision

# ── Response thresholds ────────────────────────────────────────────────────────
THRESHOLD_GOOD  = 0.58      # Above this = green (authenticated)
THRESHOLD_WARN  = 0.52      # Between LOCK and GOOD = amber warning
THRESHOLD_LOCK  = 0.46      # Below this = red (MFA triggered)

# ── MFA ────────────────────────────────────────────────────────────────────────
MFA_ISSUER  = "CBAS Authentication"
MFA_WINDOW  = 1

# ── Flask ──────────────────────────────────────────────────────────────────────
SECRET_KEY  = os.environ.get("CBAS_SECRET", "cbas-v2-dev-key-2026")
DEBUG       = os.environ.get("CBAS_DEBUG", "0") == "1"
PORT        = int(os.environ.get("CBAS_PORT", "5000"))

# ── ML Model ──────────────────────────────────────────────────────────────────
IF_CONTAMINATION = 0.05
IF_N_ESTIMATORS  = 200
SVM_NU           = 0.01
SVM_KERNEL       = "rbf"
SVM_GAMMA        = "scale"

# ── Admin ──────────────────────────────────────────────────────────────────
ADMIN_DEFAULT_USERNAME = os.environ.get("CBAS_ADMIN_USER", "admin")
ADMIN_EVENTS_PER_PAGE  = 50
