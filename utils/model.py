# utils/model.py — ML model training, calibration and scoring for CBAS v2
#
# Each enrolled user gets three joblib artefacts in MODELS_DIR:
#   <username>_IF.joblib   -> {"pipe": IsolationForest pipeline, "bias": b, "scale": s}
#   <username>_SVM.joblib  -> {"pipe": OneClassSVM pipeline,   "bias": b, "scale": s}
#   <username>_META.joblib -> {"thresholds": {...}, "n_windows": n, "trained_at": ts}
#
# Calibration: raw decision-function scores are mapped through a sigmoid whose
# bias/scale are fitted on the enrolment windows so that the *median genuine
# window* maps to ~0.65 and the genuine inter-quartile range maps to roughly
# sigmoid(+/-1). This makes scores comparable across users (unlike the previous
# hand-tuned +0.25 / x4.0 constants) and allows per-user thresholds to be
# derived from the genuine score distribution.
#
# The UserModel class is used BOTH by the Flask app (via the module-level
# train_models/score_features helpers) and by the offline KeyRecs benchmark
# (evaluate.py), guaranteeing that live and offline scoring are identical.

import os, math, time
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from config import (MODELS_DIR, IF_CONTAMINATION, IF_N_ESTIMATORS,
                    SVM_NU, SVM_KERNEL, SVM_GAMMA,
                    THRESHOLD_GOOD, THRESHOLD_WARN, THRESHOLD_LOCK, TRUST_INIT,
                    WINDOW_SIZE, PROFILE_VERSION)

# EWMA variance-shrinkage factor: (1-lambda)/(1+lambda) for lambda=0.75.
# Trust scores are EWMA-smoothed, so their spread is narrower than raw scores;
# per-user thresholds are shrunk toward the mean by sqrt of this factor.
_EWMA_VAR_RATIO = 0.25 / 1.75

_CALIB_MEDIAN_TARGET = 0.65   # median genuine window should score about this

IF_WEIGHT  = 0.70             # IF is more reliable for one-class, gets 70%
SVM_WEIGHT = 0.30


def _path(username, name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    return os.path.join(MODELS_DIR, f"{username}_{name}.joblib")


def _sigmoid(x):
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _fit_calibration(raw_scores: np.ndarray):
    """
    Fit sigmoid bias/scale on raw decision-function scores of training data.
    After calibration: median(raw) -> _CALIB_MEDIAN_TARGET, and the genuine
    IQR maps to approximately sigmoid(+1)/sigmoid(-1).
    """
    med = float(np.median(raw_scores))
    q25, q75 = np.percentile(raw_scores, [25, 75])
    iqr = max(float(q75 - q25), 1e-6)
    scale = 2.0 / iqr
    # logit of the median target, so that sigmoid((med + bias) * scale) = target
    logit_target = math.log(_CALIB_MEDIAN_TARGET / (1.0 - _CALIB_MEDIAN_TARGET))
    bias = (logit_target / scale) - med
    return bias, scale


def _derive_thresholds(genuine_scores: np.ndarray) -> dict:
    """
    Derive GOOD/WARN/LOCK thresholds from the genuine score distribution.
    Percentiles are taken on the raw score distribution and then shrunk
    toward the mean to account for EWMA smoothing of the trust score.
    Falls back to config constants if the distribution is degenerate.
    """
    mu = float(np.mean(genuine_scores))
    shrink = math.sqrt(_EWMA_VAR_RATIO)

    def pct(q):
        return mu + shrink * (float(np.percentile(genuine_scores, q)) - mu)

    # These boundaries are deliberately derived from *held-out* genuine data.
    # A genuine user should normally remain in the OK state: WARN represents
    # the lower tail of normal behaviour and LOCK represents only the extreme
    # tail, rather than the old 30th/15th/5th-percentile policy.
    lock = float(np.clip(pct(1),  0.05, 0.95))
    warn = float(np.clip(pct(3),  0.05, 0.95))
    good = float(np.clip(pct(10), 0.05, 0.95))

    # Guarantee ordering lock < warn < good
    if not (lock < warn < good):
        return {"good": THRESHOLD_GOOD, "warn": THRESHOLD_WARN,
                "lock": THRESHOLD_LOCK,
                "baseline": round(float(np.median(genuine_scores)), 4)}
    return {
        "good": round(good, 4),
        "warn": round(warn, 4),
        "lock": round(lock, 4),
        "baseline": round(float(np.median(genuine_scores)), 4),
    }


class UserModel:
    """Trained per-user anomaly detector: IF + One-Class SVM + calibration."""

    def __init__(self, contamination=IF_CONTAMINATION, nu=SVM_NU):
        self.contamination = contamination
        self.nu = nu
        self.thresholds = None
        self.n_windows = 0

    # ── Training ──────────────────────────────────────────────────────────
    def fit(self, feature_vectors: list, calibration_vectors: list | None = None) -> dict:
        X = np.array(feature_vectors, dtype=float)
        if len(X) < 5:
            raise ValueError(f"Need at least 5 feature vectors, got {len(X)}.")

        self.if_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  IsolationForest(
                n_estimators=IF_N_ESTIMATORS,
                contamination=self.contamination,
                random_state=42, n_jobs=1))
        ])
        self.if_pipe.fit(X)
        if_raw = self.if_pipe.decision_function(X)
        self.if_bias, self.if_scale = _fit_calibration(if_raw)

        self.svm_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  OneClassSVM(nu=self.nu, kernel=SVM_KERNEL, gamma=SVM_GAMMA))
        ])
        self.svm_pipe.fit(X)
        svm_raw = self.svm_pipe.decision_function(X)
        self.svm_bias, self.svm_scale = _fit_calibration(svm_raw)

        # Thresholds must be chosen on data that was not used to fit the model.
        # Falling back to the training data keeps the class usable for small
        # experiments, but application training always supplies held-out data.
        threshold_vectors = calibration_vectors or feature_vectors
        calibration_scores = self.score_batch(threshold_vectors)
        self.thresholds = _derive_thresholds(calibration_scores)
        self.n_windows  = int(len(X))

        return {
            "n_vectors"      : len(X),
            "n_calibration"  : len(threshold_vectors),
            "if_normal_pct"  : round(float(np.mean(self.if_pipe.predict(X) == 1)) * 100, 1),
            "svm_normal_pct" : round(float(np.mean(self.svm_pipe.predict(X) == 1)) * 100, 1),
            "thresholds"     : self.thresholds,
        }

    # ── Scoring ───────────────────────────────────────────────────────────
    def score(self, feature_vector: list) -> dict:
        X = np.array([feature_vector], dtype=float)

        if_raw  = float(self.if_pipe.decision_function(X)[0])
        svm_raw = float(self.svm_pipe.decision_function(X)[0])

        if_norm  = _sigmoid((if_raw  + self.if_bias)  * self.if_scale)
        svm_norm = _sigmoid((svm_raw + self.svm_bias) * self.svm_scale)
        combined = IF_WEIGHT * if_norm + SVM_WEIGHT * svm_norm

        return {
            "if_score"   : round(if_norm,  4),
            "svm_score"  : round(svm_norm, 4),
            "combined"   : round(combined, 4),
            "is_anomaly" : combined < 0.50,
            "if_raw"     : round(if_raw,   4),
            "svm_raw"    : round(svm_raw,  4),
        }

    def score_batch(self, feature_vectors) -> np.ndarray:
        """Fast combined scores for a list of feature vectors (evaluation)."""
        X = np.array(feature_vectors, dtype=float)
        if_raw  = self.if_pipe.decision_function(X)
        svm_raw = self.svm_pipe.decision_function(X)
        if_norm  = 1.0 / (1.0 + np.exp(-(if_raw  + self.if_bias)  * self.if_scale))
        svm_norm = 1.0 / (1.0 + np.exp(-(svm_raw + self.svm_bias) * self.svm_scale))
        return IF_WEIGHT * if_norm + SVM_WEIGHT * svm_norm

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self, username: str):
        joblib.dump({"pipe": self.if_pipe,  "bias": self.if_bias,  "scale": self.if_scale},
                    _path(username, "IF"))
        joblib.dump({"pipe": self.svm_pipe, "bias": self.svm_bias, "scale": self.svm_scale},
                    _path(username, "SVM"))
        joblib.dump({"thresholds": self.thresholds,
                     "n_windows": self.n_windows,
                     "window_size": WINDOW_SIZE,
                     "profile_version": PROFILE_VERSION,
                     "trained_at": time.time()},
                    _path(username, "META"))

    @classmethod
    def load(cls, username: str) -> "UserModel":
        if not models_exist(username):
            raise RuntimeError("No trained models found.")
        m = cls()
        if_bundle  = joblib.load(_path(username, "IF"))
        svm_bundle = joblib.load(_path(username, "SVM"))
        if isinstance(if_bundle, dict) and "pipe" in if_bundle:
            m.if_pipe,  m.if_bias,  m.if_scale   = if_bundle["pipe"],  if_bundle["bias"],  if_bundle["scale"]
            m.svm_pipe, m.svm_bias, m.svm_scale  = svm_bundle["pipe"], svm_bundle["bias"], svm_bundle["scale"]
        else:
            # Legacy format: bare pipelines with old hand-tuned constants
            m.if_pipe,  m.if_bias,  m.if_scale   = if_bundle,  0.25, 4.0
            m.svm_pipe, m.svm_bias, m.svm_scale  = svm_bundle, 0.25, 4.0
        m.thresholds = load_thresholds(username)
        return m


# ── Module-level helpers used by the Flask app ────────────────────────────────

def train_models(username: str, feature_vectors: list) -> dict:
    """Train on 80% of enrolment windows and calibrate on the held-out 20%."""
    if len(feature_vectors) < 10:
        raise ValueError("Need at least 10 enrolment windows to train safely.")
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(feature_vectors))
    n_calibration = max(5, int(round(len(feature_vectors) * 0.20)))
    n_calibration = min(n_calibration, len(feature_vectors) - 5)
    calibration = [feature_vectors[i] for i in indices[:n_calibration]]
    training = [feature_vectors[i] for i in indices[n_calibration:]]
    m = UserModel()
    result = m.fit(training, calibration)
    result["n_total"] = len(feature_vectors)
    m.save(username)
    return result


def models_exist(username: str) -> bool:
    if not all(os.path.exists(_path(username, n)) for n in ("IF", "SVM", "META")):
        return False
    try:
        meta = joblib.load(_path(username, "META"))
        return (meta.get("window_size") == WINDOW_SIZE and
                meta.get("profile_version") == PROFILE_VERSION)
    except Exception:
        return False


def load_thresholds(username: str) -> dict:
    """Per-user thresholds, falling back to config constants."""
    p = _path(username, "META")
    if os.path.exists(p):
        try:
            t = joblib.load(p).get("thresholds")
            if t:
                return t
        except Exception:
            pass
    return {"good": THRESHOLD_GOOD, "warn": THRESHOLD_WARN,
            "lock": THRESHOLD_LOCK, "baseline": TRUST_INIT}


def delete_models(username: str):
    for n in ("IF", "SVM", "META"):
        p = _path(username, n)
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception as e:
                print(f"[Model] Warning: could not remove {p}: {e}")



def score_features(username: str, feature_vector: list) -> dict:
    return UserModel.load(username).score(feature_vector)
