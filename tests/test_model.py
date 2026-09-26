# tests/test_model.py — unit tests for model training, calibration, scoring

import numpy as np
import pytest
from utils.model import UserModel, _derive_thresholds


def _blobs(n=40, dim=12, center=0.0, spread=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(center, spread, (n, dim))).tolist()


def test_fit_returns_thresholds_and_stats():
    m = UserModel()
    result = m.fit(_blobs())
    assert result["n_vectors"] == 40
    t = result["thresholds"]
    assert t["lock"] < t["warn"] < t["good"]
    assert all(0.05 <= v <= 0.95 for v in t.values())
    assert "baseline" in t


def test_held_out_calibration_is_reported():
    m = UserModel()
    result = m.fit(_blobs(n=40), calibration_vectors=_blobs(n=10, seed=19))
    assert result["n_vectors"] == 40
    assert result["n_calibration"] == 10
    assert result["thresholds"]["lock"] < result["thresholds"]["warn"] < result["thresholds"]["good"]


def test_fit_too_few_samples():
    m = UserModel()
    with pytest.raises(ValueError):
        m.fit(_blobs(n=3))


def test_genuine_scores_higher_than_impostor():
    m = UserModel()
    m.fit(_blobs(n=60, center=0.0, spread=0.04))
    genuine  = _blobs(n=20, center=0.0, spread=0.04, seed=7)
    impostor = _blobs(n=20, center=0.6, spread=0.04, seed=8)  # far away
    gen = m.score_batch(genuine)
    imp = m.score_batch(impostor)
    assert float(np.mean(gen)) > float(np.mean(imp))
    # Genuine median should sit near the calibration target (~0.65)
    assert 0.4 < float(np.median(gen)) < 0.9


def test_score_shape():
    m = UserModel()
    m.fit(_blobs())
    r = m.score(_blobs(n=1)[0])
    assert set(r.keys()) >= {"if_score", "svm_score", "combined", "if_raw", "svm_raw"}
    assert 0.0 <= r["combined"] <= 1.0


def test_derive_thresholds_fallback_on_degenerate():
    # Constant scores -> degenerate distribution -> config fallback ordering
    t = _derive_thresholds(np.array([0.5] * 20))
    assert t["lock"] <= t["warn"] <= t["good"]


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    import utils.model as mm
    monkeypatch.setattr(mm, "MODELS_DIR", str(tmp_path))
    m = UserModel()
    m.fit(_blobs())
    m.save("tester")
    loaded = UserModel.load("tester")
    fv = _blobs(n=1, seed=3)[0]
    assert loaded.score(fv)["combined"] == pytest.approx(m.score(fv)["combined"], abs=1e-6)
    assert loaded.thresholds == m.thresholds
