"""Tests for ``fraud_detection.models.xgboost_model``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fraud_detection.models.xgboost_model import XGBoostConfig, XGBoostFraudModel


@pytest.fixture
def small_imbalanced_data():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.normal(size=(n, 12)).astype(np.float32)
    # Create a learnable signal: positive class iff first feature > 1.5
    y = (X[:, 0] > 1.5).astype(np.int8)
    return X, y


def test_config_defaults_match_plan():
    cfg = XGBoostConfig()
    assert cfg.n_estimators == 500
    assert cfg.max_depth == 8
    assert cfg.learning_rate == pytest.approx(0.05)
    assert cfg.scale_pos_weight == pytest.approx(27.6)
    assert cfg.tree_method == "hist"
    assert cfg.eval_metric == "aucpr"


def test_fit_then_predict_proba(small_imbalanced_data):
    X, y = small_imbalanced_data
    model = XGBoostFraudModel(
        XGBoostConfig(n_estimators=20, max_depth=3, early_stopping_rounds=None)
    )
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0],)
    assert proba.min() >= 0
    assert proba.max() <= 1
    # Sanity: training accuracy on a learnable problem should beat baseline.
    pred = (proba >= 0.5).astype(np.int8)
    acc = (pred == y).mean()
    assert acc > 0.7, f"too low accuracy {acc}"


def test_predict_threshold(small_imbalanced_data):
    X, y = small_imbalanced_data
    model = XGBoostFraudModel(XGBoostConfig(n_estimators=10, early_stopping_rounds=None)).fit(X, y)
    yhat_lo = model.predict(X, threshold=0.1)
    yhat_hi = model.predict(X, threshold=0.9)
    assert yhat_lo.sum() >= yhat_hi.sum()


def test_predict_proba_before_fit_raises():
    model = XGBoostFraudModel()
    with pytest.raises(RuntimeError, match="not been fit"):
        model.predict_proba(np.zeros((1, 5)))


def test_save_load_roundtrip(tmp_path: Path, small_imbalanced_data):
    X, y = small_imbalanced_data
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    model = XGBoostFraudModel(XGBoostConfig(n_estimators=10, early_stopping_rounds=None))
    model.fit(X, y, feature_names=feature_names)
    path = tmp_path / "xgb.pkl"
    model.save(path)

    loaded = XGBoostFraudModel.load(path)
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X), atol=1e-9)
    assert loaded._feature_names == feature_names


def test_feature_importance(small_imbalanced_data):
    X, y = small_imbalanced_data
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    model = XGBoostFraudModel(XGBoostConfig(n_estimators=10, early_stopping_rounds=None))
    model.fit(X, y, feature_names=feature_names)
    fi = model.feature_importance(kind="gain")
    assert fi  # non-empty
    # Feature 0 carries the signal -- it should be in the importance map.
    assert "feature_0" in fi
    assert fi["feature_0"] > 0


def test_early_stopping_with_eval_set(small_imbalanced_data):
    X, y = small_imbalanced_data
    split = X.shape[0] // 2
    model = XGBoostFraudModel(XGBoostConfig(n_estimators=200, early_stopping_rounds=5))
    model.fit(X[:split], y[:split], X_val=X[split:], y_val=y[split:])
    # With early stopping we expect best_iteration to be set by xgboost.
    assert model.model.best_iteration is not None
