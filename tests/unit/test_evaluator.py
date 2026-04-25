"""Tests for ``fraud_detection.training.evaluator``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fraud_detection.training.evaluator import (
    EvaluationResult,
    evaluate_predictions,
    plot_calibration_curve,
    plot_pr_curve,
    plot_roc_curve,
    write_evaluation_report,
)


@pytest.fixture
def simple_predictions():
    rng = np.random.default_rng(0)
    n = 1000
    y_true = rng.choice([0, 1], size=n, p=[0.95, 0.05])
    # Score that's noisy but correlated with truth.
    y_score = np.clip(0.3 * y_true + rng.normal(loc=0.1, scale=0.2, size=n), 0, 1)
    return y_true, y_score


def test_metrics_have_expected_ranges(simple_predictions):
    y_true, y_score = simple_predictions
    res = evaluate_predictions(y_true, y_score)
    assert isinstance(res, EvaluationResult)
    assert 0 <= res.auprc <= 1
    assert 0 <= res.auroc <= 1
    assert 0 <= res.best_threshold <= 1
    assert 0 <= res.best_f1 <= 1
    assert res.n_total == len(y_true)
    assert res.n_positive == int(y_true.sum())
    assert res.fraud_rate == pytest.approx(y_true.mean(), abs=1e-6)


def test_perfect_predictor_has_auprc_one():
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    res = evaluate_predictions(y_true, y_score)
    assert res.auprc == pytest.approx(1.0)
    assert res.auroc == pytest.approx(1.0)


def test_random_predictor_auprc_close_to_base_rate():
    rng = np.random.default_rng(1)
    n = 5000
    y_true = rng.choice([0, 1], size=n, p=[0.96, 0.04])
    y_score = rng.uniform(0, 1, size=n)
    res = evaluate_predictions(y_true, y_score)
    # AUPRC for a random scorer ≈ base rate; allow a wide tolerance.
    assert 0.0 < res.auprc < 0.1
    assert 0.4 < res.auroc < 0.6


def test_top_k_precision(simple_predictions):
    y_true, y_score = simple_predictions
    res = evaluate_predictions(y_true, y_score)
    assert "top_0.01" in res.precision_at_top_pct
    # Top-1% should have higher precision than the base rate (model has signal).
    assert res.precision_at_top_pct["top_0.01"] >= res.fraud_rate


def test_confusion_matrix_components_sum_to_n(simple_predictions):
    y_true, y_score = simple_predictions
    res = evaluate_predictions(y_true, y_score)
    cm = res.confusion_matrix_at_threshold
    total = cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"]
    assert total == len(y_true)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        evaluate_predictions(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))


def test_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        evaluate_predictions(np.array([]), np.array([]))


def test_plot_pr_curve_renders(tmp_path: Path, simple_predictions):
    y_true, y_score = simple_predictions
    out = tmp_path / "pr.png"
    fig = plot_pr_curve(y_true, y_score, save_path=out)
    assert out.exists()
    assert out.stat().st_size > 1000  # non-trivial png
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_roc_curve_renders(tmp_path: Path, simple_predictions):
    y_true, y_score = simple_predictions
    out = tmp_path / "roc.png"
    fig = plot_roc_curve(y_true, y_score, save_path=out)
    assert out.exists()
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_calibration_curve_renders(tmp_path: Path, simple_predictions):
    y_true, y_score = simple_predictions
    out = tmp_path / "cal.png"
    fig = plot_calibration_curve(y_true, y_score, save_path=out)
    assert out.exists()
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_write_evaluation_report(tmp_path: Path, simple_predictions):
    y_true, y_score = simple_predictions
    paths = write_evaluation_report(y_true, y_score, output_dir=tmp_path, name="val")
    for key in ("metrics", "pr", "roc", "calibration"):
        assert paths[key].exists()
    metrics = json.loads(paths["metrics"].read_text())
    assert "auprc" in metrics
    assert metrics["n_total"] == len(y_true)
