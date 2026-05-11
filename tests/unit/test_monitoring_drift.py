"""Unit tests for the drift detection module (Phase 7)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fraud_detection.monitoring.drift import (
    DriftDetector,
    DriftDetectorConfig,
    categorical_psi,
    chi_square,
    js_divergence,
    ks_test,
    label_drift,
    population_stability_index,
    prediction_distribution_drift,
)

_RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Pure stats helpers
# ---------------------------------------------------------------------------


class TestPopulationStabilityIndex:
    def test_identical_distributions_yield_near_zero(self):
        ref = _RNG.normal(size=2000)
        psi = population_stability_index(ref, ref)
        assert psi == pytest.approx(0.0, abs=1e-9)

    def test_resampled_same_distribution_yields_low_psi(self):
        ref = _RNG.normal(size=2000)
        cur = _RNG.normal(size=2000)
        psi = population_stability_index(ref, cur)
        # Same generating distribution should be deep below the warn threshold.
        assert psi < 0.05

    def test_mean_shift_triggers_severe_psi(self):
        ref = _RNG.normal(loc=0, size=2000)
        cur = _RNG.normal(loc=2, size=2000)  # 2-sigma shift
        psi = population_stability_index(ref, cur)
        assert psi > 0.25  # severe threshold

    def test_psi_handles_empty_inputs(self):
        assert population_stability_index([], [1, 2, 3]) == 0.0
        assert population_stability_index([1, 2, 3], []) == 0.0

    def test_psi_handles_constant_reference(self):
        # Should not raise and should produce a finite number.
        psi = population_stability_index(np.zeros(100), _RNG.normal(size=100))
        assert math.isfinite(psi)


class TestCategoricalPSI:
    def test_identical_categorical_distributions_psi_zero(self):
        labels = ["a", "b", "c"] * 100
        assert categorical_psi(labels, labels) == pytest.approx(0.0, abs=1e-9)

    def test_disappearing_category_triggers_drift(self):
        ref = ["a"] * 500 + ["b"] * 500
        cur = ["a"] * 1000  # category "b" disappeared
        psi = categorical_psi(ref, cur)
        assert psi > 0.25


class TestKSTest:
    def test_same_distribution_yields_small_statistic(self):
        ref = _RNG.normal(size=2000)
        cur = _RNG.normal(size=2000)
        stat, _p = ks_test(ref, cur)
        assert stat < 0.1

    def test_shifted_distribution_yields_large_statistic(self):
        ref = _RNG.normal(loc=0, size=2000)
        cur = _RNG.normal(loc=3, size=2000)
        stat, _p = ks_test(ref, cur)
        assert stat > 0.5


class TestJSDivergence:
    def test_identical_distributions_yield_zero(self):
        ref = _RNG.normal(size=2000)
        assert js_divergence(ref, ref) == pytest.approx(0.0, abs=1e-9)

    def test_mean_shift_increases_jsd(self):
        ref = _RNG.normal(loc=0, size=2000)
        cur = _RNG.normal(loc=3, size=2000)
        jsd = js_divergence(ref, cur)
        assert jsd > 0.1


class TestChiSquare:
    def test_returns_zero_when_distributions_match(self):
        labels = ["a", "b", "c", "d"] * 250
        stat, _p = chi_square(labels, labels)
        assert stat < 1e-6

    def test_returns_large_statistic_when_categories_shift(self):
        ref = ["a"] * 1000
        cur = ["b"] * 1000
        stat, _p = chi_square(ref, cur)
        assert stat > 100


# ---------------------------------------------------------------------------
# DriftDetector integration
# ---------------------------------------------------------------------------


class TestDriftDetector:
    def _reference(self) -> dict:
        return {
            "amount": _RNG.normal(loc=50, scale=20, size=2000).tolist(),
            "product": _RNG.choice(["W", "C", "H"], size=2000, p=[0.7, 0.2, 0.1]).tolist(),
            "country": _RNG.choice(["US", "GB", "CA"], size=2000).tolist(),
        }

    def test_infers_kind_per_feature(self):
        det = DriftDetector(self._reference())
        assert "amount" in det.numeric_features
        assert "product" in det.categorical_features
        assert "country" in det.categorical_features
        assert det.n_reference == 2000

    def test_reports_no_drift_for_matching_current(self):
        ref = self._reference()
        det = DriftDetector(ref)
        report = det.detect(ref)
        # PSI(ref, ref) is bounded by binning noise; ensure overall stays
        # well below the warn threshold.
        assert report.overall_psi < 0.05
        assert report.severity == "none"
        assert report.n_severe == 0
        # Every feature gets a row.
        names = {f.feature for f in report.features}
        assert names == {"amount", "product", "country"}

    def test_detects_numeric_drift(self):
        ref = self._reference()
        cur = {
            "amount": _RNG.normal(loc=500, scale=20, size=2000).tolist(),  # 22 sigma shift
            "product": ref["product"],
            "country": ref["country"],
        }
        det = DriftDetector(ref)
        report = det.detect(cur)
        amount = next(f for f in report.features if f.feature == "amount")
        assert amount.severity == "severe"
        assert amount.mean_shift is not None and amount.mean_shift > 100
        assert report.overall_psi >= 0.25
        assert report.severity == "severe"

    def test_detects_categorical_drift(self):
        ref = self._reference()
        cur = {
            "amount": ref["amount"],
            "product": ["W"] * 2000,  # product distribution collapsed
            "country": ref["country"],
        }
        det = DriftDetector(ref)
        report = det.detect(cur)
        product = next(f for f in report.features if f.feature == "product")
        assert product.kind == "categorical"
        assert product.severity in {"moderate", "severe"}

    def test_skips_features_missing_from_current(self):
        ref = self._reference()
        cur = {"amount": ref["amount"]}  # only one feature
        report = DriftDetector(ref).detect(cur)
        assert len(report.features) == 1
        assert report.features[0].feature == "amount"

    def test_explicit_feature_kinds_override_inference(self):
        ref = {"binary_int": [0, 1] * 1000}
        # Force categorical treatment even though the values are numeric.
        det = DriftDetector(ref, categorical_features=["binary_int"])
        assert "binary_int" in det.categorical_features

    def test_top_k_orders_features_by_psi(self):
        ref = self._reference()
        cur = {
            "amount": _RNG.normal(loc=500, scale=20, size=2000).tolist(),  # huge drift
            "product": ref["product"],  # no drift
            "country": ref["country"],  # no drift
        }
        det = DriftDetector(ref)
        report = det.detect(cur)
        top = report.top(2)
        assert len(top) == 2
        assert top[0].feature == "amount"

    def test_severity_thresholds_via_config(self):
        ref = self._reference()
        cur = {
            "amount": _RNG.normal(loc=55, scale=20, size=2000).tolist(),  # mild drift
            "product": ref["product"],
            "country": ref["country"],
        }
        # With permissive thresholds the same mild shift looks fine.
        permissive = DriftDetector(ref, config=DriftDetectorConfig(psi_warn=1.0, psi_alert=2.0))
        assert permissive.detect(cur).severity == "none"

        # With aggressive thresholds (warn at any non-zero PSI), it becomes moderate.
        strict = DriftDetector(ref, config=DriftDetectorConfig(psi_warn=1e-9, psi_alert=10.0))
        assert strict.detect(cur).severity == "moderate"

    def test_to_dict_round_trip(self):
        det = DriftDetector(self._reference())
        report = det.detect(self._reference())
        d = report.to_dict()
        assert d["n_features"] == 3
        assert "features" in d and isinstance(d["features"], list)
        assert d["severity"] in {"none", "moderate", "severe"}


class TestLabelDrift:
    def test_label_drift_reports_zero_for_matching_rates(self):
        labels = [1] * 35 + [0] * 965  # 3.5% positive
        result = label_drift(labels, reference_rate=0.035)
        assert result["n"] == 1000
        assert result["production_rate"] == pytest.approx(0.035, abs=1e-6)
        assert result["absolute_drift"] == pytest.approx(0.0, abs=1e-6)

    def test_label_drift_detects_spike(self):
        labels = [1] * 200 + [0] * 800  # 20%
        result = label_drift(labels, reference_rate=0.035)
        assert result["production_rate"] == pytest.approx(0.20, abs=1e-6)
        assert result["absolute_drift"] == pytest.approx(0.165, abs=1e-6)

    def test_label_drift_handles_empty(self):
        result = label_drift([], reference_rate=0.05)
        assert result["n"] == 0
        assert result["production_rate"] == 0.0
        assert result["absolute_drift"] == 0.0


class TestPredictionDistributionDrift:
    def test_zero_psi_on_identical_distributions(self):
        scores = _RNG.uniform(0, 1, size=1000)
        assert prediction_distribution_drift(scores, scores) == pytest.approx(0.0, abs=1e-9)

    def test_severe_psi_when_distribution_shifts(self):
        ref = _RNG.uniform(0, 0.5, size=1000)
        cur = _RNG.uniform(0.5, 1.0, size=1000)
        psi = prediction_distribution_drift(cur, ref)
        # Shifting from the bottom half to the top half is a severe move.
        assert psi >= 0.25
