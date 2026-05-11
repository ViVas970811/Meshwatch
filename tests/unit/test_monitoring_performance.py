"""Unit tests for the production performance tracker (Phase 7)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from fraud_detection.monitoring.performance import (
    PerformanceTracker,
    auprc,
    auroc,
)

_RNG = np.random.default_rng(42)


class TestAUCMetrics:
    def test_auroc_perfect_separation(self):
        y_true = [0, 0, 0, 1, 1, 1]
        y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        assert auroc(y_true, y_score) == pytest.approx(1.0, abs=1e-9)

    def test_auroc_no_signal(self):
        y_true = [0, 0, 1, 1] * 100
        y_score = [0.5] * 400
        # AUROC under no signal is 0.5; with random tie-breaking the value
        # can fluctuate but should stay close to 0.5.
        assert 0.4 < auroc(y_true, y_score) < 0.6

    def test_auroc_handles_single_class(self):
        # All labels the same -- AUROC undefined; expect NaN.
        result = auroc([0, 0, 0], [0.1, 0.2, 0.3])
        assert result != result  # NaN

    def test_auprc_perfect_separation(self):
        y_true = [0, 0, 0, 1, 1, 1]
        y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        assert auprc(y_true, y_score) == pytest.approx(1.0, abs=1e-9)

    def test_auprc_handles_no_positives(self):
        result = auprc([0, 0, 0], [0.1, 0.2, 0.3])
        assert result != result  # NaN


class TestPerformanceTracker:
    def test_records_predictions_without_labels(self):
        t = PerformanceTracker(max_records=10)
        for i in range(5):
            t.record_prediction(transaction_id=f"tx_{i}", score=0.5)
        assert t.n_records == 5
        assert t.n_labelled == 0

    def test_label_attaches_to_existing_record(self):
        t = PerformanceTracker()
        t.record_prediction("tx_1", 0.9)
        assert t.record_label("tx_1", 1) is True
        assert t.n_labelled == 1

    def test_label_returns_false_when_record_missing(self):
        t = PerformanceTracker()
        assert t.record_label("unknown_tx", 1) is False

    def test_ring_buffer_evicts_oldest(self):
        t = PerformanceTracker(max_records=3)
        for i in range(5):
            t.record_prediction(f"tx_{i}", 0.5)
        assert t.n_records == 3

    def test_reset_clears_buffer(self):
        t = PerformanceTracker()
        t.record_prediction("tx_1", 0.5)
        t.reset()
        assert t.n_records == 0

    def test_snapshot_with_no_labels_returns_empty(self):
        t = PerformanceTracker()
        for i in range(5):
            t.record_prediction(f"tx_{i}", 0.5)
        snap = t.snapshot()
        assert snap.n_total == 5
        assert snap.n_labelled == 0
        assert snap.precision == 0.0
        assert snap.recall == 0.0

    def test_snapshot_computes_classification_metrics(self):
        t = PerformanceTracker(threshold=0.5)
        # TP: 2, FP: 1, FN: 1, TN: 2 -- precision 2/3, recall 2/3
        t.record_prediction("a", 0.9, label=1)
        t.record_prediction("b", 0.8, label=1)
        t.record_prediction("c", 0.7, label=0)
        t.record_prediction("d", 0.3, label=1)
        t.record_prediction("e", 0.2, label=0)
        t.record_prediction("f", 0.1, label=0)
        snap = t.snapshot()
        assert snap.tp == 2
        assert snap.fp == 1
        assert snap.fn == 1
        assert snap.tn == 2
        assert snap.precision == pytest.approx(2 / 3, abs=1e-6)
        assert snap.recall == pytest.approx(2 / 3, abs=1e-6)
        assert snap.f1 == pytest.approx(2 / 3, abs=1e-6)

    def test_snapshot_windowing_filters_old_records(self):
        t = PerformanceTracker()
        old = datetime.now(timezone.utc) - timedelta(hours=2)
        recent = datetime.now(timezone.utc)
        t.record_prediction("old_1", 0.9, label=1, timestamp=old)
        t.record_prediction("recent_1", 0.9, label=1, timestamp=recent)
        # 1-hour window should only include the recent record.
        snap = t.snapshot(window=timedelta(hours=1))
        assert snap.n_total == 1
        assert snap.n_labelled == 1

    def test_snapshot_auroc_perfect_separation(self):
        t = PerformanceTracker(threshold=0.5)
        for i, lbl in enumerate([0, 0, 0, 1, 1, 1]):
            t.record_prediction(f"tx_{i}", 0.1 + 0.15 * i, label=lbl)
        snap = t.snapshot()
        assert snap.auroc == pytest.approx(1.0, abs=1e-9)
        assert snap.auprc == pytest.approx(1.0, abs=1e-9)

    def test_snapshot_handles_single_class_labels(self):
        t = PerformanceTracker()
        for i in range(5):
            t.record_prediction(f"tx_{i}", 0.5, label=0)
        snap = t.snapshot()
        # AUROC undefined with one class -> None.
        assert snap.auroc is None
        # AUPRC requires at least one positive -> None.
        assert snap.auprc is None

    def test_to_dict_round_trip(self):
        t = PerformanceTracker()
        t.record_prediction("tx_1", 0.9, label=1)
        snap = t.snapshot()
        d = snap.to_dict()
        assert d["n_labelled"] == 1
        assert isinstance(d["window_start"], str)
        assert isinstance(d["generated_at"], str)

    def test_thread_safety_can_be_locked(self):
        # We don't run an actual race here but exercise the lock path.
        t = PerformanceTracker()
        t.record_prediction("tx_1", 0.5, label=1)
        with t._lock:
            assert t.n_records == 1
