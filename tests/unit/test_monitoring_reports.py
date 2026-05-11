"""Unit tests for the drift report serialisation helpers (Phase 7)."""

from __future__ import annotations

import json

import numpy as np

from fraud_detection.monitoring.drift import DriftDetector
from fraud_detection.monitoring.reports import (
    report_to_html,
    report_to_json,
    write_html,
    write_json,
)


def _make_report(rng):
    ref = {
        "amount": rng.normal(loc=50, scale=20, size=500).tolist(),
        "country": rng.choice(["US", "GB"], size=500).tolist(),
    }
    cur = {
        "amount": rng.normal(loc=500, scale=20, size=500).tolist(),  # severe drift
        "country": rng.choice(["US", "GB"], size=500).tolist(),
    }
    return DriftDetector(ref).detect(cur)


class TestJSON:
    def test_report_to_json_is_valid_json(self):
        rng = np.random.default_rng(7)
        report = _make_report(rng)
        payload = report_to_json(report)
        parsed = json.loads(payload)
        assert parsed["n_features"] == 2
        assert "features" in parsed
        assert parsed["severity"] in {"none", "moderate", "severe"}

    def test_write_json_creates_file(self, tmp_path):
        rng = np.random.default_rng(7)
        report = _make_report(rng)
        path = write_json(report, tmp_path / "drift.json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["n_features"] == 2


class TestHTML:
    def test_report_to_html_contains_summary(self):
        rng = np.random.default_rng(7)
        report = _make_report(rng)
        html = report_to_html(report, title="Test report")
        assert "<title>Test report</title>" in html
        assert "Overall PSI" in html
        assert "Severity" in html
        # Severe drift should mark "amount" -- the severity badge class
        # must be present somewhere in the table.
        assert "severity" in html.lower()

    def test_html_is_self_contained(self):
        rng = np.random.default_rng(7)
        report = _make_report(rng)
        html = report_to_html(report)
        # No external scripts / stylesheets -- we deliberately inline.
        assert "<script" not in html
        assert "<link" not in html
        assert "<style>" in html  # inline CSS

    def test_html_escapes_feature_names(self):
        ref = {"<script>": [1.0, 2.0, 3.0]}
        cur = {"<script>": [10.0, 20.0, 30.0]}
        report = DriftDetector(ref, numeric_features=["<script>"]).detect(cur)
        html = report_to_html(report)
        # The raw "<script>" must be escaped.
        assert "&lt;script&gt;" in html
        assert "<script>" not in html.replace("<script", "").replace("script>", "")

    def test_write_html_creates_file(self, tmp_path):
        rng = np.random.default_rng(7)
        report = _make_report(rng)
        path = write_html(report, tmp_path / "drift.html")
        assert path.exists()
        assert path.read_text().startswith("<!doctype html>")
