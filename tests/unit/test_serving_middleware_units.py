"""Unit tests for the no-op fallback paths inside ``serving.middleware``.

These cover branches that only fire when ``prometheus_client`` is
unavailable -- which doesn't happen in our test environment by default,
so we exercise the no-op class directly.
"""

from __future__ import annotations

from fraud_detection.serving.middleware import _MetricsRegistry, _NoopMetric


class TestNoopMetric:
    def test_labels_returns_self(self):
        m = _NoopMetric()
        assert m.labels("a", "b") is m

    def test_inc_dec_observe_set_all_return_none(self):
        m = _NoopMetric()
        assert m.inc() is None
        assert m.inc(2) is None
        assert m.dec() is None
        assert m.observe(0.42) is None
        assert m.set(1.23) is None


class TestMetricsRegistryRender:
    def test_render_returns_bytes(self):
        # Whichever path -- prometheus_client present or absent -- render
        # must always return bytes.
        body = _MetricsRegistry().render()
        assert isinstance(body, bytes)

    def test_content_type_is_set(self):
        reg = _MetricsRegistry()
        assert isinstance(reg.content_type, str)
        assert "text" in reg.content_type or "openmetrics" in reg.content_type
