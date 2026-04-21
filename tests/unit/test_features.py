"""Tests for the Phase 2 feature builders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.features.aggregated import (
    AGGREGATED_FEATURES,
    ALL_AGGREGATED_FEATURES,
    IDENTITY_FEATURES,
    AggregatedFeatureBuilder,
)
from fraud_detection.features.graph_features import GRAPH_FEATURES, GraphFeatureBuilder
from fraud_detection.features.pipeline import ALL_ENGINEERED_FEATURES, FeaturePipeline
from fraud_detection.features.temporal import (
    ALL_TEMPORAL_AMOUNT_FEATURES,
    AMOUNT_FEATURES,
    TEMPORAL_FEATURES,
    TemporalFeatureBuilder,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def feature_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 250
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "isFraud": rng.choice([0, 1], size=n, p=[0.95, 0.05]).astype(np.int8),
            "TransactionDT": np.sort(rng.integers(0, 30 * 86400, size=n)).astype(np.int64),
            "TransactionAmt": np.abs(rng.normal(75, 40, n)).clip(min=1.0),
            "ProductCD": rng.choice(list("WCHSR"), size=n),
            "card1": rng.integers(1000, 1050, n),
            "addr1": rng.integers(100, 500, n).astype(float),
            "addr2": rng.integers(10, 99, n).astype(float),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com"], n),
            "R_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com"], n),
            "DeviceType": rng.choice(["desktop", "mobile"], n),
            "DeviceInfo": rng.choice(["Windows", "iPhone", "Android"], n),
            "id_01": rng.normal(size=n),
            "id_02": rng.integers(0, 1000, n).astype(float),
        }
    )
    for i in range(307, 317):
        df[f"V{i}"] = rng.normal(size=n)
    for i in range(1, 15):
        df[f"C{i}"] = rng.integers(0, 10, n).astype(np.float32)
    for i in range(1, 10):
        df[f"M{i}"] = rng.choice([0, 1], n).astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Temporal builder
# ---------------------------------------------------------------------------


def test_temporal_outputs_39_features(feature_df: pd.DataFrame):
    tb = TemporalFeatureBuilder()
    out = tb.fit_transform(feature_df)
    assert out.shape[1] == 39
    assert len(TEMPORAL_FEATURES) == 23
    assert len(AMOUNT_FEATURES) == 16
    assert len(ALL_TEMPORAL_AMOUNT_FEATURES) == 39


def test_temporal_has_no_nans(feature_df: pd.DataFrame):
    out = TemporalFeatureBuilder().fit_transform(feature_df)
    assert out.isna().sum().sum() == 0


def test_temporal_transform_idempotent(feature_df: pd.DataFrame):
    tb = TemporalFeatureBuilder()
    out1 = tb.fit_transform(feature_df)
    out2 = tb.transform(feature_df)
    np.testing.assert_allclose(out1.to_numpy(np.float64), out2.to_numpy(np.float64), atol=1e-10)


def test_temporal_cyclical_features_bounded(feature_df: pd.DataFrame):
    out = TemporalFeatureBuilder().fit_transform(feature_df)
    for col in ("feat_hour_sin", "feat_hour_cos", "feat_dow_sin", "feat_dow_cos"):
        assert out[col].between(-1.0, 1.0).all(), f"{col} outside [-1, 1]"


def test_temporal_seconds_since_last_non_negative(feature_df: pd.DataFrame):
    out = TemporalFeatureBuilder().fit_transform(feature_df)
    assert (out["feat_seconds_since_last_txn"] >= 0).all()
    assert (out["feat_seconds_since_last_card_txn"] >= 0).all()


def test_temporal_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="must be fit"):
        TemporalFeatureBuilder().transform(pd.DataFrame())


# ---------------------------------------------------------------------------
# Aggregated builder
# ---------------------------------------------------------------------------


def test_aggregated_outputs_52_features(feature_df: pd.DataFrame):
    ab = AggregatedFeatureBuilder()
    out = ab.fit_transform(feature_df)
    assert out.shape[1] == 52
    assert len(AGGREGATED_FEATURES) == 36
    assert len(IDENTITY_FEATURES) == 16
    assert len(ALL_AGGREGATED_FEATURES) == 52


def test_aggregated_has_no_nans(feature_df: pd.DataFrame):
    out = AggregatedFeatureBuilder().fit_transform(feature_df)
    assert out.isna().sum().sum() == 0, f"NaNs in: {out.columns[out.isna().any()].tolist()}"


def test_aggregated_transform_idempotent(feature_df: pd.DataFrame):
    ab = AggregatedFeatureBuilder()
    out1 = ab.fit_transform(feature_df)
    out2 = ab.transform(feature_df)
    np.testing.assert_allclose(out1.to_numpy(np.float64), out2.to_numpy(np.float64), atol=1e-10)


def test_aggregated_fraud_rate_in_unit_interval(feature_df: pd.DataFrame):
    out = AggregatedFeatureBuilder().fit_transform(feature_df)
    for col in (
        "feat_card_fraud_rate",
        "feat_email_fraud_rate",
        "feat_addr_fraud_rate",
        "feat_device_fraud_rate",
    ):
        assert out[col].between(0.0, 1.0).all(), f"{col} out of [0, 1]"


def test_aggregated_email_mismatch_is_binary(feature_df: pd.DataFrame):
    out = AggregatedFeatureBuilder().fit_transform(feature_df)
    assert set(out["feat_email_mismatch"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Graph features builder
# ---------------------------------------------------------------------------


def test_graph_features_outputs_28(feature_df: pd.DataFrame):
    gb = GraphFeatureBuilder()
    train_mask = pd.Series([True] * len(feature_df))
    out = gb.fit_transform(feature_df, training_mask=train_mask)
    assert out.shape[1] == 28
    assert len(GRAPH_FEATURES) == 28


def test_graph_features_no_nans(feature_df: pd.DataFrame):
    out = GraphFeatureBuilder().fit_transform(feature_df)
    assert out.isna().sum().sum() == 0


def test_graph_degree_features_positive(feature_df: pd.DataFrame):
    out = GraphFeatureBuilder().fit_transform(feature_df)
    for col in (
        "feat_gr_card_degree",
        "feat_gr_addr_degree",
        "feat_gr_email_degree",
        "feat_gr_device_degree",
        "feat_gr_ip_degree",
    ):
        assert (out[col] > 0).all(), f"{col} has zero/negative values"


def test_graph_transform_idempotent(feature_df: pd.DataFrame):
    gb = GraphFeatureBuilder()
    out1 = gb.fit_transform(feature_df)
    out2 = gb.transform(feature_df)
    np.testing.assert_allclose(out1.to_numpy(np.float64), out2.to_numpy(np.float64), atol=1e-10)


def test_graph_component_size_at_least_1(feature_df: pd.DataFrame):
    out = GraphFeatureBuilder().fit_transform(feature_df)
    assert (out["feat_gr_component_size"] >= 1).all()


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def test_pipeline_outputs_119_features_plus_id_cols(feature_df: pd.DataFrame):
    pipeline = FeaturePipeline()
    out = pipeline.fit_transform(feature_df, training_mask=pd.Series([True] * len(feature_df)))
    assert len(ALL_ENGINEERED_FEATURES) == 119
    # ID cols: TransactionID, isFraud, TransactionDT = 3 extras.
    assert out.shape[1] == 122


def test_pipeline_has_no_nan(feature_df: pd.DataFrame):
    out = FeaturePipeline().fit_transform(feature_df)
    assert out.isna().sum().sum() == 0


def test_pipeline_save_load_roundtrip(tmp_path: Path, feature_df: pd.DataFrame):
    pipeline = FeaturePipeline()
    first = pipeline.fit_transform(feature_df)
    p = tmp_path / "pipeline.pkl"
    pipeline.save(p)
    assert p.exists()
    loaded = FeaturePipeline.load(p)
    second = loaded.transform(feature_df)
    # Compare only feature columns; IDs may have dtype differences.
    feat_cols = [c for c in first.columns if c.startswith("feat_")]
    np.testing.assert_allclose(
        first[feat_cols].to_numpy(np.float64),
        second[feat_cols].to_numpy(np.float64),
        atol=1e-10,
    )


def test_pipeline_save_before_fit_raises(tmp_path: Path):
    with pytest.raises(RuntimeError, match="not been fit"):
        FeaturePipeline().save(tmp_path / "x.pkl")
