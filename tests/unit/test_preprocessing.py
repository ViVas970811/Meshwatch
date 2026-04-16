"""Tests for ``fraud_detection.data.preprocessing``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.data.preprocessing import (
    AMOUNT_COL,
    IEEECISPreprocessor,
    group_columns,
)
from fraud_detection.utils.config import AppConfig

# ---------------------------------------------------------------------------
# group_columns
# ---------------------------------------------------------------------------


def test_group_columns_classifies_families(synthetic_ieee_df: pd.DataFrame):
    groups = group_columns(synthetic_ieee_df)
    assert set(groups.v_features) == {"V1", "V99", "V250"}
    assert set(groups.d_features) == {"D1", "D2"}
    assert set(groups.c_features) == {"C1", "C2"}
    assert set(groups.m_features) == {"M1", "M2"}
    assert set(groups.id_numeric) == {"id_01", "id_02"}
    assert set(groups.id_categorical) == {"id_30", "id_31"}
    assert set(groups.email_cols) == {"P_emaildomain", "R_emaildomain"}


def test_group_columns_respects_reserved(synthetic_ieee_df: pd.DataFrame):
    groups = group_columns(synthetic_ieee_df, target="isFraud", time_col="TransactionDT")
    # Target / time / join key should not appear anywhere.
    all_grouped = (
        groups.v_features
        + groups.d_features
        + groups.c_features
        + groups.m_features
        + groups.id_numeric
        + groups.id_categorical
        + groups.email_cols
        + groups.other_categorical
        + groups.other_numeric
    )
    assert "isFraud" not in all_grouped
    assert "TransactionDT" not in all_grouped
    assert "TransactionID" not in all_grouped


# ---------------------------------------------------------------------------
# handle_missing_values
# ---------------------------------------------------------------------------


def _make_preprocessor() -> IEEECISPreprocessor:
    cfg = AppConfig()
    return IEEECISPreprocessor(cfg)


def test_handle_missing_drops_v_columns_above_threshold(synthetic_ieee_df: pd.DataFrame):
    pp = _make_preprocessor()
    out = pp.handle_missing_values(synthetic_ieee_df)

    # V250 is 100% missing -> must be dropped.
    assert "V250" in pp.state.dropped_v_columns
    assert "V250" not in out.columns
    # V1 is mostly present -> must be kept.
    assert "V1" in out.columns


def test_handle_missing_fills_and_adds_indicators(synthetic_ieee_df: pd.DataFrame):
    pp = _make_preprocessor()
    out = pp.handle_missing_values(synthetic_ieee_df)

    # No NaNs remain in any feature column after fill.
    for col in ("V1", "V99", "D1", "C2", "M1", "id_01", "id_30", "P_emaildomain"):
        if col in out.columns:
            assert out[col].isna().sum() == 0, f"{col} still has NaNs"

    # Indicator columns created for V and D and id_numeric groups.
    assert "V1__isna" in out.columns
    assert "D1__isna" in out.columns
    assert "id_01__isna" in out.columns
    # M / emails use "missing"/"unknown" sentinels, no indicator.
    assert "M1__isna" not in out.columns
    assert "P_emaildomain__isna" not in out.columns


def test_handle_missing_is_stable_under_transform(synthetic_ieee_df: pd.DataFrame):
    pp = _make_preprocessor()
    # Fit on first half, transform second half -> same dropped V-cols applied.
    mid = len(synthetic_ieee_df) // 2
    _ = pp.fit_transform(synthetic_ieee_df.iloc[:mid])
    dropped = list(pp.state.dropped_v_columns)

    out = pp.transform(synthetic_ieee_df.iloc[mid:])
    for col in dropped:
        assert col not in out.columns


# ---------------------------------------------------------------------------
# encode_categoricals
# ---------------------------------------------------------------------------


def test_encode_categoricals_outputs_integers(synthetic_ieee_df: pd.DataFrame):
    pp = _make_preprocessor()
    step1 = pp.handle_missing_values(synthetic_ieee_df)
    out = pp.encode_categoricals(step1)
    for col in ("P_emaildomain", "M1", "id_30", "DeviceType"):
        if col in out.columns:
            assert pd.api.types.is_integer_dtype(out[col]), f"{col} not integer-encoded"


def test_encode_categoricals_handles_unseen_categories(synthetic_ieee_df: pd.DataFrame):
    pp = _make_preprocessor()
    _ = pp.fit_transform(synthetic_ieee_df)

    # Construct a frame where P_emaildomain has a previously unseen value.
    novel = synthetic_ieee_df.head(5).copy()
    novel["P_emaildomain"] = "brandnew.xyz"
    out = pp.transform(novel)
    # Unseen categories map to 0 (reserved for "unknown").
    assert (out["P_emaildomain"] == 0).all()


# ---------------------------------------------------------------------------
# normalize_numerics
# ---------------------------------------------------------------------------


def test_normalize_numerics_adds_log_amount(synthetic_ieee_df: pd.DataFrame):
    # Disable scaling so we can check that log1p is always >= 0 pre-scale.
    cfg = AppConfig()
    cfg.preprocessing.normalize = "none"
    cfg.preprocessing.clip_quantile = 1.0  # disable clipping
    pp = IEEECISPreprocessor(cfg)
    out = pp.fit_transform(synthetic_ieee_df)
    assert f"{AMOUNT_COL}__log1p" in out.columns
    # log1p(x) >= 0 for x >= 0.
    assert (out[f"{AMOUNT_COL}__log1p"] >= 0).all()


def test_normalize_numerics_scales_log_amount_column(synthetic_ieee_df: pd.DataFrame):
    # When scaling is enabled, the log1p column participates in standardization.
    pp = _make_preprocessor()
    out = pp.fit_transform(synthetic_ieee_df)
    assert f"{AMOUNT_COL}__log1p" in out.columns
    # After standardization, no NaNs and finite values.
    col = out[f"{AMOUNT_COL}__log1p"]
    assert col.notna().all()
    assert np.isfinite(col).all()


def test_normalize_numerics_zero_mean_unit_var(synthetic_ieee_df: pd.DataFrame):
    cfg = AppConfig()
    cfg.preprocessing.normalize = "standard"
    pp = IEEECISPreprocessor(cfg)
    out = pp.fit_transform(synthetic_ieee_df)

    # At least one numeric column should be ~0-mean / ~1-std.
    cols = [c for c in pp.state.numeric_columns if c in out.columns]
    assert cols
    sample = out[cols[0]].to_numpy()
    # Allow some tolerance -- StandardScaler uses n not n-1 divisor.
    assert np.isfinite(sample).all()
    assert abs(sample.mean()) < 1e-6
    assert abs(sample.std(ddof=0) - 1.0) < 1e-6


def test_normalize_numerics_none_preserves_values(synthetic_ieee_df: pd.DataFrame):
    cfg = AppConfig()
    cfg.preprocessing.normalize = "none"
    cfg.preprocessing.clip_quantile = 1.0  # disable clipping
    pp = IEEECISPreprocessor(cfg)
    out = pp.fit_transform(synthetic_ieee_df)
    # Amount column isn't scaled; values should match raw within rounding.
    assert out[AMOUNT_COL].iloc[0] == pytest.approx(synthetic_ieee_df[AMOUNT_COL].iloc[0], rel=1e-6)


# ---------------------------------------------------------------------------
# fit / transform / save / load
# ---------------------------------------------------------------------------


def test_fit_transform_and_transform_consistency(synthetic_ieee_df: pd.DataFrame):
    pp = _make_preprocessor()
    first = pp.fit_transform(synthetic_ieee_df)
    again = pp.transform(synthetic_ieee_df)
    # Shapes should match and numeric columns should be identical.
    assert first.shape == again.shape
    for col in pp.state.numeric_columns:
        if col in first.columns:
            np.testing.assert_allclose(first[col].values, again[col].values, atol=1e-10)


def test_save_and_load_roundtrip(tmp_path: Path, synthetic_ieee_df: pd.DataFrame):
    pp = _make_preprocessor()
    fitted = pp.fit_transform(synthetic_ieee_df)

    state_path = tmp_path / "pp.pkl"
    pp.save(state_path)
    assert state_path.exists()
    assert (state_path.with_suffix(".summary.json")).exists()

    loaded = IEEECISPreprocessor.load(state_path)
    out = loaded.transform(synthetic_ieee_df)
    assert out.shape == fitted.shape
    # Numeric columns match bit-for-bit.
    for col in pp.state.numeric_columns:
        if col in fitted.columns:
            np.testing.assert_allclose(out[col].values, fitted[col].values, atol=1e-10)


def test_transform_before_fit_raises():
    pp = _make_preprocessor()
    with pytest.raises(RuntimeError, match="not been fit"):
        pp.transform(pd.DataFrame({"TransactionID": [1]}))


def test_no_unexpected_nans_post_pipeline(synthetic_ieee_df: pd.DataFrame):
    """Phase 1 acceptance: 'Processed DataFrame has no unexpected NaN values'."""
    pp = _make_preprocessor()
    out = pp.fit_transform(synthetic_ieee_df)
    # No NaNs anywhere.
    na_counts = out.isna().sum()
    assert (na_counts == 0).all(), f"Unexpected NaNs in: {na_counts[na_counts > 0].to_dict()}"
