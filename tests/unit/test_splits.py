"""Tests for ``fraud_detection.data.splits``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.data.splits import SplitResult, TemporalSplitter
from fraud_detection.utils.config import AppConfig

# ---------------------------------------------------------------------------
# Core chronological correctness
# ---------------------------------------------------------------------------


def _make_frame(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Strictly-increasing timestamps so the boundary assertion is meaningful.
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": np.sort(rng.integers(0, 10**9, size=n)),
            "isFraud": rng.choice([0, 1], size=n, p=[0.95, 0.05]),
            "value": rng.normal(size=n),
        }
    )


def test_basic_split_ratios_and_totals():
    df = _make_frame(1000)
    splitter = TemporalSplitter(AppConfig())
    res = splitter.split(df)

    assert len(res.train) + len(res.val) + len(res.test) == 1000
    assert len(res.train) == 600  # 60%
    assert len(res.val) == 200  # 20%
    assert len(res.test) == 200  # 20%


def test_non_overlap_assertion_holds_on_unique_times():
    df = _make_frame(500)
    res = TemporalSplitter(AppConfig()).split(df)
    # Strict non-overlap: train_max <= val_min and val_max <= test_min.
    assert res.train["TransactionDT"].max() <= res.val["TransactionDT"].min()
    assert res.val["TransactionDT"].max() <= res.test["TransactionDT"].min()


def test_non_overlap_assertion_is_triggered_when_disabled_still_valid():
    # Even with ties (duplicates), the stable sort pushes earlier rows first,
    # so the non-overlap inequality still holds weakly.
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": np.repeat(np.arange(n // 3), 3),  # many ties
            "isFraud": rng.choice([0, 1], size=n),
        }
    )
    res = TemporalSplitter(AppConfig()).split(df)
    assert res.train["TransactionDT"].max() <= res.val["TransactionDT"].min()
    assert res.val["TransactionDT"].max() <= res.test["TransactionDT"].min()


def test_custom_fractions_via_config():
    df = _make_frame(1000)
    cfg = AppConfig(splits={"train_frac": 0.7, "val_frac": 0.15, "test_frac": 0.15})
    res = TemporalSplitter(cfg).split(df)
    assert len(res.train) == 700
    assert len(res.val) == 150
    assert len(res.test) == 150


def test_summary_contains_expected_keys():
    df = _make_frame(500)
    res = TemporalSplitter(AppConfig()).split(df)
    s = res.summary()
    for key in (
        "n_train",
        "n_val",
        "n_test",
        "boundary_train_val",
        "boundary_val_test",
        "time_column",
        "fraud_rate_train",
        "fraud_rate_val",
        "fraud_rate_test",
    ):
        assert key in s


def test_save_parquet_creates_three_files(tmp_path: Path):
    df = _make_frame(200)
    res = TemporalSplitter(AppConfig()).split(df)
    paths = res.save_parquet(tmp_path)
    for name in ("train", "val", "test"):
        assert paths[name].exists()
        loaded = pd.read_parquet(paths[name])
        # Column set should match the input.
        assert set(loaded.columns) == set(df.columns)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_missing_time_column_raises():
    bad = pd.DataFrame({"TransactionID": [1, 2, 3], "isFraud": [0, 0, 1]})
    with pytest.raises(ValueError, match="not found"):
        TemporalSplitter(AppConfig()).split(bad)


def test_empty_frame_raises():
    empty = pd.DataFrame({"TransactionDT": pd.Series(dtype=int)})
    with pytest.raises(ValueError, match="empty"):
        TemporalSplitter(AppConfig()).split(empty)


def test_nan_in_time_column_raises():
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [1.0, 2.0, np.nan, 4.0],
            "isFraud": [0, 0, 0, 1],
        }
    )
    with pytest.raises(ValueError, match="NaN"):
        TemporalSplitter(AppConfig()).split(df)


def test_non_temporal_strategy_raises():
    cfg = AppConfig(
        splits={
            "strategy": "random",
            "train_frac": 0.6,
            "val_frac": 0.2,
            "test_frac": 0.2,
        }
    )
    df = _make_frame(50)
    with pytest.raises(ValueError, match="temporal"):
        TemporalSplitter(cfg).split(df)


def test_split_result_is_dataclass_like():
    df = _make_frame(100)
    res = TemporalSplitter(AppConfig()).split(df)
    assert isinstance(res, SplitResult)
    assert res.time_column == "TransactionDT"
