"""Temporal train/val/test splits for the IEEE-CIS dataset.

The plan's acceptance criterion for Phase 1 is:

    > Temporal splits pass non-overlap assertion

We sort by ``TransactionDT`` and slice chronologically 60/20/20 so no
validation or test row has a timestamp strictly before the end of the
training slice. This prevents future-to-past leakage that would inflate
offline metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.utils.config import AppConfig, SplitsConfig, load_config
from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Split container
# ---------------------------------------------------------------------------


@dataclass
class SplitResult:
    """Result of a temporal split."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    time_column: str
    boundaries: tuple[Any, Any] = field(default=(None, None))  # (train/val, val/test)

    # ---- Introspection ---------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Human-readable summary for logs / notebooks."""

        def _rate(df: pd.DataFrame, col: str) -> float | None:
            if col not in df.columns:
                return None
            return float(df[col].mean())

        target = None
        for candidate in ("isFraud", "label", "target"):
            if candidate in self.train.columns:
                target = candidate
                break

        out = {
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_test": len(self.test),
            "boundary_train_val": self.boundaries[0],
            "boundary_val_test": self.boundaries[1],
            "time_column": self.time_column,
        }
        if target is not None:
            out["fraud_rate_train"] = _rate(self.train, target)
            out["fraud_rate_val"] = _rate(self.val, target)
            out["fraud_rate_test"] = _rate(self.test, target)
        return out

    # ---- Persistence -----------------------------------------------------

    def save_parquet(self, output_dir: str | Path) -> dict[str, Path]:
        """Write the three splits as parquet files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "train": output_dir / "train.parquet",
            "val": output_dir / "val.parquet",
            "test": output_dir / "test.parquet",
        }
        for name, path in paths.items():
            df = getattr(self, name)
            df.to_parquet(path, index=False)
        log.info("splits_saved", **{k: str(v) for k, v in paths.items()})
        return paths


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------


class TemporalSplitter:
    """Split a DataFrame chronologically by a time column.

    The *fractions* (train/val/test) define slice sizes along the time axis.
    Ties in the time column are broken by stable sort order, so boundary
    rows with the same timestamp always land in the earlier split -- no
    future leakage.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        time_column: str | None = None,
    ) -> None:
        self.config: AppConfig = config or load_config()
        self.splits_cfg: SplitsConfig = self.config.splits
        self.time_column: str = time_column or self.config.dataset.time_column

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, df: pd.DataFrame) -> SplitResult:
        """Return a :class:`SplitResult` with the three chronological slices.

        Raises
        ------
        ValueError
            If the time column is missing, the dataframe is empty, or the
            non-overlap assertion fails.
        """
        self._validate_input(df)

        cfg = self.splits_cfg
        if cfg.strategy != "temporal":
            msg = f"TemporalSplitter only supports strategy='temporal' (got '{cfg.strategy}')."
            raise ValueError(msg)

        ordered = df.sort_values(self.time_column, kind="mergesort").reset_index(drop=True)

        n = len(ordered)
        n_train = int(n * cfg.train_frac)
        n_val = int(n * cfg.val_frac)
        # Assign the remainder to test to avoid losing rows from integer truncation.
        n_test = n - n_train - n_val

        train_df = ordered.iloc[:n_train].copy()
        val_df = ordered.iloc[n_train : n_train + n_val].copy()
        test_df = ordered.iloc[n_train + n_val :].copy()

        boundaries = self._boundaries(train_df, val_df, test_df)

        result = SplitResult(
            train=train_df,
            val=val_df,
            test=test_df,
            time_column=self.time_column,
            boundaries=boundaries,
        )

        if cfg.assert_non_overlap:
            self._assert_non_overlap(result)

        log.info(
            "temporal_split_complete",
            n_total=n,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            boundary_train_val=boundaries[0],
            boundary_val_test=boundaries[1],
        )
        return result

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_input(self, df: pd.DataFrame) -> None:
        if self.time_column not in df.columns:
            msg = (
                f"Time column '{self.time_column}' not found in dataframe "
                f"(columns: {list(df.columns)[:10]}{'...' if df.shape[1] > 10 else ''})."
            )
            raise ValueError(msg)
        if df.empty:
            msg = "Cannot split an empty dataframe."
            raise ValueError(msg)
        if df[self.time_column].isna().any():
            n_na = int(df[self.time_column].isna().sum())
            msg = f"Time column '{self.time_column}' has {n_na} NaN values -- drop or impute first."
            raise ValueError(msg)

    def _boundaries(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ) -> tuple[Any, Any]:
        train_max = train[self.time_column].max() if not train.empty else None
        val_max = val[self.time_column].max() if not val.empty else None
        # Cast numpy scalars to native python for cleaner logs / json.
        return _to_native(train_max), _to_native(val_max)

    def _assert_non_overlap(self, r: SplitResult) -> None:
        t = r.time_column
        train_max = r.train[t].max() if not r.train.empty else None
        val_min = r.val[t].min() if not r.val.empty else None
        val_max = r.val[t].max() if not r.val.empty else None
        test_min = r.test[t].min() if not r.test.empty else None

        if train_max is not None and val_min is not None and val_min < train_max:
            msg = (
                f"Non-overlap assertion failed: val starts at {val_min} "
                f"which is before train ends at {train_max}."
            )
            raise AssertionError(msg)
        if val_max is not None and test_min is not None and test_min < val_max:
            msg = (
                f"Non-overlap assertion failed: test starts at {test_min} "
                f"which is before val ends at {val_max}."
            )
            raise AssertionError(msg)


def _to_native(val: Any) -> Any:
    """Convert numpy / pandas scalars to Python primitives for JSON-friendly logs."""
    if val is None or isinstance(val, (int, float, str, bool)):
        return val
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    return val


__all__ = ["SplitResult", "TemporalSplitter"]
