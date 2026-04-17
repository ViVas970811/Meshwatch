"""IEEE-CIS preprocessing pipeline.

This module implements the missing-value strategy from the implementation
plan (page 4) and a stateful fit/transform API so the *same* transformation
is applied at train and serve time.

The pipeline has four stages:

1. :meth:`IEEECISPreprocessor.load_raw`
   Load transaction + identity CSVs and left-join on TransactionID.
2. :meth:`IEEECISPreprocessor.handle_missing_values`
   Drop V-columns that exceed a missing-fraction threshold, then fill the
   remaining per-group missing values with sentinels and (optionally) add
   binary indicator columns.
3. :meth:`IEEECISPreprocessor.encode_categoricals`
   Integer-encode string columns using stable per-category dictionaries
   learned during fit.
4. :meth:`IEEECISPreprocessor.normalize_numerics`
   Scale numeric columns with StandardScaler/RobustScaler (configurable),
   optionally clipping at an upper quantile and adding ``log1p(amount)``.

Call :meth:`fit_transform` in training and :meth:`transform` in serving.
"""

from __future__ import annotations

import json
import pickle
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from fraud_detection.utils.config import (
    AppConfig,
    MissingGroupStrategy,
    PreprocessingConfig,
    load_config,
)
from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Column-group regexes (compiled once)
# ---------------------------------------------------------------------------

_V_RE = re.compile(r"^V\d+$")
_D_RE = re.compile(r"^D\d+$")
_C_RE = re.compile(r"^C\d+$")
_M_RE = re.compile(r"^M\d+$")
_ID_RE = re.compile(r"^id[_-]?\d+$")  # tolerates both 'id_01' and 'id-01'

EMAIL_COLS: tuple[str, ...] = ("P_emaildomain", "R_emaildomain")
DEVICE_CAT_COLS: tuple[str, ...] = ("DeviceType", "DeviceInfo")
PRODUCT_COL = "ProductCD"
CARD_COLS: tuple[str, ...] = ("card1", "card2", "card3", "card4", "card5", "card6")
ADDR_COLS: tuple[str, ...] = ("addr1", "addr2")
AMOUNT_COL = "TransactionAmt"
TARGET_COL_DEFAULT = "isFraud"
TIME_COL_DEFAULT = "TransactionDT"


# ---------------------------------------------------------------------------
# Fitted state (persistable)
# ---------------------------------------------------------------------------


@dataclass
class FittedState:
    """Everything learned during :meth:`fit` so we can replay it at serve time."""

    dropped_v_columns: list[str] = field(default_factory=list)
    indicator_columns: list[str] = field(default_factory=list)
    categorical_mappings: dict[str, dict[Any, int]] = field(default_factory=dict)
    numeric_columns: list[str] = field(default_factory=list)
    # Columns the catch-all sweep saw NaN in at fit time -- must be re-applied
    # at transform time even if the transform batch happens to have no NaN,
    # otherwise the downstream scaler receives a different feature count.
    leaked_numeric_columns: list[str] = field(default_factory=list)
    leaked_string_columns: list[str] = field(default_factory=list)
    amount_clip_value: float | None = None
    scaler: Any | None = None  # StandardScaler | RobustScaler | None
    n_features_out: int = 0

    # ---- Persistence -----------------------------------------------------

    def to_file(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, path: str | Path) -> FittedState:
        with Path(path).open("rb") as f:
            state: FittedState = pickle.load(f)
        return state

    # ---- Introspection ---------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "dropped_v_columns": len(self.dropped_v_columns),
            "indicator_columns": len(self.indicator_columns),
            "categorical_columns": len(self.categorical_mappings),
            "numeric_columns": len(self.numeric_columns),
            "amount_clip_value": self.amount_clip_value,
            "scaler": type(self.scaler).__name__ if self.scaler is not None else None,
            "n_features_out": self.n_features_out,
        }


# ---------------------------------------------------------------------------
# Column grouping helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnGroups:
    """Columns grouped by the IEEE-CIS schema families."""

    v_features: list[str]
    d_features: list[str]
    c_features: list[str]
    m_features: list[str]
    id_numeric: list[str]
    id_categorical: list[str]
    email_cols: list[str]
    other_categorical: list[str]
    other_numeric: list[str]


def _detect_id_numeric_vs_categorical(
    df: pd.DataFrame, id_cols: Iterable[str]
) -> tuple[list[str], list[str]]:
    """id_01..id_38 is a mix of numeric and string columns -- split by dtype."""
    numeric, categorical = [], []
    for col in id_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def group_columns(
    df: pd.DataFrame,
    *,
    target: str = TARGET_COL_DEFAULT,
    time_col: str = TIME_COL_DEFAULT,
) -> ColumnGroups:
    """Partition ``df.columns`` into the IEEE-CIS feature families."""
    cols = list(df.columns)

    v_features = [c for c in cols if _V_RE.match(c)]
    d_features = [c for c in cols if _D_RE.match(c)]
    c_features = [c for c in cols if _C_RE.match(c)]
    m_features = [c for c in cols if _M_RE.match(c)]
    id_cols = [c for c in cols if _ID_RE.match(c)]
    id_numeric, id_categorical = _detect_id_numeric_vs_categorical(df, id_cols)
    email_cols = [c for c in EMAIL_COLS if c in cols]

    # The known_categorical set = M + string id + emails + device + ProductCD + card4/card6.
    # Anything else that isn't numeric and isn't the target/time becomes "other_categorical".
    known_categorical = set(
        m_features
        + id_categorical
        + email_cols
        + list(DEVICE_CAT_COLS)
        + [PRODUCT_COL, "card4", "card6"]
    )

    reserved = {target, time_col, "TransactionID"}
    other_categorical: list[str] = []
    other_numeric: list[str] = []
    handled = set(
        v_features + d_features + c_features + m_features + id_numeric + id_categorical + email_cols
    )

    for col in cols:
        if col in reserved or col in handled:
            continue
        if col in known_categorical:
            other_categorical.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            other_numeric.append(col)
        else:
            other_categorical.append(col)

    return ColumnGroups(
        v_features=v_features,
        d_features=d_features,
        c_features=c_features,
        m_features=m_features,
        id_numeric=id_numeric,
        id_categorical=id_categorical,
        email_cols=email_cols,
        other_categorical=other_categorical,
        other_numeric=other_numeric,
    )


# ---------------------------------------------------------------------------
# The preprocessor
# ---------------------------------------------------------------------------


class IEEECISPreprocessor:
    """Stateful preprocessor for the IEEE-CIS Fraud Detection dataset.

    Parameters
    ----------
    config
        Optional :class:`AppConfig`. If ``None`` we load the default YAML.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config: AppConfig = config or load_config()
        self.pp: PreprocessingConfig = self.config.preprocessing
        self.state: FittedState = FittedState()
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Stage 1: load
    # ------------------------------------------------------------------

    def load_raw(
        self,
        *,
        transaction_path: Path | str | None = None,
        identity_path: Path | str | None = None,
        nrows: int | None = None,
    ) -> pd.DataFrame:
        """Load and left-join the transaction + identity tables.

        Parameters
        ----------
        transaction_path, identity_path
            Override paths; default to ``data/raw/train_transaction.csv`` etc.
        nrows
            Optional row cap (useful in tests / dev subset mode).

        Notes
        -----
        On 16GB machines the full 590K * ~430-column frame will OOM with
        default float64 dtypes. We down-cast numeric columns to float32/int32
        at read time -- pandas' defaults blow up to ~1.7GB per block when
        consolidating, vs. ~850MB here.
        """
        raw_dir = self.config.paths.data_raw
        ds = self.config.dataset

        tx_path = Path(transaction_path) if transaction_path else raw_dir / ds.transaction_file
        id_path = Path(identity_path) if identity_path else raw_dir / ds.identity_file

        if not tx_path.exists():
            msg = f"Transaction file not found: {tx_path}. Run `make download-data` first."
            raise FileNotFoundError(msg)

        log.info("load_transaction_csv", path=str(tx_path))
        tx = self._read_csv_memory_efficient(tx_path, nrows=nrows)

        if id_path.exists():
            log.info("load_identity_csv", path=str(id_path))
            ident = self._read_csv_memory_efficient(id_path, nrows=nrows)
            # Kaggle's test_identity uses ``id-01..id-38`` (dash) while
            # train_identity uses ``id_01..id_38`` (underscore). Normalize
            # to underscores so fit/transform see the same column names.
            ident.columns = [
                c.replace("-", "_") if c.startswith("id-") and c[3:].isdigit() else c
                for c in ident.columns
            ]
            before = len(tx)
            df = tx.merge(ident, how="left", on=ds.join_key)
            log.info(
                "identity_merge_complete",
                tx_rows=before,
                identity_rows=len(ident),
                merged_rows=len(df),
            )
        else:
            log.warning("identity_file_missing", path=str(id_path))
            df = tx

        # Optional subset for local dev -- honor FRAUD_DATASET__USE_SUBSET.
        if ds.use_subset and len(df) > ds.subset_size:
            # Sort by time to keep the temporal structure intact.
            df = df.sort_values(ds.time_column).head(ds.subset_size).reset_index(drop=True)
            log.info("subset_applied", n=len(df))

        return df

    # ------------------------------------------------------------------
    # Memory-efficient CSV reader
    # ------------------------------------------------------------------

    @classmethod
    def _read_csv_memory_efficient(cls, path: Path, *, nrows: int | None = None) -> pd.DataFrame:
        """Read a CSV with narrower numeric dtypes to fit in 16GB.

        Strategy:
            * Peek at the header row to identify column families (V/D/C/id_*).
            * Supply an explicit ``dtype`` map: float32 for V/D/id_numeric,
              int16 for C/M-flag columns and addr1/addr2/card1-6.
            * Let pandas infer everything else (TransactionID, strings, etc.).
        """
        # Peek at the header to get column names without loading data.
        header = pd.read_csv(path, nrows=0).columns.tolist()

        dtype_map: dict[str, Any] = {}
        for col in header:
            if _V_RE.match(col) or _D_RE.match(col):
                dtype_map[col] = "float32"
            elif _C_RE.match(col):
                # C columns are count-valued but contain NaN -> use float32
                # (nullable Int32 would be better but is slower to read).
                dtype_map[col] = "float32"
            elif _ID_RE.match(col):
                # id_01-id_38 is a mix of numeric and string columns; pandas
                # will coerce string columns to object regardless of the hint.
                dtype_map[col] = "float32"
            elif col in {"TransactionAmt", "dist1", "dist2"} or col in {
                "addr1",
                "addr2",
                "card1",
                "card2",
                "card3",
                "card5",
            }:
                dtype_map[col] = "float32"
            elif col == "isFraud":
                dtype_map[col] = "int8"
            elif col == "TransactionDT":
                dtype_map[col] = "int32"
            # card4, card6, M*, email, device, product, TransactionID -> let
            # pandas pick (object or int64 as appropriate).

        # pandas will raise for any id_* columns that are actually strings; we
        # strip those from the map on first attempt via the exception path.
        try:
            return pd.read_csv(path, nrows=nrows, dtype=dtype_map, low_memory=False)
        except ValueError as exc:
            # Second attempt: drop the specific offending id_* key(s) from the
            # dtype map and retry. Rare in practice; IEEE-CIS has id_12-id_38
            # as strings but we handle via downstream logic so we just fall
            # back to object dtype for anything that wouldn't coerce.
            log.warning("load_csv_dtype_retry", exc=str(exc))
            for col in list(dtype_map):
                if _ID_RE.match(col):
                    dtype_map.pop(col, None)
            return pd.read_csv(path, nrows=nrows, dtype=dtype_map, low_memory=False)

    # ------------------------------------------------------------------
    # Stage 2: missing values
    # ------------------------------------------------------------------

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply per-group missing-value strategies.

        When ``self._fitted`` is False, this also *learns* which V-columns
        to drop (those exceeding ``drop_threshold``). On subsequent calls
        (``transform``), the cached ``dropped_v_columns`` are used instead.
        """
        df = df.copy()
        groups = group_columns(
            df,
            target=self.config.dataset.target,
            time_col=self.config.dataset.time_column,
        )
        strat = self.pp.missing_strategy

        # --- V features ----------------------------------------------------
        # Drop columns whose missing fraction exceeds the threshold (fit only).
        v_cols_present = [c for c in groups.v_features if c in df.columns]
        if not self._fitted:
            threshold = strat.v_features.drop_threshold or 1.0
            missing_frac = df[v_cols_present].isna().mean()
            to_drop = missing_frac[missing_frac > threshold].index.tolist()
            self.state.dropped_v_columns = list(to_drop)
            log.info(
                "v_features_drop",
                threshold=threshold,
                dropped=len(to_drop),
                kept=len(v_cols_present) - len(to_drop),
            )
        # Actually drop.
        drop_now = [c for c in self.state.dropped_v_columns if c in df.columns]
        if drop_now:
            df = df.drop(columns=drop_now)
        v_cols_remaining = [c for c in v_cols_present if c not in drop_now]

        df = self._apply_group_strategy(
            df,
            columns=v_cols_remaining,
            strategy=strat.v_features,
            prefix="v",
        )

        # --- D features ----------------------------------------------------
        df = self._apply_group_strategy(
            df, columns=groups.d_features, strategy=strat.d_features, prefix="d"
        )

        # --- C features ----------------------------------------------------
        df = self._apply_group_strategy(
            df, columns=groups.c_features, strategy=strat.c_features, prefix="c"
        )

        # --- M features (string-valued) ------------------------------------
        df = self._apply_group_strategy(
            df, columns=groups.m_features, strategy=strat.m_features, prefix="m"
        )

        # --- id_* numeric + categorical -----------------------------------
        df = self._apply_group_strategy(
            df, columns=groups.id_numeric, strategy=strat.id_numeric, prefix="id_num"
        )
        df = self._apply_group_strategy(
            df, columns=groups.id_categorical, strategy=strat.id_categorical, prefix="id_cat"
        )

        # --- Email domains ------------------------------------------------
        df = self._apply_group_strategy(
            df, columns=groups.email_cols, strategy=strat.email_domains, prefix="email"
        )

        # --- Catch-all: anything still NaN (dist1/dist2, addr1/addr2,
        #     card2/3/5 etc. that escape the V/D/C/M/id regex) -------------
        df = self._fill_remaining_nans(df)

        return df

    def _apply_group_strategy(
        self,
        df: pd.DataFrame,
        *,
        columns: list[str],
        strategy: MissingGroupStrategy,
        prefix: str,
    ) -> pd.DataFrame:
        """Apply ``strategy`` to ``columns``, optionally adding indicators.

        Indicator columns are built in a single ``pd.concat`` to avoid
        fragmenting the frame (which triggers pandas' PerformanceWarning on
        wide IEEE-CIS-sized frames).
        """
        if not columns:
            return df

        if strategy.add_indicator:
            indicators = {f"{col}__isna": df[col].isna().astype(np.int8) for col in columns}
            df = pd.concat([df, pd.DataFrame(indicators, index=df.index)], axis=1, copy=False)
            if not self._fitted:
                for name in indicators:
                    if name not in self.state.indicator_columns:
                        self.state.indicator_columns.append(name)

        fill_value = strategy.fill_value
        df[columns] = df[columns].fillna(fill_value)
        log.debug(
            "missing_filled",
            prefix=prefix,
            n_cols=len(columns),
            fill_value=fill_value,
            indicator=strategy.add_indicator,
        )
        return df

    def _fill_remaining_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sweep any columns that still have NaN after group strategies.

        At fit time: detect columns with current NaN, memoise them in
        ``state.leaked_{numeric,string}_columns``, fill + add indicators.

        At transform time: operate on the **memoised** column set, not what
        happens to be NaN in the current batch. Small serve batches often
        have no NaN in columns that leaked at fit, but the downstream scaler
        still expects their indicators -- so we always emit them (filled
        with zeros when the current batch has no NaN).
        """
        reserved = {
            self.config.dataset.target,
            self.config.dataset.time_column,
            self.config.dataset.join_key,
        }

        if not self._fitted:
            na_counts = df.isna().sum()
            leaked = [c for c, n in na_counts.items() if n > 0 and c not in reserved]
            numeric_leaked = [c for c in leaked if pd.api.types.is_numeric_dtype(df[c])]
            string_leaked = [c for c in leaked if c not in numeric_leaked]
            self.state.leaked_numeric_columns = list(numeric_leaked)
            self.state.leaked_string_columns = list(string_leaked)
        else:
            numeric_leaked = [c for c in self.state.leaked_numeric_columns if c in df.columns]
            string_leaked = [c for c in self.state.leaked_string_columns if c in df.columns]

        if not numeric_leaked and not string_leaked:
            return df

        if numeric_leaked:
            indicators = {f"{col}__isna": df[col].isna().astype(np.int8) for col in numeric_leaked}
            df = pd.concat([df, pd.DataFrame(indicators, index=df.index)], axis=1, copy=False)
            if not self._fitted:
                for name in indicators:
                    if name not in self.state.indicator_columns:
                        self.state.indicator_columns.append(name)
            df[numeric_leaked] = df[numeric_leaked].fillna(-999)

        if string_leaked:
            df[string_leaked] = df[string_leaked].fillna("unknown")

        log.info(
            "remaining_nans_filled",
            numeric=len(numeric_leaked),
            string=len(string_leaked),
            fitted=self._fitted,
        )
        return df

    # ------------------------------------------------------------------
    # Stage 3: categorical encoding
    # ------------------------------------------------------------------

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Integer-encode string columns with stable mappings.

        Unknown categories at transform time (unseen at fit) map to 0 and
        surface as a warning log.
        """
        df = df.copy()
        groups = group_columns(
            df,
            target=self.config.dataset.target,
            time_col=self.config.dataset.time_column,
        )
        # All categorical cols we know about.
        cat_cols: list[str] = [
            *groups.m_features,
            *groups.id_categorical,
            *groups.email_cols,
            *(c for c in DEVICE_CAT_COLS if c in df.columns),
            *groups.other_categorical,
        ]
        cat_cols = list(dict.fromkeys(cat_cols))  # dedupe, preserve order

        for col in cat_cols:
            if col not in df.columns:
                continue
            series = df[col].astype("string").fillna("unknown")
            if not self._fitted:
                categories = pd.unique(series)
                # Reserve 0 for "unknown" / unseen.
                mapping: dict[Any, int] = {"unknown": 0}
                for cat in categories:
                    if cat not in mapping:
                        mapping[cat] = len(mapping)
                self.state.categorical_mappings[col] = mapping
            else:
                mapping = self.state.categorical_mappings.get(col, {"unknown": 0})

            encoded = series.map(mapping)
            unseen_mask = encoded.isna()
            if unseen_mask.any():
                log.warning(
                    "categorical_unseen_values",
                    column=col,
                    n_unseen=int(unseen_mask.sum()),
                )
                encoded = encoded.fillna(0)

            df[col] = encoded.astype(np.int32)

        return df

    # ------------------------------------------------------------------
    # Stage 4: numeric normalization
    # ------------------------------------------------------------------

    def normalize_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log-transform amount (optional), clip outliers, then scale."""
        df = df.copy()

        # Log amount -- add as new column to preserve raw value for reporting.
        if self.pp.log_amount and AMOUNT_COL in df.columns:
            df[f"{AMOUNT_COL}__log1p"] = np.log1p(df[AMOUNT_COL].clip(lower=0))

        # Clip amount at upper quantile (fit-time only learns the threshold).
        if AMOUNT_COL in df.columns and 0.5 < self.pp.clip_quantile < 1.0:
            if not self._fitted:
                self.state.amount_clip_value = float(df[AMOUNT_COL].quantile(self.pp.clip_quantile))
            if self.state.amount_clip_value is not None:
                df[AMOUNT_COL] = df[AMOUNT_COL].clip(upper=self.state.amount_clip_value)

        # Determine numeric columns to scale -- exclude target, time, ID, and
        # integer-encoded categoricals (they're already dense integers).
        reserved = {
            self.config.dataset.target,
            self.config.dataset.time_column,
            self.config.dataset.join_key,
        }
        cat_cols = set(self.state.categorical_mappings.keys())
        if not self._fitted:
            numeric_cols = [
                c
                for c in df.columns
                if c not in reserved
                and c not in cat_cols
                and pd.api.types.is_numeric_dtype(df[c])
                and not c.endswith("__isna")  # leave binary indicators unscaled
            ]
            self.state.numeric_columns = numeric_cols

        numeric_cols = [c for c in self.state.numeric_columns if c in df.columns]

        if self.pp.normalize != "none" and numeric_cols:
            if not self._fitted:
                scaler = StandardScaler() if self.pp.normalize == "standard" else RobustScaler()
                scaler.fit(df[numeric_cols].to_numpy())
                self.state.scaler = scaler
            scaler = self.state.scaler
            if scaler is not None:
                df[numeric_cols] = scaler.transform(df[numeric_cols].to_numpy())

        return df

    # ------------------------------------------------------------------
    # fit / transform / fit_transform
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit + transform in one pass."""
        self._fitted = False
        out = self.handle_missing_values(df)
        out = self.encode_categoricals(out)
        out = self.normalize_numerics(out)
        self.state.n_features_out = out.shape[1]
        self._fitted = True
        log.info("preprocessor_fit_complete", **self.state.summary(), rows=len(out))
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply a previously fit state to new data."""
        if not self._fitted:
            msg = "Preprocessor has not been fit. Call fit_transform() first or load a state."
            raise RuntimeError(msg)
        out = self.handle_missing_values(df)
        out = self.encode_categoricals(out)
        out = self.normalize_numerics(out)
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the fitted state to ``path`` (pickle)."""
        if not self._fitted:
            msg = "Cannot save a preprocessor that has not been fit."
            raise RuntimeError(msg)
        path = Path(path)
        self.state.to_file(path)
        # Also dump a human-readable summary next to it for debugging.
        summary_path = path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(self.state.summary(), indent=2), encoding="utf-8")
        log.info("preprocessor_saved", path=str(path), summary_path=str(summary_path))

    @classmethod
    def load(cls, path: str | Path, config: AppConfig | None = None) -> IEEECISPreprocessor:
        """Load a previously-saved preprocessor."""
        pp = cls(config)
        pp.state = FittedState.from_file(path)
        pp._fitted = True
        log.info("preprocessor_loaded", path=str(path), **pp.state.summary())
        return pp


__all__ = [
    "ADDR_COLS",
    "AMOUNT_COL",
    "CARD_COLS",
    "DEVICE_CAT_COLS",
    "EMAIL_COLS",
    "PRODUCT_COL",
    "ColumnGroups",
    "FittedState",
    "IEEECISPreprocessor",
    "group_columns",
]
