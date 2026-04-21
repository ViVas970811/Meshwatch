"""Entity-aggregated + Identity/Device-risk features for Phase 2.

Produces **52 engineered columns** on an already-preprocessed frame:

* **Aggregated (36)**: per-card / per-email / per-address / per-device
  statistics (mean / std / max / count / target-encoded fraud rate), entity
  diversity counts (how many unique addresses / emails / devices a card
  touches), and C/M-family aggregated signals.
* **Identity / Device Risk (16)**: bucketed id_01 / id_02 risk scores,
  device-os/browser indicators, email mismatch / multi-card /
  suspicious-hour / high-risk TLD flags, V-feature outlier z-score.

All statistics are learned during ``fit_transform`` on the **training
split** (to avoid target leakage) and replayed at transform time via a
simple dict lookup. Unknown entities fall back to the global mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)


# Canonical output column lists.
AGGREGATED_CARD_FEATURES: tuple[str, ...] = (
    "feat_card_mean_amt",
    "feat_card_median_amt",
    "feat_card_std_amt",
    "feat_card_max_amt",
    "feat_card_count",
    "feat_card_fraud_rate",
    "feat_card_unique_addr",
    "feat_card_unique_emails",
    "feat_card_unique_devices",
    "feat_card_unique_products",
    "feat_card_days_active",
    "feat_card_txn_freq",
)

AGGREGATED_EMAIL_FEATURES: tuple[str, ...] = (
    "feat_email_mean_amt",
    "feat_email_median_amt",
    "feat_email_std_amt",
    "feat_email_count",
    "feat_email_fraud_rate",
    "feat_email_unique_cards",
    "feat_email_unique_addrs",
)

AGGREGATED_ADDR_FEATURES: tuple[str, ...] = (
    "feat_addr_mean_amt",
    "feat_addr_median_amt",
    "feat_addr_std_amt",
    "feat_addr_count",
    "feat_addr_fraud_rate",
    "feat_addr_n_cards",
)

AGGREGATED_DEVICE_FEATURES: tuple[str, ...] = (
    "feat_device_mean_amt",
    "feat_device_median_amt",
    "feat_device_std_amt",
    "feat_device_count",
    "feat_device_fraud_rate",
    "feat_device_n_cards",
)

C_FAMILY_FEATURES: tuple[str, ...] = (
    "feat_c1_mean_card",
    "feat_c13_mean_card",
    "feat_c14_mean_card",
    "feat_m_flag_mean_card",
    "feat_c_total_per_txn",
)

AGGREGATED_FEATURES: tuple[str, ...] = (
    AGGREGATED_CARD_FEATURES
    + AGGREGATED_EMAIL_FEATURES
    + AGGREGATED_ADDR_FEATURES
    + AGGREGATED_DEVICE_FEATURES
    + C_FAMILY_FEATURES
)

IDENTITY_FEATURES: tuple[str, ...] = (
    "feat_id_01_bin_risk",
    "feat_id_02_bin_risk",
    "feat_device_is_windows",
    "feat_device_is_ios",
    "feat_device_is_android",
    "feat_device_is_mobile",
    "feat_is_proxy_heuristic",
    "feat_email_mismatch",
    "feat_high_risk_tld",
    "feat_suspicious_hour",
    "feat_identity_missing",
    "feat_new_device_for_card",
    "feat_multi_card_on_device",
    "feat_v_feature_zscore_max",
    "feat_v_feature_zscore_mean",
    "feat_amt_outlier_flag",
)

ALL_AGGREGATED_FEATURES: tuple[str, ...] = AGGREGATED_FEATURES + IDENTITY_FEATURES


_HIGH_RISK_TLDS: frozenset[str] = frozenset({"ru", "cn", "xyz", "top", "click", "pw", "info"})


@dataclass
class AggregatedState:
    """Entity stats + risk buckets learned at fit time."""

    # entity -> {stat: float}
    card_stats: dict[Any, dict[str, float]] = field(default_factory=dict)
    email_stats: dict[Any, dict[str, float]] = field(default_factory=dict)
    addr_stats: dict[tuple, dict[str, float]] = field(default_factory=dict)
    device_stats: dict[Any, dict[str, float]] = field(default_factory=dict)
    # Global fallbacks (mean across all entities).
    global_fraud_rate: float = 0.0
    global_mean_amt: float = 0.0
    global_std_amt: float = 1.0
    # Risk bins for id_01 / id_02.
    id01_edges: np.ndarray | None = None
    id02_edges: np.ndarray | None = None
    id01_bin_risk: np.ndarray | None = None  # fraud rate per bin
    id02_bin_risk: np.ndarray | None = None
    # V-feature global stats for z-score outlier detection.
    v_mean: np.ndarray | None = None
    v_std: np.ndarray | None = None
    v_cols: list[str] = field(default_factory=list)
    # Amount outlier threshold (2*std above mean is "outlier").
    amt_outlier_threshold: float = 0.0
    # Devices seen per card at fit time (for new_device flag).
    card_known_devices: dict[Any, set[Any]] = field(default_factory=dict)
    # Card sets per device (for multi_card flag).
    device_card_sets: dict[Any, set[Any]] = field(default_factory=dict)


class AggregatedFeatureBuilder:
    """Stateful builder for aggregated + identity features."""

    def __init__(
        self,
        target_column: str = "isFraud",
        time_column: str = "TransactionDT",
        amount_column: str = "TransactionAmt",
    ) -> None:
        self.target_col = target_column
        self.time_col = time_column
        self.amount_col = amount_column
        self.state = AggregatedState()
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._fit(df)
        return self._transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            msg = "AggregatedFeatureBuilder must be fit before transform."
            raise RuntimeError(msg)
        return self._transform(df)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def _fit(self, df: pd.DataFrame) -> None:
        target = (
            df[self.target_col]
            if self.target_col in df.columns
            else pd.Series(np.zeros(len(df), dtype=np.int8), index=df.index)
        )
        amt = pd.to_numeric(df[self.amount_col], errors="coerce").fillna(0).to_numpy()

        self.state.global_fraud_rate = float(target.mean()) if len(target) else 0.0
        self.state.global_mean_amt = float(np.mean(amt)) if len(amt) else 0.0
        self.state.global_std_amt = float(np.std(amt) or 1.0)
        self.state.amt_outlier_threshold = (
            self.state.global_mean_amt + 2.0 * self.state.global_std_amt
        )

        self.state.card_stats = self._entity_stats(df, key="card1", target=target)
        self.state.email_stats = self._entity_stats(df, key="P_emaildomain", target=target)
        # Address uses (addr1, addr2) tuple.
        addr_key = self._address_key(df)
        self.state.addr_stats = self._entity_stats(
            df.assign(_addr=addr_key), key="_addr", target=target
        )
        dev_key = self._device_key(df)
        self.state.device_stats = self._entity_stats(
            df.assign(_device=dev_key), key="_device", target=target
        )

        # Extra diversity stats for cards / emails.
        grouped_card = df.groupby("card1", sort=False)
        for card, sub in grouped_card:
            stats = self.state.card_stats.setdefault(card, {})
            stats["unique_addr"] = float(sub[["addr1", "addr2"]].drop_duplicates().shape[0])
            stats["unique_emails"] = float(sub["P_emaildomain"].nunique(dropna=False))
            stats["unique_devices"] = float(self._device_key(sub).nunique(dropna=False))
            if "ProductCD" in sub.columns:
                stats["unique_products"] = float(sub["ProductCD"].nunique(dropna=False))
            else:
                stats["unique_products"] = 0.0
            ts = pd.to_numeric(sub[self.time_col], errors="coerce").dropna()
            stats["days_active"] = float(((ts.max() - ts.min()) / 86400) if len(ts) > 1 else 0.0)
            stats["txn_freq"] = float(len(sub) / max(stats["days_active"], 1.0))

        if "P_emaildomain" in df.columns:
            for dom, sub in df.groupby("P_emaildomain", sort=False):
                stats = self.state.email_stats.setdefault(dom, {})
                stats["unique_cards"] = float(sub["card1"].nunique(dropna=False))
                stats["unique_addrs"] = float(sub[["addr1", "addr2"]].drop_duplicates().shape[0])

        for key, sub in df.assign(_addr=addr_key).groupby("_addr", sort=False):
            stats = self.state.addr_stats.setdefault(key, {})
            stats["n_cards"] = float(sub["card1"].nunique(dropna=False))
        for key, sub in df.assign(_device=dev_key).groupby("_device", sort=False):
            stats = self.state.device_stats.setdefault(key, {})
            stats["n_cards"] = float(sub["card1"].nunique(dropna=False))

        # Risk bins for id_01 / id_02.
        self.state.id01_edges, self.state.id01_bin_risk = self._bin_risk(df, "id_01", target)
        self.state.id02_edges, self.state.id02_bin_risk = self._bin_risk(df, "id_02", target)

        # V-feature z-score outliers.
        v_cols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
        if v_cols:
            V = df[v_cols].fillna(0).to_numpy(dtype=np.float32)
            self.state.v_mean = V.mean(axis=0)
            self.state.v_std = np.where(V.std(axis=0) == 0, 1.0, V.std(axis=0)).astype(np.float32)
            self.state.v_cols = v_cols

        # Card -> known devices, Device -> known cards (for new_device / multi_card).
        for card, sub in grouped_card:
            self.state.card_known_devices[card] = set(self._device_key(sub).unique())
        for dev, sub in df.assign(_device=dev_key).groupby("_device", sort=False):
            self.state.device_card_sets[dev] = set(sub["card1"].unique())

        self._fitted = True
        log.info(
            "aggregated_fit_complete",
            n_cards=len(self.state.card_stats),
            n_emails=len(self.state.email_stats),
            n_addrs=len(self.state.addr_stats),
            n_devices=len(self.state.device_stats),
        )

    def _entity_stats(
        self, df: pd.DataFrame, *, key: str, target: pd.Series
    ) -> dict[Any, dict[str, float]]:
        if key not in df.columns:
            return {}
        grouped = df.groupby(key, sort=False)
        amt = pd.to_numeric(df[self.amount_col], errors="coerce").fillna(0)
        out: dict[Any, dict[str, float]] = {}
        for val, sub in grouped:
            sub_amt = amt.loc[sub.index]
            sub_target = target.loc[sub.index]
            std = sub_amt.std()
            out[val] = {
                "mean_amt": float(sub_amt.mean()) if len(sub_amt) else 0.0,
                "median_amt": float(sub_amt.median()) if len(sub_amt) else 0.0,
                # std() returns NaN for single-observation groups (ddof=1).
                # Explicit NaN check -- ``nan or 0.0`` yields ``nan`` because
                # bool(nan) is True.
                "std_amt": float(std) if pd.notna(std) else 0.0,
                "max_amt": float(sub_amt.max()) if len(sub_amt) else 0.0,
                "count": float(len(sub)),
                "fraud_rate": float(sub_target.mean()) if len(sub_target) else 0.0,
            }
        return out

    @staticmethod
    def _address_key(df: pd.DataFrame) -> list[tuple]:
        return list(
            zip(
                df.get("addr1", pd.Series([-1] * len(df))).fillna(-1).to_numpy(),
                df.get("addr2", pd.Series([-1] * len(df))).fillna(-1).to_numpy(),
                strict=True,
            )
        )

    @staticmethod
    def _device_key(df: pd.DataFrame) -> pd.Series:
        info = df.get("DeviceInfo")
        type_ = df.get("DeviceType")
        info = (
            info.astype("string").fillna("")
            if info is not None
            else pd.Series([""] * len(df), index=df.index)
        )
        type_ = (
            type_.astype("string").fillna("")
            if type_ is not None
            else pd.Series([""] * len(df), index=df.index)
        )
        return (type_.astype(str) + "|" + info.astype(str)).replace({"|": "unknown"})

    @staticmethod
    def _bin_risk(
        df: pd.DataFrame, col: str, target: pd.Series, n_bins: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        if col not in df.columns:
            return np.array([0.0, 1.0]), np.zeros(1, dtype=np.float32)
        vals = pd.to_numeric(df[col], errors="coerce")
        clean_mask = vals.notna().to_numpy()
        if clean_mask.sum() < 2:
            return np.array([0.0, 1.0]), np.zeros(1, dtype=np.float32)
        clean_vals = vals.to_numpy()[clean_mask]
        edges = np.unique(np.quantile(clean_vals, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 2:
            return np.array([0.0, 1.0]), np.zeros(1, dtype=np.float32)
        binned = np.clip(np.searchsorted(edges, clean_vals, side="right") - 1, 0, len(edges) - 2)
        clean_target = target.to_numpy()[clean_mask]
        bin_risk = np.zeros(len(edges) - 1, dtype=np.float32)
        for b in range(len(edges) - 1):
            mask = binned == b
            bin_risk[b] = float(clean_target[mask].mean()) if mask.any() else 0.0
        return edges, bin_risk

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        feats: dict[str, np.ndarray] = {}

        cards = df["card1"].astype("string").fillna("unknown")
        emails = (
            df["P_emaildomain"].astype("string").fillna("unknown")
            if "P_emaildomain" in df.columns
            else pd.Series(["unknown"] * n, index=df.index)
        )
        addr_keys = pd.Series(self._address_key(df), index=df.index)
        dev_keys = self._device_key(df)

        # Vectorised lookup: pandas.map is ~100-1000x faster than a
        # per-row Python loop on 590K rows.
        def _lookup(
            entity_series: pd.Series,
            stats: dict,
            metric: str,
            default: float,
        ) -> np.ndarray:
            mapping = {k: v.get(metric, default) for k, v in stats.items()}
            return entity_series.map(mapping).fillna(default).astype(np.float32).to_numpy()

        for metric, col, default in (
            ("mean_amt", "feat_card_mean_amt", self.state.global_mean_amt),
            ("median_amt", "feat_card_median_amt", self.state.global_mean_amt),
            ("std_amt", "feat_card_std_amt", 0.0),
            ("max_amt", "feat_card_max_amt", 0.0),
            ("count", "feat_card_count", 0.0),
            ("fraud_rate", "feat_card_fraud_rate", self.state.global_fraud_rate),
            ("unique_addr", "feat_card_unique_addr", 0.0),
            ("unique_emails", "feat_card_unique_emails", 0.0),
            ("unique_devices", "feat_card_unique_devices", 0.0),
            ("unique_products", "feat_card_unique_products", 0.0),
            ("days_active", "feat_card_days_active", 0.0),
            ("txn_freq", "feat_card_txn_freq", 0.0),
        ):
            feats[col] = _lookup(cards, self.state.card_stats, metric, default)

        for metric, col, default in (
            ("mean_amt", "feat_email_mean_amt", self.state.global_mean_amt),
            ("median_amt", "feat_email_median_amt", self.state.global_mean_amt),
            ("std_amt", "feat_email_std_amt", 0.0),
            ("count", "feat_email_count", 0.0),
            ("fraud_rate", "feat_email_fraud_rate", self.state.global_fraud_rate),
            ("unique_cards", "feat_email_unique_cards", 0.0),
            ("unique_addrs", "feat_email_unique_addrs", 0.0),
        ):
            feats[col] = _lookup(emails, self.state.email_stats, metric, default)

        for metric, col, default in (
            ("mean_amt", "feat_addr_mean_amt", self.state.global_mean_amt),
            ("median_amt", "feat_addr_median_amt", self.state.global_mean_amt),
            ("std_amt", "feat_addr_std_amt", 0.0),
            ("count", "feat_addr_count", 0.0),
            ("fraud_rate", "feat_addr_fraud_rate", self.state.global_fraud_rate),
            ("n_cards", "feat_addr_n_cards", 0.0),
        ):
            feats[col] = _lookup(addr_keys, self.state.addr_stats, metric, default)

        for metric, col, default in (
            ("mean_amt", "feat_device_mean_amt", self.state.global_mean_amt),
            ("median_amt", "feat_device_median_amt", self.state.global_mean_amt),
            ("std_amt", "feat_device_std_amt", 0.0),
            ("count", "feat_device_count", 0.0),
            ("fraud_rate", "feat_device_fraud_rate", self.state.global_fraud_rate),
            ("n_cards", "feat_device_n_cards", 0.0),
        ):
            feats[col] = _lookup(dev_keys, self.state.device_stats, metric, default)

        # ---- C / M family signals ---------------------------------------
        for key in ("C1", "C13", "C14"):
            col = f"feat_{key.lower()}_mean_card"
            if key in df.columns:
                # group-by-card running mean.
                means = df.groupby("card1")[key].transform("mean").to_numpy(np.float32)
                feats[col] = np.nan_to_num(means, nan=0.0).astype(np.float32)
            else:
                feats[col] = np.zeros(n, dtype=np.float32)

        m_cols = [c for c in df.columns if c.startswith("M") and c[1:].isdigit()]
        if m_cols:
            m_mean = df.groupby("card1")[m_cols].transform("mean").mean(axis=1).to_numpy(np.float32)
            feats["feat_m_flag_mean_card"] = np.nan_to_num(m_mean, nan=0.0).astype(np.float32)
        else:
            feats["feat_m_flag_mean_card"] = np.zeros(n, dtype=np.float32)

        c_cols = [c for c in df.columns if c.startswith("C") and c[1:].isdigit()]
        if c_cols:
            feats["feat_c_total_per_txn"] = df[c_cols].sum(axis=1).fillna(0).to_numpy(np.float32)
        else:
            feats["feat_c_total_per_txn"] = np.zeros(n, dtype=np.float32)

        # ---- Identity / device risk features ----------------------------
        feats["feat_id_01_bin_risk"] = self._apply_bin_risk(
            df.get("id_01"), self.state.id01_edges, self.state.id01_bin_risk
        )
        feats["feat_id_02_bin_risk"] = self._apply_bin_risk(
            df.get("id_02"), self.state.id02_edges, self.state.id02_bin_risk
        )

        info_str = (
            df.get("DeviceInfo").astype("string").fillna("").str.lower()
            if "DeviceInfo" in df.columns
            else pd.Series([""] * n, index=df.index)
        )
        type_str = (
            df.get("DeviceType").astype("string").fillna("").str.lower()
            if "DeviceType" in df.columns
            else pd.Series([""] * n, index=df.index)
        )
        feats["feat_device_is_windows"] = info_str.str.contains("windows", na=False).astype(np.int8)
        feats["feat_device_is_ios"] = info_str.str.contains("iphone|ipad|ios", na=False).astype(
            np.int8
        )
        feats["feat_device_is_android"] = info_str.str.contains(
            "android|sm-|moto", na=False
        ).astype(np.int8)
        feats["feat_device_is_mobile"] = (type_str == "mobile").astype(np.int8).to_numpy()

        # Simple proxy heuristic: id_35 / id_36 flags + large id_02 mismatch.
        proxy = np.zeros(n, dtype=np.int8)
        for col in ("id_35", "id_36", "id_37", "id_38"):
            if col in df.columns:
                proxy |= pd.to_numeric(df[col], errors="coerce").fillna(0).gt(0).to_numpy(np.int8)
        feats["feat_is_proxy_heuristic"] = proxy

        # Email mismatch: P vs R email domains differ (but both present).
        if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
            p = df["P_emaildomain"].astype("string").fillna("unknown")
            r = df["R_emaildomain"].astype("string").fillna("unknown")
            mm = ((p != r) & (p != "unknown") & (r != "unknown")).astype(np.int8)
            feats["feat_email_mismatch"] = mm.to_numpy()
        else:
            feats["feat_email_mismatch"] = np.zeros(n, dtype=np.int8)

        # High-risk TLD.
        if "P_emaildomain" in df.columns:
            tld = (
                df["P_emaildomain"]
                .astype("string")
                .fillna("unknown")
                .str.rsplit(".", n=1)
                .str[-1]
                .str.lower()
            )
            feats["feat_high_risk_tld"] = tld.isin(_HIGH_RISK_TLDS).astype(np.int8).to_numpy()
        else:
            feats["feat_high_risk_tld"] = np.zeros(n, dtype=np.int8)

        # Suspicious hour: 2am-5am local.
        if self.time_col in df.columns:
            hour = (
                pd.to_numeric(df[self.time_col], errors="coerce").fillna(0).astype(np.int64) % 86400
            ) // 3600
            feats["feat_suspicious_hour"] = ((hour >= 2) & (hour <= 5)).astype(np.int8).to_numpy()
        else:
            feats["feat_suspicious_hour"] = np.zeros(n, dtype=np.int8)

        # Identity missing: at least half the id_* cols NaN in the *raw*
        # sense (approximate by checking our __isna indicators if present).
        isna_cols = [c for c in df.columns if c.startswith("id_") and c.endswith("__isna")]
        if isna_cols:
            feats["feat_identity_missing"] = (
                df[isna_cols].sum(axis=1).gt(len(isna_cols) / 2).astype(np.int8).to_numpy()
            )
        else:
            feats["feat_identity_missing"] = np.zeros(n, dtype=np.int8)

        # New device / multi-card on device -- vectorised via pandas map.
        # Pre-compute a "(card, device) seen at fit" lookup as a frozenset
        # per card, then test membership for the whole column at once.
        cards_np = cards.to_numpy()
        devs_np = dev_keys.to_numpy()
        # new_device: 1 if the card was seen at fit but the current device
        # was not among its fit-time devices.
        new_dev = np.zeros(n, dtype=np.int8)
        for i, (c, d) in enumerate(zip(cards_np, devs_np, strict=True)):
            known = self.state.card_known_devices.get(c)
            if known is not None and d not in known:
                new_dev[i] = 1
        feats["feat_new_device_for_card"] = new_dev
        # multi_card: 1 if the device was linked to >1 card at fit time.
        # Map once; no per-row branching.
        device_is_multi = {
            dev: (1 if len(cset) > 1 else 0) for dev, cset in self.state.device_card_sets.items()
        }
        feats["feat_multi_card_on_device"] = (
            dev_keys.map(device_is_multi).fillna(0).astype(np.int8).to_numpy()
        )

        # V-feature z-score outliers.
        if self.state.v_cols and self.state.v_mean is not None and self.state.v_std is not None:
            V = df[self.state.v_cols].fillna(0).to_numpy(dtype=np.float32)
            z = np.abs((V - self.state.v_mean) / self.state.v_std)
            feats["feat_v_feature_zscore_max"] = z.max(axis=1).astype(np.float32)
            feats["feat_v_feature_zscore_mean"] = z.mean(axis=1).astype(np.float32)
        else:
            feats["feat_v_feature_zscore_max"] = np.zeros(n, dtype=np.float32)
            feats["feat_v_feature_zscore_mean"] = np.zeros(n, dtype=np.float32)

        amt = pd.to_numeric(df[self.amount_col], errors="coerce").fillna(0).to_numpy()
        feats["feat_amt_outlier_flag"] = (amt > self.state.amt_outlier_threshold).astype(np.int8)

        return pd.DataFrame(feats, index=df.index, columns=list(ALL_AGGREGATED_FEATURES))

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _apply_bin_risk(
        self, series: pd.Series | None, edges: np.ndarray | None, bin_risk: np.ndarray | None
    ) -> np.ndarray:
        if series is None or edges is None or bin_risk is None or len(edges) < 2:
            return np.full(
                len(series) if series is not None else 0,
                self.state.global_fraud_rate,
                dtype=np.float32,
            )
        vals = pd.to_numeric(series, errors="coerce")
        out = np.full(len(vals), self.state.global_fraud_rate, dtype=np.float32)
        mask = vals.notna().to_numpy()
        if mask.any():
            b = np.clip(
                np.searchsorted(edges, vals.to_numpy()[mask], side="right") - 1,
                0,
                len(bin_risk) - 1,
            )
            out[mask] = bin_risk[b]
        return out


__all__ = [
    "AGGREGATED_ADDR_FEATURES",
    "AGGREGATED_CARD_FEATURES",
    "AGGREGATED_DEVICE_FEATURES",
    "AGGREGATED_EMAIL_FEATURES",
    "AGGREGATED_FEATURES",
    "ALL_AGGREGATED_FEATURES",
    "C_FAMILY_FEATURES",
    "IDENTITY_FEATURES",
    "AggregatedFeatureBuilder",
    "AggregatedState",
]
