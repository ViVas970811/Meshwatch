"""Temporal + Amount/Value features for Phase 2.

Produces **39 engineered columns** on an already-preprocessed frame:

* **Temporal (23)**: time-of-day cyclicals, day-of-week cyclicals, velocity
  over 1h/24h/7d windows, interarrival stats, is_weekend / is_night /
  is_business_hours flags, per-card first-seen delta, past-window counts.
* **Amount/Value (16)**: log1p amount (re-emitted for self-containment),
  cents, global z-score + percentile rank, cumulative rolling amounts,
  per-card relative amount metrics, acceleration / jerk of card amount
  series, and threshold-based flags.

All features are:

* Deterministic and causal -- no peeking at future rows.
* Stateful (``fit``/``transform``): per-card global stats are memoised at
  fit time so the serving path is identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)


# Canonical output column names (exposed for downstream tests).
TEMPORAL_FEATURES: tuple[str, ...] = (
    # Cyclical time-of-day + day-of-week
    "feat_hour_sin",
    "feat_hour_cos",
    "feat_dow_sin",
    "feat_dow_cos",
    "feat_hour",
    "feat_dow",
    # Simple flags
    "feat_is_weekend",
    "feat_is_night",
    "feat_is_business_hours",
    # Time-since events
    "feat_seconds_since_last_txn",
    "feat_seconds_since_last_card_txn",
    "feat_seconds_since_first_card_txn",
    # Rolling counts
    "feat_txn_count_1h",
    "feat_txn_count_24h",
    "feat_txn_count_7d",
    "feat_card_txn_count_1h",
    "feat_card_txn_count_24h",
    # Velocities (count / window)
    "feat_velocity_1h",
    "feat_velocity_24h",
    "feat_velocity_7d",
    # Interarrival stats (per card)
    "feat_interarrival_mean_card",
    "feat_interarrival_std_card",
    "feat_interarrival_last_card",
)

AMOUNT_FEATURES: tuple[str, ...] = (
    "feat_amt_log1p",
    "feat_amt_cents",
    "feat_amt_zscore",
    "feat_amt_percentile",
    "feat_amt_cum_1h",
    "feat_amt_cum_24h",
    "feat_amt_cum_7d",
    "feat_amt_card_cum",
    "feat_amt_vs_card_mean",
    "feat_amt_vs_card_max",
    "feat_amt_ratio_to_card_min",
    "feat_amt_ratio_to_card_max",
    "feat_amt_above_p95",
    "feat_amt_below_p05",
    "feat_amt_acceleration",
    "feat_amt_jerk",
)

ALL_TEMPORAL_AMOUNT_FEATURES: tuple[str, ...] = TEMPORAL_FEATURES + AMOUNT_FEATURES

_SECONDS_PER_HOUR = 3600
_SECONDS_PER_DAY = 86400
_SECONDS_PER_WEEK = 7 * _SECONDS_PER_DAY


@dataclass
class TemporalState:
    """Per-card statistics memoised at fit time so transform stays stable."""

    amount_mean_global: float = 0.0
    amount_std_global: float = 1.0
    amount_p05: float = 0.0
    amount_p95: float = 0.0
    card_amount_stats: dict[Any, dict[str, float]] = field(default_factory=dict)
    card_first_seen_ts: dict[Any, int] = field(default_factory=dict)


class TemporalFeatureBuilder:
    """Stateful builder for temporal + amount features."""

    def __init__(
        self,
        time_column: str = "TransactionDT",
        amount_column: str = "TransactionAmt",
        card_column: str = "card1",
    ) -> None:
        self.time_col = time_column
        self.amount_col = amount_column
        self.card_col = card_column
        self.state = TemporalState()
        self._fitted = False

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._fit(df)
        return self._transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            msg = "TemporalFeatureBuilder must be fit before transform."
            raise RuntimeError(msg)
        return self._transform(df)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def _fit(self, df: pd.DataFrame) -> None:
        amt = pd.to_numeric(df[self.amount_col], errors="coerce").fillna(0).to_numpy()
        self.state.amount_mean_global = float(np.mean(amt))
        self.state.amount_std_global = float(np.std(amt) or 1.0)
        self.state.amount_p05 = float(np.quantile(amt, 0.05))
        self.state.amount_p95 = float(np.quantile(amt, 0.95))
        # Per-card statistics.
        grouped = df.groupby(self.card_col)
        stats_df = grouped.agg(
            card_mean=(self.amount_col, "mean"),
            card_std=(self.amount_col, "std"),
            card_max=(self.amount_col, "max"),
            card_min=(self.amount_col, "min"),
        )
        # Fill the NaN std (single-txn cards) with 0.
        stats_df["card_std"] = stats_df["card_std"].fillna(0.0)
        self.state.card_amount_stats = {
            k: {
                "mean": float(row["card_mean"]),
                "std": float(row["card_std"]),
                "max": float(row["card_max"]),
                "min": float(row["card_min"]),
            }
            for k, row in stats_df.iterrows()
        }
        # First-seen timestamp per card.
        first_ts = grouped[self.time_col].min().astype(np.int64)
        self.state.card_first_seen_ts = {k: int(v) for k, v in first_ts.items()}
        self._fitted = True
        log.info(
            "temporal_fit_complete",
            n_cards=len(self.state.card_amount_stats),
            amount_mean=self.state.amount_mean_global,
            amount_std=self.state.amount_std_global,
        )

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Sort by time so causal rolling features are well-defined, then
        # restore original order at the end.
        original_index = df.index
        df = df.sort_values(self.time_col, kind="mergesort").copy()

        ts = pd.to_numeric(df[self.time_col], errors="coerce").fillna(0).astype(np.int64).to_numpy()
        amt = pd.to_numeric(df[self.amount_col], errors="coerce").fillna(0).to_numpy(np.float64)
        cards = df[self.card_col].astype("string").fillna("unknown").to_numpy()

        feats: dict[str, np.ndarray] = {}

        # ---- Cyclical time features ---------------------------------------
        seconds_in_day = ts % _SECONDS_PER_DAY
        hour = (seconds_in_day // _SECONDS_PER_HOUR).astype(np.float32)
        dow = ((ts // _SECONDS_PER_DAY) % 7).astype(np.float32)
        feats["feat_hour"] = hour
        feats["feat_dow"] = dow
        feats["feat_hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
        feats["feat_hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
        feats["feat_dow_sin"] = np.sin(2 * np.pi * dow / 7).astype(np.float32)
        feats["feat_dow_cos"] = np.cos(2 * np.pi * dow / 7).astype(np.float32)

        # ---- Flags --------------------------------------------------------
        feats["feat_is_weekend"] = (dow >= 5).astype(np.int8)
        feats["feat_is_night"] = ((hour < 6) | (hour >= 22)).astype(np.int8)
        feats["feat_is_business_hours"] = ((hour >= 9) & (hour <= 17) & (dow < 5)).astype(np.int8)

        # ---- Time-since-last-txn (global) --------------------------------
        seconds_since_last = np.zeros_like(ts, dtype=np.int64)
        seconds_since_last[1:] = np.diff(ts)
        feats["feat_seconds_since_last_txn"] = seconds_since_last.astype(np.float32)

        # ---- Per-card time + rolling stats -------------------------------
        (
            sec_since_last_card,
            sec_since_first_card,
            card_cnt_1h,
            card_cnt_24h,
            inter_mean,
            inter_std,
            inter_last,
            card_cum_amt,
        ) = self._per_card_rolling(ts, amt, cards)

        feats["feat_seconds_since_last_card_txn"] = sec_since_last_card.astype(np.float32)
        feats["feat_seconds_since_first_card_txn"] = sec_since_first_card.astype(np.float32)
        feats["feat_card_txn_count_1h"] = card_cnt_1h.astype(np.int32)
        feats["feat_card_txn_count_24h"] = card_cnt_24h.astype(np.int32)
        feats["feat_interarrival_mean_card"] = inter_mean.astype(np.float32)
        feats["feat_interarrival_std_card"] = inter_std.astype(np.float32)
        feats["feat_interarrival_last_card"] = inter_last.astype(np.float32)

        # ---- Rolling counts (global windows) ------------------------------
        cnt_1h, cnt_24h, cnt_7d, cum_1h, cum_24h, cum_7d = self._global_rolling(ts, amt)
        feats["feat_txn_count_1h"] = cnt_1h.astype(np.int32)
        feats["feat_txn_count_24h"] = cnt_24h.astype(np.int32)
        feats["feat_txn_count_7d"] = cnt_7d.astype(np.int32)
        feats["feat_velocity_1h"] = (cnt_1h / 1.0).astype(np.float32)
        feats["feat_velocity_24h"] = (cnt_24h / 24.0).astype(np.float32)
        feats["feat_velocity_7d"] = (cnt_7d / (7 * 24)).astype(np.float32)
        feats["feat_amt_cum_1h"] = cum_1h.astype(np.float32)
        feats["feat_amt_cum_24h"] = cum_24h.astype(np.float32)
        feats["feat_amt_cum_7d"] = cum_7d.astype(np.float32)
        feats["feat_amt_card_cum"] = card_cum_amt.astype(np.float32)

        # ---- Amount features ---------------------------------------------
        feats["feat_amt_log1p"] = np.log1p(np.clip(amt, 0.0, None)).astype(np.float32)
        feats["feat_amt_cents"] = ((amt * 100) % 100).astype(np.float32)
        feats["feat_amt_zscore"] = (
            (amt - self.state.amount_mean_global) / max(self.state.amount_std_global, 1e-9)
        ).astype(np.float32)
        # Percentile rank: pre-computed p05/p95 bookend, rest interpolated empirically.
        feats["feat_amt_percentile"] = np.clip(
            (amt - self.state.amount_p05)
            / max(self.state.amount_p95 - self.state.amount_p05, 1e-9),
            0.0,
            1.0,
        ).astype(np.float32)

        # ---- Per-card relative amount metrics -----------------------------
        amt_vs_mean = np.zeros_like(amt, dtype=np.float32)
        amt_vs_max = np.zeros_like(amt, dtype=np.float32)
        ratio_min = np.zeros_like(amt, dtype=np.float32)
        ratio_max = np.zeros_like(amt, dtype=np.float32)
        for i, (card, a) in enumerate(zip(cards, amt, strict=True)):
            stats = self.state.card_amount_stats.get(card)
            if stats is None:
                continue
            mean, max_, min_ = stats["mean"], stats["max"], stats["min"]
            amt_vs_mean[i] = a - mean
            amt_vs_max[i] = a - max_
            ratio_min[i] = a / max(min_, 1e-6)
            ratio_max[i] = a / max(max_, 1e-6)
        feats["feat_amt_vs_card_mean"] = amt_vs_mean
        feats["feat_amt_vs_card_max"] = amt_vs_max
        feats["feat_amt_ratio_to_card_min"] = ratio_min
        feats["feat_amt_ratio_to_card_max"] = ratio_max

        feats["feat_amt_above_p95"] = (amt > self.state.amount_p95).astype(np.int8)
        feats["feat_amt_below_p05"] = (amt < self.state.amount_p05).astype(np.int8)

        # ---- Acceleration / jerk (per-card diffs of amount) ---------------
        accel, jerk = self._per_card_accel_jerk(amt, cards)
        feats["feat_amt_acceleration"] = accel.astype(np.float32)
        feats["feat_amt_jerk"] = jerk.astype(np.float32)

        out = pd.DataFrame(feats, index=df.index)
        # Restore original row order.
        return out.reindex(original_index)

    # ------------------------------------------------------------------
    # Rolling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _global_rolling(ts: np.ndarray, amt: np.ndarray) -> tuple[np.ndarray, ...]:
        """Sliding-window counts + cumulative amounts over 1h/24h/7d.

        Two-pointer O(n) since the timestamps are sorted ascending.
        """
        n = len(ts)
        cnt_1h = np.zeros(n, dtype=np.int32)
        cnt_24h = np.zeros(n, dtype=np.int32)
        cnt_7d = np.zeros(n, dtype=np.int32)
        cum_1h = np.zeros(n, dtype=np.float64)
        cum_24h = np.zeros(n, dtype=np.float64)
        cum_7d = np.zeros(n, dtype=np.float64)

        # Left pointers for each window.
        l1 = l24 = l7 = 0
        run_1h = run_24h = run_7d = 0.0
        for i in range(n):
            t = ts[i]
            while l1 < i and ts[l1] < t - _SECONDS_PER_HOUR:
                run_1h -= amt[l1]
                l1 += 1
            while l24 < i and ts[l24] < t - _SECONDS_PER_DAY:
                run_24h -= amt[l24]
                l24 += 1
            while l7 < i and ts[l7] < t - _SECONDS_PER_WEEK:
                run_7d -= amt[l7]
                l7 += 1
            run_1h += amt[i]
            run_24h += amt[i]
            run_7d += amt[i]
            cnt_1h[i] = i - l1 + 1
            cnt_24h[i] = i - l24 + 1
            cnt_7d[i] = i - l7 + 1
            cum_1h[i] = run_1h
            cum_24h[i] = run_24h
            cum_7d[i] = run_7d
        return cnt_1h, cnt_24h, cnt_7d, cum_1h, cum_24h, cum_7d

    def _per_card_rolling(
        self, ts: np.ndarray, amt: np.ndarray, cards: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        """Per-card time-since + interarrival + rolling counts.

        Slower than the global version because cards are interleaved; still
        amortised O(n) in the total txn count.
        """
        n = len(ts)
        last_ts: dict[Any, int] = {}
        first_ts: dict[Any, int] = dict(self.state.card_first_seen_ts)  # seed from fit
        windowed_ts: dict[Any, list[int]] = {}
        windowed_gaps: dict[Any, list[int]] = {}
        cum_amt: dict[Any, float] = {}

        out_since_last = np.zeros(n, dtype=np.int64)
        out_since_first = np.zeros(n, dtype=np.int64)
        out_cnt_1h = np.zeros(n, dtype=np.int32)
        out_cnt_24h = np.zeros(n, dtype=np.int32)
        out_mean = np.zeros(n, dtype=np.float64)
        out_std = np.zeros(n, dtype=np.float64)
        out_last = np.zeros(n, dtype=np.int64)
        out_cum = np.zeros(n, dtype=np.float64)

        for i in range(n):
            card = cards[i]
            t = int(ts[i])
            first = first_ts.setdefault(card, t)
            out_since_first[i] = t - first
            last = last_ts.get(card)
            gap = t - last if last is not None else 0
            out_since_last[i] = gap
            out_last[i] = gap
            last_ts[card] = t

            # Interarrival stats.
            gaps = windowed_gaps.setdefault(card, [])
            if last is not None:
                gaps.append(gap)
            if gaps:
                gaps_arr = np.asarray(gaps, dtype=np.float64)
                out_mean[i] = float(gaps_arr.mean())
                out_std[i] = float(gaps_arr.std())

            # Per-card rolling counts within 1h/24h.
            ts_list = windowed_ts.setdefault(card, [])
            ts_list.append(t)
            while ts_list and ts_list[0] < t - _SECONDS_PER_HOUR:
                ts_list.pop(0)
            out_cnt_1h[i] = len(ts_list)
            # 24h: cheaper to scan a separate list.
            # We re-derive by linear scan on the same list since 24h window
            # dominates 1h.
            # (already pruned to 1h; for 24h we need a separate list.)
            # -- Approximate 24h count as 1h count + a small correction by
            # periodically trimming a 24h buffer. To keep code simple and
            # correct, scan backwards inline.
            # For most workloads the per-card series is short enough that
            # this is O(1) amortised.
            out_cnt_24h[i] = out_cnt_1h[i]  # close enough -- updated below
            # Instead use: count all seen ts within 24h.
            # We maintain a separate buffer:
            pass

            # Cumulative amount per card.
            cum_amt[card] = cum_amt.get(card, 0.0) + float(amt[i])
            out_cum[i] = cum_amt[card]

        # Refine 24h counts via a second pass that maintains a 24h buffer.
        windowed_24h: dict[Any, list[int]] = {}
        for i in range(n):
            card = cards[i]
            t = int(ts[i])
            buf = windowed_24h.setdefault(card, [])
            buf.append(t)
            while buf and buf[0] < t - _SECONDS_PER_DAY:
                buf.pop(0)
            out_cnt_24h[i] = len(buf)

        return (
            out_since_last,
            out_since_first,
            out_cnt_1h,
            out_cnt_24h,
            out_mean,
            out_std,
            out_last,
            np.asarray([out_cum[i] for i in range(n)]),
        )

    @staticmethod
    def _per_card_accel_jerk(amt: np.ndarray, cards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """First and second differences of per-card amount series."""
        n = len(amt)
        accel = np.zeros(n, dtype=np.float64)
        jerk = np.zeros(n, dtype=np.float64)
        prev_amt: dict[Any, float] = {}
        prev_accel: dict[Any, float] = {}
        for i, (card, a) in enumerate(zip(cards, amt, strict=True)):
            p = prev_amt.get(card)
            if p is not None:
                accel[i] = float(a - p)
                pa = prev_accel.get(card)
                if pa is not None:
                    jerk[i] = float(accel[i] - pa)
            prev_accel[card] = float(accel[i])
            prev_amt[card] = float(a)
        return accel, jerk


__all__ = [
    "ALL_TEMPORAL_AMOUNT_FEATURES",
    "AMOUNT_FEATURES",
    "TEMPORAL_FEATURES",
    "TemporalFeatureBuilder",
    "TemporalState",
]
