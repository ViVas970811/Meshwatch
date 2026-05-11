"""Production performance tracker (Phase 7).

Production monitoring for a fraud model is fundamentally a labelled-arrival
problem: predictions come out in real time, but ground-truth labels show up
later -- chargebacks land 30-90 days after the transaction. We track the
rolling window of ``(prediction, label, timestamp)`` triples and recompute
classification metrics on demand.

The tracker is designed to be safe to import on a CPU-only laptop:

* :mod:`sklearn.metrics` is used when available for AUROC / AUPRC -- otherwise
  we fall back to the trapezoidal-rule implementation defined here so the
  public API stays the same.
* :mod:`mlflow` is optional -- if installed, :meth:`PerformanceTracker.log_mlflow`
  pushes the latest snapshot under a ``production`` run.

The tracker is bounded (a configurable ``max_records``) so memory stays
finite even under sustained traffic.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any

import numpy as np

try:
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    _HAVE_SKLEARN = True
except ImportError:  # pragma: no cover -- sklearn is in the base deps
    _HAVE_SKLEARN = False


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PerformanceSnapshot:
    """Aggregate classification metrics over a window of labelled predictions."""

    n_total: int = 0
    n_labelled: int = 0
    n_positive: int = 0
    n_predicted_positive: int = 0
    threshold: float = 0.5

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    auroc: float | None = None
    auprc: float | None = None
    brier: float | None = None

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    window_start: datetime | None = None
    window_end: datetime | None = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("window_start", "window_end", "generated_at"):
            v = d.get(k)
            if isinstance(v, datetime):
                d[k] = v.isoformat()
        return d


# ---------------------------------------------------------------------------
# AUC helpers (with pure-numpy fallback)
# ---------------------------------------------------------------------------


def _auroc_numpy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Trapezoidal-rule AUROC -- works without sklearn."""
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    order = np.argsort(-y_score)
    y = y_true[order]
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    p = (y == 1).sum()
    n = (y == 0).sum()
    if p == 0 or n == 0:
        return float("nan")
    tpr = tps / p
    fpr = fps / n
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))


def _auprc_numpy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average-precision via running precision-recall (sklearn-compatible shape)."""
    if y_true.size == 0 or (y_true == 1).sum() == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y = y_true[order]
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    precision = tps / (tps + fps + 1e-12)
    recall = tps / max((y == 1).sum(), 1)
    # AP = sum_i (recall_i - recall_{i-1}) * precision_i  (sklearn definition)
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def auroc(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    yt = np.asarray(list(y_true), dtype=np.int64)
    ys = np.asarray(list(y_score), dtype=np.float64)
    if _HAVE_SKLEARN and yt.size and len(np.unique(yt)) == 2:
        try:
            return float(roc_auc_score(yt, ys))
        except Exception:  # pragma: no cover -- degraded labels
            return float("nan")
    return _auroc_numpy(yt, ys)


def auprc(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    yt = np.asarray(list(y_true), dtype=np.int64)
    ys = np.asarray(list(y_score), dtype=np.float64)
    if _HAVE_SKLEARN and yt.size and (yt == 1).any():
        try:
            return float(average_precision_score(yt, ys))
        except Exception:  # pragma: no cover
            return float("nan")
    return _auprc_numpy(yt, ys)


# ---------------------------------------------------------------------------
# PerformanceTracker
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Record:
    transaction_id: int | str
    score: float
    timestamp: datetime
    label: int | None = None


class PerformanceTracker:
    """Bounded ring-buffer of (score, label, timestamp) records.

    Designed to be safe for concurrent use from the FastAPI app's request
    handlers (predictions append) and a background job (labels merge in
    from chargeback events). Both paths take the same lock; metric
    computation is read-only and copies the buffer first.

    Parameters
    ----------
    max_records
        Hard cap on the ring buffer. Older records are evicted when full.
    threshold
        Decision threshold for precision/recall/F1. AUC metrics ignore it.
    """

    def __init__(self, *, max_records: int = 5_000, threshold: float = 0.7) -> None:
        self._records: deque[_Record] = deque(maxlen=max_records)
        self._lock = RLock()
        self.threshold = float(threshold)

    # ------------------------------------------------------------------
    # writers
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        transaction_id: int | str,
        score: float,
        *,
        timestamp: datetime | None = None,
        label: int | None = None,
    ) -> None:
        """Append a prediction. ``label`` may be supplied later via :meth:`record_label`."""
        with self._lock:
            self._records.append(
                _Record(
                    transaction_id=transaction_id,
                    score=float(score),
                    timestamp=timestamp or datetime.now(timezone.utc),
                    label=None if label is None else int(label),
                )
            )

    def record_label(self, transaction_id: int | str, label: int) -> bool:
        """Attach a ground-truth label to an existing record.

        Returns ``True`` if the record was found, ``False`` if it has
        already evicted from the bounded buffer (a common, acceptable
        outcome when labels arrive months after the prediction).
        """
        with self._lock:
            for r in reversed(self._records):
                if r.transaction_id == transaction_id:
                    r.label = int(label)
                    return True
        return False

    def reset(self) -> None:
        with self._lock:
            self._records.clear()

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    @property
    def n_records(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def n_labelled(self) -> int:
        with self._lock:
            return sum(1 for r in self._records if r.label is not None)

    # ------------------------------------------------------------------
    # readers
    # ------------------------------------------------------------------

    def snapshot(
        self,
        *,
        window: timedelta | None = None,
        now: datetime | None = None,
    ) -> PerformanceSnapshot:
        """Compute classification metrics over the requested window.

        Parameters
        ----------
        window
            Only consider records with ``timestamp >= now - window``. When
            ``None`` -- use the entire buffer.
        now
            Reference time. Defaults to ``datetime.now(timezone.utc)``.
        """
        now = now or datetime.now(timezone.utc)
        with self._lock:
            records = list(self._records)

        if window is not None:
            cutoff = now - window
            records = [r for r in records if r.timestamp >= cutoff]

        n_total = len(records)
        labelled = [r for r in records if r.label is not None]
        n_labelled = len(labelled)

        if not labelled:
            return PerformanceSnapshot(
                n_total=n_total,
                n_labelled=0,
                threshold=self.threshold,
                window_start=records[0].timestamp if records else None,
                window_end=records[-1].timestamp if records else None,
            )

        y_true = np.array([r.label for r in labelled], dtype=np.int64)
        y_score = np.array([r.score for r in labelled], dtype=np.float64)
        y_pred = (y_score >= self.threshold).astype(np.int64)

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-12) if (tp + fp + fn) else 0.0
        accuracy = (tp + tn) / max(n_labelled, 1)

        # sklearn returns the same numbers but its support array is convenient
        # for sanity-checking when present; falls back to manual otherwise.
        if _HAVE_SKLEARN and len(np.unique(y_true)) == 2:
            try:
                precision_sk, recall_sk, f1_sk, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="binary", zero_division=0
                )
                precision = float(precision_sk)
                recall = float(recall_sk)
                f1 = float(f1_sk)
            except Exception:  # pragma: no cover -- degraded class distribution
                pass

        au_roc = auroc(y_true, y_score) if len(np.unique(y_true)) == 2 else None
        au_prc = auprc(y_true, y_score) if (y_true == 1).any() else None
        if au_roc is not None and math.isnan(au_roc):
            au_roc = None
        if au_prc is not None and math.isnan(au_prc):
            au_prc = None

        brier = float(np.mean((y_score - y_true) ** 2))

        return PerformanceSnapshot(
            n_total=n_total,
            n_labelled=n_labelled,
            n_positive=int((y_true == 1).sum()),
            n_predicted_positive=int(y_pred.sum()),
            threshold=self.threshold,
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            accuracy=float(accuracy),
            auroc=au_roc,
            auprc=au_prc,
            brier=brier,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            window_start=records[0].timestamp if records else None,
            window_end=records[-1].timestamp if records else None,
        )

    # ------------------------------------------------------------------
    # exports
    # ------------------------------------------------------------------

    def log_mlflow(
        self,
        snapshot: PerformanceSnapshot,
        *,
        run_name: str = "production",
        experiment: str = "meshwatch-production",
    ) -> bool:
        """Push a snapshot to MLflow as a fresh run inside an experiment.

        Returns ``True`` when MLflow is installed *and* a tracking server is
        reachable (or a local SQLite store is writable). Returns ``False``
        silently otherwise so callers can keep operating in degraded mode.
        """
        try:  # pragma: no cover -- exercised in integration env only
            import mlflow

            mlflow.set_experiment(experiment)
            with mlflow.start_run(run_name=run_name):
                params = {
                    "threshold": snapshot.threshold,
                    "n_total": snapshot.n_total,
                    "n_labelled": snapshot.n_labelled,
                }
                for k, v in params.items():
                    mlflow.log_param(k, v)
                metrics = {
                    "precision": snapshot.precision,
                    "recall": snapshot.recall,
                    "f1": snapshot.f1,
                    "accuracy": snapshot.accuracy,
                    "brier": snapshot.brier or 0.0,
                }
                if snapshot.auroc is not None:
                    metrics["auroc"] = snapshot.auroc
                if snapshot.auprc is not None:
                    metrics["auprc"] = snapshot.auprc
                for k, v in metrics.items():
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        continue
                    mlflow.log_metric(k, float(v))
            return True
        except Exception:
            return False


__all__ = [
    "PerformanceSnapshot",
    "PerformanceTracker",
    "auprc",
    "auroc",
]
