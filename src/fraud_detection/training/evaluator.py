"""Evaluation metrics + report generation for the fraud detector.

Phase 3 acceptance asks for an evaluation report containing PR/ROC curves
and a calibration plot. This module computes the underlying metrics (so
they're CI-testable) and renders the figures via matplotlib.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class EvaluationResult:
    """All scalar metrics from a single evaluation.

    The Phase 3 acceptance criteria check ``auprc`` (>= 0.65 GNN-only,
    >= 0.70 ensemble); the rest are reported for completeness.
    """

    n_total: int
    n_positive: int
    fraud_rate: float
    auprc: float
    auroc: float
    log_loss: float
    best_threshold: float
    best_f1: float
    best_precision: float
    best_recall: float
    precision_at_top_pct: dict[str, float] = field(default_factory=dict)
    confusion_matrix_at_threshold: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float | int | dict]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"AUPRC={self.auprc:.4f}  AUROC={self.auroc:.4f}  "
            f"LogLoss={self.log_loss:.4f}  best_F1={self.best_f1:.4f} "
            f"@thr={self.best_threshold:.3f}"
        )


def _precision_at_top_k(y_true: np.ndarray, y_score: np.ndarray, pct: float) -> float:
    """Precision in the top ``pct`` (e.g. 0.01 = top 1%) of scores."""
    n = len(y_score)
    k = max(1, int(np.ceil(n * pct)))
    top_idx = np.argpartition(-y_score, k - 1)[:k]
    return float(y_true[top_idx].mean())


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray) -> EvaluationResult:
    """Compute the full Phase 3 metric bundle from raw predictions."""
    if y_true.shape != y_score.shape:
        msg = f"shape mismatch: y_true={y_true.shape}, y_score={y_score.shape}"
        raise ValueError(msg)
    if len(y_true) == 0:
        msg = "y_true is empty"
        raise ValueError(msg)
    y_true = np.asarray(y_true).astype(np.int8)
    y_score = np.asarray(y_score).astype(np.float64)

    n = len(y_true)
    n_pos = int(y_true.sum())
    fraud_rate = float(n_pos / n)

    auprc = float(average_precision_score(y_true, y_score)) if n_pos > 0 else float("nan")
    auroc = float(roc_auc_score(y_true, y_score)) if n_pos > 0 and n_pos < n else float("nan")
    ll = float(log_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7), labels=[0, 1]))

    # Threshold sweep on the PR curve to find best F1.
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1s = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-9, None)
    best_idx = int(np.argmax(f1s))
    # PR curve thresholds has length len(precisions)-1 -- last index has no thr.
    best_thr = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1 = float(f1s[best_idx])
    best_p = float(precisions[best_idx])
    best_r = float(recalls[best_idx])

    # Precision in top-k buckets (operational metrics analysts care about).
    top_pct: dict[str, float] = {}
    for pct in (0.001, 0.01, 0.05, 0.1):
        if int(np.ceil(n * pct)) >= 1:
            top_pct[f"top_{pct:g}"] = _precision_at_top_k(y_true, y_score, pct)

    # Confusion matrix at best threshold.
    y_pred = (y_score >= best_thr).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cm = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    result = EvaluationResult(
        n_total=int(n),
        n_positive=int(n_pos),
        fraud_rate=fraud_rate,
        auprc=auprc,
        auroc=auroc,
        log_loss=ll,
        best_threshold=best_thr,
        best_f1=best_f1,
        best_precision=best_p,
        best_recall=best_r,
        precision_at_top_pct=top_pct,
        confusion_matrix_at_threshold=cm,
    )
    log.info(
        "evaluation_complete",
        **{k: v for k, v in asdict(result).items() if not isinstance(v, dict)},
    )
    return result


# ---------------------------------------------------------------------------
# Plotting -- imports matplotlib lazily so headless test environments don't
# need it just to import the module.
# ---------------------------------------------------------------------------


def _lazy_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, *, save_path: Path | None = None):
    plt = _lazy_mpl()
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(recalls, precisions, lw=2, label=f"AUPRC = {auprc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, *, save_path: Path | None = None):
    plt = _lazy_mpl()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUROC = {auroc:.4f}")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", alpha=0.6, label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int = 10,
    save_path: Path | None = None,
):
    plt = _lazy_mpl()
    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", alpha=0.6, label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", lw=2, label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction positive")
    ax.set_title(f"Calibration ({n_bins}-bin quantile)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig


def write_evaluation_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    output_dir: Path | str,
    name: str = "eval",
) -> dict[str, Path]:
    """Compute metrics + render the three plots, return dict of artefacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate_predictions(y_true, y_score)

    paths: dict[str, Path] = {}
    paths["metrics"] = output_dir / f"{name}_metrics.json"
    paths["metrics"].write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    paths["pr"] = output_dir / f"{name}_pr.png"
    plot_pr_curve(y_true, y_score, save_path=paths["pr"])
    paths["roc"] = output_dir / f"{name}_roc.png"
    plot_roc_curve(y_true, y_score, save_path=paths["roc"])
    paths["calibration"] = output_dir / f"{name}_calibration.png"
    plot_calibration_curve(y_true, y_score, save_path=paths["calibration"])
    log.info("evaluation_report_written", dir=str(output_dir), files=list(paths.keys()))
    return paths


__all__ = [
    "EvaluationResult",
    "evaluate_predictions",
    "plot_calibration_curve",
    "plot_pr_curve",
    "plot_roc_curve",
    "write_evaluation_report",
]
