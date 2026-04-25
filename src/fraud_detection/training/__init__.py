"""Training loop, callbacks, evaluator (Phase 3)."""

from fraud_detection.training.callbacks import EarlyStopping, ModelCheckpoint
from fraud_detection.training.evaluator import (
    EvaluationResult,
    evaluate_predictions,
    plot_calibration_curve,
    plot_pr_curve,
    plot_roc_curve,
    write_evaluation_report,
)
from fraud_detection.training.trainer import Trainer, TrainerConfig, ensure_temporal_masks

__all__ = [
    "EarlyStopping",
    "EvaluationResult",
    "ModelCheckpoint",
    "Trainer",
    "TrainerConfig",
    "ensure_temporal_masks",
    "evaluate_predictions",
    "plot_calibration_curve",
    "plot_pr_curve",
    "plot_roc_curve",
    "write_evaluation_report",
]
