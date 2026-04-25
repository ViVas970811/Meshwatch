"""Training callbacks: early-stopping + model checkpointing.

The plan calls for ``patience=15`` epochs on validation AUPRC. We expose
that as a small, side-effect-free state machine -- the trainer asks the
callback whether to stop and which checkpoint to load at the end.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)

Mode = Literal["max", "min"]


@dataclass
class EarlyStopping:
    """Stop training when ``metric`` hasn't improved for ``patience`` epochs.

    Parameters
    ----------
    patience
        Number of epochs without improvement before stopping. Plan: 15.
    mode
        ``"max"`` (e.g. AUPRC, AUROC) or ``"min"`` (e.g. loss).
    min_delta
        Minimum change in the metric to qualify as an improvement.
    """

    patience: int = 15
    mode: Mode = "max"
    min_delta: float = 1e-4
    best_score: float | None = field(default=None, init=False)
    best_epoch: int = field(default=-1, init=False)
    counter: int = field(default=0, init=False)
    should_stop: bool = field(default=False, init=False)

    def step(self, score: float, *, epoch: int) -> bool:
        """Record a new score; return True if training should stop now."""
        improved = self._is_improvement(score)
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def reset(self) -> None:
        self.best_score = None
        self.best_epoch = -1
        self.counter = 0
        self.should_stop = False


@dataclass
class ModelCheckpoint:
    """Track the best model state by validation metric.

    Two storage modes:
    * In-memory only (default): ``best_state_dict`` holds the deep-copied
      ``state_dict`` of the best model so we can restore at the end of
      training without touching disk -- fast and avoids polluting the
      filesystem during hyperparam search.
    * On-disk: pass ``path`` to also write a checkpoint file every time we
      improve.

    Parameters
    ----------
    monitor
        Name of the metric we minimise/maximise (informational only).
    mode
        ``"max"`` for AUPRC/AUROC, ``"min"`` for loss.
    path
        Optional file path; if given, write the state dict there each
        improvement.
    """

    monitor: str = "val_auprc"
    mode: Mode = "max"
    path: Path | None = None
    best_score: float | None = field(default=None, init=False)
    best_epoch: int = field(default=-1, init=False)
    best_state_dict: dict[str, torch.Tensor] | None = field(default=None, init=False)
    extra: dict[str, Any] = field(default_factory=dict, init=False)

    def step(
        self,
        score: float,
        *,
        epoch: int,
        model: nn.Module,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """If ``score`` improves, snapshot the model. Returns ``True`` on update."""
        if not self._is_improvement(score):
            return False
        self.best_score = score
        self.best_epoch = epoch
        self.best_state_dict = deepcopy(model.state_dict())
        self.extra = dict(extra or {})
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": self.best_state_dict,
                    "epoch": epoch,
                    "score": score,
                    "monitor": self.monitor,
                    "extra": self.extra,
                },
                self.path,
            )
            log.info(
                "checkpoint_saved",
                path=str(self.path),
                epoch=epoch,
                metric=self.monitor,
                score=score,
            )
        return True

    def restore(self, model: nn.Module) -> nn.Module:
        """Load the best state dict back into ``model`` (in place)."""
        if self.best_state_dict is None:
            log.warning("checkpoint_restore_no_state")
            return model
        model.load_state_dict(self.best_state_dict)
        log.info(
            "checkpoint_restored",
            epoch=self.best_epoch,
            metric=self.monitor,
            score=self.best_score,
        )
        return model

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        return score > self.best_score if self.mode == "max" else score < self.best_score


__all__ = ["EarlyStopping", "Mode", "ModelCheckpoint"]
