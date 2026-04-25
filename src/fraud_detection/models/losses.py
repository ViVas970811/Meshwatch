"""Loss functions for fraud-detection training.

The IEEE-CIS dataset is heavily imbalanced (3.5% positive rate, 27.6:1
neg/pos ratio), so vanilla BCE under-weights the rare-but-important fraud
class. We use **Focal Loss** (Lin et al. 2017, https://arxiv.org/abs/1708.02002)
with the implementation-plan settings:

    alpha = 0.75     # up-weight positives
    gamma = 2.0      # focus on hard examples

Focal Loss applies a modulating factor ``(1 - p_t)^gamma`` that down-weights
easy examples (p_t -> 1) and emphasises hard ones (p_t -> 0). This is the
right tool for fraud detection where the model can trivially memorise the
majority non-fraud class.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 -- conventional alias


class FocalLoss(nn.Module):
    """Binary Focal Loss with class-balancing alpha.

    Parameters
    ----------
    alpha
        Weight applied to the positive class. Higher = stronger upweight of
        the rare class. Plan default: 0.75.
    gamma
        Focusing parameter. 0 reduces to weighted BCE; higher values further
        suppress easy negatives. Plan default: 2.0.
    reduction
        ``"mean"`` (default), ``"sum"``, or ``"none"``.
    pos_weight
        Optional ``torch.Tensor`` for additional positive-class scaling on top
        of ``alpha``. Mutually compatible -- use one or both.

    Notes
    -----
    Operates on **raw logits** (not sigmoids). This avoids the numerical
    instability of computing log(sigmoid(x)) explicitly: we use
    ``binary_cross_entropy_with_logits`` under the hood which applies the
    standard log-sum-exp trick.
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            msg = f"alpha must be in [0, 1] (got {alpha})"
            raise ValueError(msg)
        if gamma < 0.0:
            msg = f"gamma must be >= 0 (got {gamma})"
            raise ValueError(msg)
        if reduction not in ("mean", "sum", "none"):
            msg = f"reduction must be one of mean/sum/none (got '{reduction}')"
            raise ValueError(msg)

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the focal loss.

        Parameters
        ----------
        logits
            Raw model outputs, any shape. Will be flattened.
        targets
            Binary labels (0 or 1), same shape as ``logits``.
        """
        if logits.shape != targets.shape:
            msg = (
                f"logits and targets must have the same shape "
                f"(got {tuple(logits.shape)} vs {tuple(targets.shape)})"
            )
            raise ValueError(msg)

        targets_f = targets.float()
        # BCE per element (no reduction yet) -- numerically stable via
        # log-sum-exp internally.
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets_f,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        # p_t = p if y==1 else 1-p
        p = torch.sigmoid(logits)
        p_t = p * targets_f + (1.0 - p) * (1.0 - targets_f)
        # alpha_t = alpha if y==1 else 1-alpha
        alpha_t = self.alpha * targets_f + (1.0 - self.alpha) * (1.0 - targets_f)
        # focal factor = (1 - p_t)^gamma
        focal = (1.0 - p_t).clamp(min=0.0).pow(self.gamma)

        loss = alpha_t * focal * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, gamma={self.gamma}, reduction={self.reduction!r}"


__all__ = ["FocalLoss"]
