"""Tests for ``fraud_detection.models.losses.FocalLoss``."""

from __future__ import annotations

import math

import pytest
import torch

from fraud_detection.models.losses import FocalLoss


def test_focal_reduces_to_weighted_bce_when_gamma_zero():
    """gamma=0 -> alpha-weighted BCE. Sanity-check via numerical comparison."""
    logits = torch.tensor([0.5, -0.5, 2.0, -2.0])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss_fn = FocalLoss(alpha=0.5, gamma=0.0, reduction="none")
    out = loss_fn(logits, targets)

    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    expected = 0.5 * bce  # alpha=0.5 applies symmetrically
    torch.testing.assert_close(out, expected)


def test_focal_downweights_easy_examples():
    """Easy positive (p ~ 1) should have far smaller loss than hard one."""
    logits = torch.tensor([10.0, 0.1])  # very confident vs. uncertain positive
    targets = torch.tensor([1.0, 1.0])
    losses = FocalLoss(alpha=0.75, gamma=2.0, reduction="none")(logits, targets)
    assert losses[0].item() < losses[1].item() * 1e-3, (
        f"Easy positive should be heavily down-weighted "
        f"(got {losses[0].item()} vs {losses[1].item()})"
    )


def test_focal_upweights_positives_when_alpha_high():
    logits = torch.tensor([0.0, 0.0])
    targets_pos = torch.tensor([1.0, 1.0])
    targets_neg = torch.tensor([0.0, 0.0])
    pos_loss = FocalLoss(alpha=0.9, gamma=0.0)(logits, targets_pos)
    neg_loss = FocalLoss(alpha=0.9, gamma=0.0)(logits, targets_neg)
    # alpha = 0.9 -> positives weighted 9x heavier than negatives.
    assert pos_loss.item() > 8 * neg_loss.item()


def test_focal_mean_reduction_is_average_of_none():
    logits = torch.randn(50)
    targets = torch.randint(0, 2, (50,)).float()
    none = FocalLoss(reduction="none")(logits, targets)
    mean = FocalLoss(reduction="mean")(logits, targets)
    torch.testing.assert_close(mean, none.mean())


def test_focal_sum_reduction():
    logits = torch.randn(20)
    targets = torch.randint(0, 2, (20,)).float()
    none = FocalLoss(reduction="none")(logits, targets)
    s = FocalLoss(reduction="sum")(logits, targets)
    torch.testing.assert_close(s, none.sum())


def test_focal_validates_shapes():
    pp = FocalLoss()
    with pytest.raises(ValueError, match="same shape"):
        pp(torch.zeros(5), torch.zeros(7))


def test_focal_validates_alpha():
    with pytest.raises(ValueError, match="alpha"):
        FocalLoss(alpha=1.5)
    with pytest.raises(ValueError, match="alpha"):
        FocalLoss(alpha=-0.1)


def test_focal_validates_gamma():
    with pytest.raises(ValueError, match="gamma"):
        FocalLoss(gamma=-1.0)


def test_focal_validates_reduction():
    with pytest.raises(ValueError, match="reduction"):
        FocalLoss(reduction="banana")


def test_focal_gradient_flows():
    logits = torch.randn(30, requires_grad=True)
    targets = torch.randint(0, 2, (30,)).float()
    loss = FocalLoss()(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_focal_handles_extreme_logits_without_nan():
    """Stress test with extreme logits to check numerical stability."""
    logits = torch.tensor([1e9, -1e9, 1e9, -1e9])
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    loss = FocalLoss()(logits, targets)
    assert torch.isfinite(loss).all(), f"Got non-finite loss: {loss}"


def test_focal_repr_includes_params():
    loss = FocalLoss(alpha=0.7, gamma=1.5)
    s = repr(loss)
    assert "0.7" in s
    assert "1.5" in s


def test_focal_is_module():
    """Should be a proper nn.Module so it can be pickled into checkpoints."""
    loss = FocalLoss()
    assert isinstance(loss, torch.nn.Module)
    # Make sure named_parameters / state_dict round-trip cleanly.
    sd = loss.state_dict()
    loss2 = FocalLoss()
    loss2.load_state_dict(sd)


def test_focal_with_pos_weight_buffer():
    pw = torch.tensor([10.0])
    loss = FocalLoss(pos_weight=pw)
    logits = torch.tensor([0.5, -0.5])
    targets = torch.tensor([1.0, 0.0])
    out = loss(logits, targets)
    assert math.isfinite(out.item())
