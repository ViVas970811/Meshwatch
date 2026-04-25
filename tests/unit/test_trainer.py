"""Tests for ``fraud_detection.training.trainer.Trainer``."""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

from fraud_detection.models import FraudHeteroGNN
from fraud_detection.training import Trainer, TrainerConfig, ensure_temporal_masks


def _learnable_hetero(n_tx: int = 200, seed: int = 0) -> HeteroData:
    """Build a tiny HeteroData where the label is recoverable from features.

    Specifically: y[i] = 1 iff transaction.x[i, 0] > 0. The GNN doesn't
    even need to use graph structure to learn this -- but it gives the
    trainer a signal so we can validate that AUPRC > 0.5 in a few epochs.
    """
    g = torch.Generator().manual_seed(seed)
    data = HeteroData()
    tx = torch.randn(n_tx, 4, generator=g)
    data["transaction"].x = tx
    data["transaction"].y = (tx[:, 0] > 0).int()

    n_card = 20
    data["card"].x = torch.randn(n_card, 3, generator=g)
    src = torch.arange(n_tx)
    dst = torch.randint(0, n_card, (n_tx,), generator=g)
    data["transaction", "uses_card", "card"].edge_index = torch.stack([src, dst])

    src = torch.randint(0, n_card, (30,), generator=g)
    dst = torch.randint(0, n_card, (30,), generator=g)
    data["card", "shared_address", "card"].edge_index = torch.stack([src, dst])
    return data


def _make_model(data: HeteroData) -> FraudHeteroGNN:
    node_dims = {nt: data[nt].num_node_features for nt in data.node_types}
    return FraudHeteroGNN(
        node_feature_dims=node_dims,
        edge_types=data.edge_types,
        hidden_dim=16,
        embedding_dim=8,
        n_layers=2,
        heads=2,
        classifier_hidden=8,
    )


def test_ensure_temporal_masks_sets_60_20_20():
    data = _learnable_hetero(n_tx=100)
    data = ensure_temporal_masks(data, target_node_type="transaction")
    train = int(data["transaction"].train_mask.sum())
    val = int(data["transaction"].val_mask.sum())
    test = int(data["transaction"].test_mask.sum())
    assert train == 60
    assert val == 20
    assert test == 20


def test_ensure_temporal_masks_idempotent():
    data = _learnable_hetero(n_tx=100)
    a = ensure_temporal_masks(data, target_node_type="transaction")
    snapshot_train = a["transaction"].train_mask.clone()
    b = ensure_temporal_masks(a, target_node_type="transaction")
    assert torch.equal(snapshot_train, b["transaction"].train_mask)


def test_trainer_runs_few_epochs_without_error():
    data = _learnable_hetero(n_tx=120)
    data = ToUndirected()(data)
    data = ensure_temporal_masks(data, target_node_type="transaction")
    model = _make_model(data)
    cfg = TrainerConfig(
        epochs=3,
        batch_size=32,
        early_stop_patience=10,
        mlflow_enabled=False,
        log_every_n_epochs=10,
    )
    trainer = Trainer(model, cfg)
    out = trainer.fit(data)
    assert "model" in out
    assert "history" in out
    assert len(out["history"]) == 3
    for row in out["history"]:
        for k in ("train_loss", "val_auprc", "val_auroc", "lr"):
            assert k in row


def test_trainer_history_loss_decreases_on_learnable_signal():
    """On a trivially-learnable problem the loss should drop in 5 epochs."""
    torch.manual_seed(7)
    data = _learnable_hetero(n_tx=300)
    data = ToUndirected()(data)
    data = ensure_temporal_masks(data, target_node_type="transaction")
    model = _make_model(data)
    cfg = TrainerConfig(
        epochs=5,
        batch_size=64,
        learning_rate=5e-3,
        early_stop_patience=20,
        mlflow_enabled=False,
        log_every_n_epochs=10,
    )
    trainer = Trainer(model, cfg)
    out = trainer.fit(data)
    losses = [row["train_loss"] for row in out["history"]]
    # Final loss should be at most the initial loss (allowing for noise).
    assert losses[-1] <= losses[0] + 1e-3, f"Loss did not decrease: {losses}"


def test_trainer_early_stop_records_best_epoch():
    data = _learnable_hetero(n_tx=120)
    data = ToUndirected()(data)
    data = ensure_temporal_masks(data, target_node_type="transaction")
    model = _make_model(data)
    cfg = TrainerConfig(
        epochs=3,
        batch_size=32,
        early_stop_patience=2,
        mlflow_enabled=False,
        log_every_n_epochs=10,
    )
    trainer = Trainer(model, cfg)
    out = trainer.fit(data)
    assert out["best_epoch"] >= 0
    assert out["best_val_auprc"] is not None and out["best_val_auprc"] >= 0


def test_trainer_requires_masks():
    data = _learnable_hetero(n_tx=80)
    data = ToUndirected()(data)
    # Don't call ensure_temporal_masks -- training should fail loudly.
    model = _make_model(data)
    trainer = Trainer(model, TrainerConfig(epochs=1, mlflow_enabled=False))
    import pytest

    with pytest.raises(ValueError, match="train_mask"):
        trainer.fit(data)
