"""Tests for ``fraud_detection.models.hetero_gnn.FraudHeteroGNN``."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from fraud_detection.models.hetero_gnn import FraudHeteroGNN


def _toy_data(n_tx: int = 50, n_card: int = 10, n_device: int = 5) -> HeteroData:
    data = HeteroData()
    data["transaction"].x = torch.randn(n_tx, 8)
    data["card"].x = torch.randn(n_card, 4)
    data["device"].x = torch.randn(n_device, 3)
    data["transaction"].y = torch.randint(0, 2, (n_tx,)).int()

    src = torch.randint(0, n_tx, (n_tx,))
    dst = torch.randint(0, n_card, (n_tx,))
    data["transaction", "uses_card", "card"].edge_index = torch.stack([src, dst])

    src = torch.randint(0, n_tx, (n_tx,))
    dst = torch.randint(0, n_device, (n_tx,))
    data["transaction", "from_device", "device"].edge_index = torch.stack([src, dst])

    src = torch.randint(0, n_card, (15,))
    dst = torch.randint(0, n_card, (15,))
    data["card", "shared_address", "card"].edge_index = torch.stack([src, dst])
    return data


def test_forward_returns_logits_per_transaction():
    data = _toy_data()
    model = FraudHeteroGNN(
        node_feature_dims={"transaction": 8, "card": 4, "device": 3},
        edge_types=data.edge_types,
    )
    logits = model(data)
    assert logits.shape == (data["transaction"].num_nodes,)
    assert torch.isfinite(logits).all()


def test_get_embeddings_default_dim_64():
    data = _toy_data()
    model = FraudHeteroGNN(
        node_feature_dims={"transaction": 8, "card": 4, "device": 3},
        edge_types=data.edge_types,
    )
    emb = model.get_embeddings(data)
    assert emb.shape == (data["transaction"].num_nodes, 64)


def test_get_embeddings_subset():
    data = _toy_data()
    model = FraudHeteroGNN(
        node_feature_dims={"transaction": 8, "card": 4, "device": 3},
        edge_types=data.edge_types,
    )
    target_idx = torch.tensor([1, 5, 10, 25])
    emb = model.get_embeddings(data, target_indices=target_idx)
    assert emb.shape == (4, 64)


def test_custom_dims():
    data = _toy_data()
    model = FraudHeteroGNN(
        node_feature_dims={"transaction": 8, "card": 4, "device": 3},
        edge_types=data.edge_types,
        hidden_dim=64,
        embedding_dim=32,
        n_layers=2,
        heads=2,
        classifier_hidden=16,
    )
    emb = model.get_embeddings(data)
    assert emb.shape == (data["transaction"].num_nodes, 32)


def test_n_parameters_positive():
    data = _toy_data()
    model = FraudHeteroGNN(
        node_feature_dims={"transaction": 8, "card": 4, "device": 3},
        edge_types=data.edge_types,
    )
    assert model.n_parameters() > 0


def test_validation_invalid_target():
    with pytest.raises(ValueError, match="target_node_type"):
        FraudHeteroGNN(
            node_feature_dims={"foo": 8},
            edge_types=[("foo", "rel", "foo")],
            target_node_type="transaction",  # not in dims
        )


def test_validation_n_layers():
    with pytest.raises(ValueError, match="n_layers"):
        FraudHeteroGNN(
            node_feature_dims={"transaction": 8, "card": 4, "device": 3},
            edge_types=[("transaction", "uses_card", "card")],
            n_layers=0,
        )


def test_state_dict_roundtrip():
    data = _toy_data()
    a = FraudHeteroGNN(
        node_feature_dims={"transaction": 8, "card": 4, "device": 3},
        edge_types=data.edge_types,
    )
    b = FraudHeteroGNN(
        node_feature_dims={"transaction": 8, "card": 4, "device": 3},
        edge_types=data.edge_types,
    )
    b.load_state_dict(a.state_dict())
    a.eval()
    b.eval()
    with torch.no_grad():
        torch.testing.assert_close(a(data), b(data))


def test_gradient_flow():
    data = _toy_data()
    model = FraudHeteroGNN(
        node_feature_dims={"transaction": 8, "card": 4, "device": 3},
        edge_types=data.edge_types,
    )
    logits = model(data)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, data["transaction"].y.float()
    )
    loss.backward()
    n_with_grad = sum(
        1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all()
    )
    assert n_with_grad > 0
