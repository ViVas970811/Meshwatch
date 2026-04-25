"""Tests for ``fraud_detection.models.gnn_layers``."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from fraud_detection.models.gnn_layers import GAT_RELATIONS, HeteroGNNLayer, _is_gat_edge


def _toy_hetero(hidden: int = 16) -> HeteroData:
    """Build a tiny HeteroData with the canonical 3-node-type / 3-edge-type layout."""
    data = HeteroData()
    data["transaction"].x = torch.randn(20, hidden)
    data["card"].x = torch.randn(8, hidden)
    data["device"].x = torch.randn(5, hidden)

    # transaction -> card (SAGE)
    src = torch.randint(0, 20, (30,))
    dst = torch.randint(0, 8, (30,))
    data["transaction", "uses_card", "card"].edge_index = torch.stack([src, dst])

    # transaction -> device (SAGE)
    src = torch.randint(0, 20, (25,))
    dst = torch.randint(0, 5, (25,))
    data["transaction", "from_device", "device"].edge_index = torch.stack([src, dst])

    # card <-> card (GAT)
    src = torch.randint(0, 8, (12,))
    dst = torch.randint(0, 8, (12,))
    data["card", "shared_address", "card"].edge_index = torch.stack([src, dst])
    return data


def test_is_gat_edge_for_card_card():
    assert _is_gat_edge(("card", "shared_address", "card"))
    assert _is_gat_edge(("card", "shared_device", "card"))
    assert _is_gat_edge(("card", "rev_shared_address", "card"))


def test_is_gat_edge_for_tx_entity():
    assert not _is_gat_edge(("transaction", "uses_card", "card"))
    assert not _is_gat_edge(("transaction", "from_device", "device"))


def test_layer_forward_runs():
    data = _toy_hetero()
    layer = HeteroGNNLayer(
        edge_types=data.edge_types,
        node_types=data.node_types,
        in_dim=16,
        out_dim=16,
        heads=2,
    )
    out = layer(dict(data.x_dict), dict(data.edge_index_dict))
    assert set(out.keys()) == {"transaction", "card", "device"}
    for nt in out:
        assert out[nt].shape == data[nt].x.shape, f"shape changed on {nt}"
        assert torch.isfinite(out[nt]).all()


def test_layer_changes_dim():
    """When in_dim != out_dim, residual goes through projection."""
    data = _toy_hetero(hidden=16)
    layer = HeteroGNNLayer(
        edge_types=data.edge_types,
        node_types=data.node_types,
        in_dim=16,
        out_dim=32,
    )
    out = layer(dict(data.x_dict), dict(data.edge_index_dict))
    for nt in data.node_types:
        assert out[nt].shape == (data[nt].x.shape[0], 32)


def test_layer_dispatches_correct_conv_types():
    data = _toy_hetero()
    layer = HeteroGNNLayer(
        edge_types=data.edge_types,
        node_types=data.node_types,
        in_dim=16,
        out_dim=16,
    )
    # The HeteroConv has a ``convs`` attribute mapping edge -> module.
    convs = layer.conv.convs
    from torch_geometric.nn import GATConv, SAGEConv

    for et, conv in convs.items():
        if _is_gat_edge(et):
            assert isinstance(conv, GATConv), f"{et} should be GAT, got {type(conv)}"
        else:
            assert isinstance(conv, SAGEConv), f"{et} should be SAGE, got {type(conv)}"


def test_layer_gradients_flow():
    data = _toy_hetero()
    layer = HeteroGNNLayer(
        edge_types=data.edge_types,
        node_types=data.node_types,
        in_dim=16,
        out_dim=16,
    )
    x_dict = {nt: data[nt].x.clone().requires_grad_() for nt in data.node_types}
    out = layer(x_dict, dict(data.edge_index_dict))
    loss = sum(o.sum() for o in out.values())
    loss.backward()
    for nt in data.node_types:
        assert x_dict[nt].grad is not None
        assert torch.isfinite(x_dict[nt].grad).all()


def test_layer_validates_dims():
    data = _toy_hetero()
    with pytest.raises(ValueError, match="positive"):
        HeteroGNNLayer(
            edge_types=data.edge_types,
            node_types=data.node_types,
            in_dim=0,
            out_dim=16,
        )


def test_gat_relations_constant():
    assert "shared_address" in GAT_RELATIONS
    assert "shared_device" in GAT_RELATIONS
    assert "rev_shared_address" in GAT_RELATIONS
