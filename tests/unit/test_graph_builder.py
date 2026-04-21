"""Tests for ``fraud_detection.data.graph_builder``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

from fraud_detection.data.graph_builder import (
    EDGE_TYPES,
    NODE_FEATURE_DIMS,
    NODE_TYPES,
    HeteroGraphBuilder,
)


@pytest.fixture
def synthetic_processed_df() -> pd.DataFrame:
    """Schema-faithful small frame (~250 rows) resembling post-preprocessing output."""
    rng = np.random.default_rng(42)
    n = 250
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "isFraud": rng.choice([0, 1], size=n, p=[0.96, 0.04]).astype(np.int8),
            "TransactionDT": np.sort(rng.integers(0, 30 * 86400, size=n)).astype(np.int32),
            "TransactionAmt": np.abs(rng.normal(75, 40, n)).clip(min=1.0).astype(np.float32),
            "TransactionAmt__log1p": np.log1p(np.abs(rng.normal(75, 40, n))).astype(np.float32),
            "ProductCD": rng.choice(list("WCHSR"), size=n),
            "card1": rng.integers(1000, 1050, n),
            "card2": rng.integers(100, 600, size=n).astype(float),
            "card3": rng.integers(100, 200, size=n).astype(float),
            "card4": rng.choice(["visa", "mastercard", "amex"], size=n),
            "card5": rng.integers(100, 300, size=n).astype(float),
            "card6": rng.choice(["debit", "credit"], size=n),
            "addr1": rng.integers(100, 500, n).astype(float),
            "addr2": rng.integers(10, 99, n).astype(float),
            "dist1": rng.uniform(0, 100, n).astype(np.float32),
            "dist2": rng.uniform(0, 100, n).astype(np.float32),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com"], size=n),
            "DeviceType": rng.choice(["desktop", "mobile"], size=n),
            "DeviceInfo": rng.choice(["Windows", "iPhone", "Android"], size=n),
        }
    )
    for i in range(1, 7):
        df[f"id_0{i}"] = rng.normal(size=n)
    for i in range(307, 317):
        df[f"V{i}"] = rng.normal(size=n).astype(np.float32)
    for i in range(1, 15):
        df[f"C{i}"] = rng.integers(0, 10, n).astype(np.float32)
    for i in range(1, 16):
        df[f"D{i}"] = rng.integers(0, 365, n).astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Construction / schema
# ---------------------------------------------------------------------------


def test_build_hetero_data_returns_hetero_data(synthetic_processed_df: pd.DataFrame):
    gb = HeteroGraphBuilder()
    data = gb.build_hetero_data(synthetic_processed_df)
    assert isinstance(data, HeteroData)


def test_all_node_types_present_with_declared_dims(synthetic_processed_df: pd.DataFrame):
    data = HeteroGraphBuilder().build_hetero_data(synthetic_processed_df)
    for node_type in NODE_TYPES:
        assert node_type in data.node_types, f"missing {node_type}"
        assert data[node_type].num_nodes > 0, f"{node_type} has zero nodes"
        assert data[node_type].x.shape[1] == NODE_FEATURE_DIMS[node_type]


def test_all_edge_types_present(synthetic_processed_df: pd.DataFrame):
    data = HeteroGraphBuilder().build_hetero_data(synthetic_processed_df)
    for edge_type in EDGE_TYPES:
        assert edge_type in data.edge_types, f"missing {edge_type}"
        edge_index = data[edge_type].edge_index
        assert edge_index.shape[0] == 2
        assert edge_index.dtype == torch.long


def test_transaction_tx_entity_edges_match_row_count(synthetic_processed_df: pd.DataFrame):
    """Each transaction must emit exactly one edge per tx->entity relation."""
    data = HeteroGraphBuilder().build_hetero_data(synthetic_processed_df)
    n = len(synthetic_processed_df)
    for edge_type in EDGE_TYPES:
        if edge_type[0] == "transaction":
            assert data[edge_type].num_edges == n, (
                f"{edge_type} has {data[edge_type].num_edges} edges, expected {n}"
            )


def test_labels_match_input(synthetic_processed_df: pd.DataFrame):
    data = HeteroGraphBuilder().build_hetero_data(synthetic_processed_df)
    assert hasattr(data["transaction"], "y")
    assert int(data["transaction"].y.sum()) == int(synthetic_processed_df["isFraud"].sum())


def test_masks_attached_when_provided(synthetic_processed_df: pd.DataFrame):
    n = len(synthetic_processed_df)
    train = pd.Series([i < 150 for i in range(n)])
    val = pd.Series([150 <= i < 200 for i in range(n)])
    test = pd.Series([i >= 200 for i in range(n)])
    data = HeteroGraphBuilder().build_hetero_data(
        synthetic_processed_df,
        train_mask=train,
        val_mask=val,
        test_mask=test,
    )
    for mask_name, expected in (("train_mask", 150), ("val_mask", 50), ("test_mask", 50)):
        mask = getattr(data["transaction"], mask_name)
        assert mask.dtype == torch.bool
        assert int(mask.sum()) == expected


def test_node_features_are_float32(synthetic_processed_df: pd.DataFrame):
    data = HeteroGraphBuilder().build_hetero_data(synthetic_processed_df)
    for node_type in NODE_TYPES:
        assert data[node_type].x.dtype == torch.float32


def test_no_nan_in_node_features(synthetic_processed_df: pd.DataFrame):
    data = HeteroGraphBuilder().build_hetero_data(synthetic_processed_df)
    for node_type in NODE_TYPES:
        assert not torch.isnan(data[node_type].x).any(), f"{node_type} has NaN features"


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


def test_state_save_load_roundtrip(tmp_path: Path, synthetic_processed_df: pd.DataFrame):
    gb = HeteroGraphBuilder()
    gb.build_hetero_data(synthetic_processed_df)
    assert gb._fitted

    state_path = tmp_path / "state.pkl"
    gb.save_state(state_path)
    assert state_path.exists()

    gb2 = HeteroGraphBuilder.load_state(state_path)
    assert gb2._fitted
    assert gb2.state.node_index_maps.keys() == gb.state.node_index_maps.keys()
    assert gb2.state.merchant_kmeans is not None


# ---------------------------------------------------------------------------
# Edge index sanity
# ---------------------------------------------------------------------------


def test_edge_indices_refer_to_valid_nodes(synthetic_processed_df: pd.DataFrame):
    data = HeteroGraphBuilder().build_hetero_data(synthetic_processed_df)
    for src_type, _, dst_type in EDGE_TYPES:
        edge_index = data[src_type, _, dst_type].edge_index
        if edge_index.numel() == 0:
            continue
        assert edge_index[0].max().item() < data[src_type].num_nodes
        assert edge_index[1].max().item() < data[dst_type].num_nodes


def test_shared_address_edges_symmetric(synthetic_processed_df: pd.DataFrame):
    """We emit both directions of card-card shared edges."""
    data = HeteroGraphBuilder().build_hetero_data(synthetic_processed_df)
    edge_index = data["card", "shared_address", "card"].edge_index
    if edge_index.numel() == 0:
        return
    # Flipping src/dst should still be in the edge set.
    pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True))
    for a, b in pairs:
        assert (b, a) in pairs, f"reverse edge ({b}, {a}) missing"
