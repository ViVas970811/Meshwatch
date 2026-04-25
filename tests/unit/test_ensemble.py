"""Tests for ``fraud_detection.models.ensemble.FraudEnsemble``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData

from fraud_detection.models import FraudEnsemble, FraudHeteroGNN, XGBoostConfig, XGBoostFraudModel


def _toy_data(n_tx: int = 80, seed: int = 0) -> HeteroData:
    g = torch.Generator().manual_seed(seed)
    data = HeteroData()
    tx = torch.randn(n_tx, 4, generator=g)
    data["transaction"].x = tx
    data["transaction"].y = (tx[:, 0] > 0).int()
    data["card"].x = torch.randn(20, 3, generator=g)

    src = torch.arange(n_tx)
    dst = torch.randint(0, 20, (n_tx,), generator=g)
    data["transaction", "uses_card", "card"].edge_index = torch.stack([src, dst])

    src = torch.randint(0, 20, (15,), generator=g)
    dst = torch.randint(0, 20, (15,), generator=g)
    data["card", "shared_address", "card"].edge_index = torch.stack([src, dst])
    return data


def _toy_gnn(data: HeteroData) -> FraudHeteroGNN:
    return FraudHeteroGNN(
        node_feature_dims={nt: data[nt].num_node_features for nt in data.node_types},
        edge_types=data.edge_types,
        hidden_dim=16,
        embedding_dim=8,
        n_layers=2,
        heads=2,
    )


def test_ensemble_fit_predict_runs():
    data = _toy_data()
    gnn = _toy_gnn(data)
    ensemble = FraudEnsemble(
        gnn=gnn,
        xgboost_model=XGBoostFraudModel(
            XGBoostConfig(n_estimators=20, max_depth=3, early_stopping_rounds=None)
        ),
    )
    n = data["transaction"].num_nodes
    train_idx = torch.arange(0, n // 2)
    val_idx = torch.arange(n // 2, n)

    tab = np.random.default_rng(0).normal(size=(n, 5)).astype(np.float32)
    y = data["transaction"].y.numpy().astype(np.int8)
    cols = [f"feat_{i}" for i in range(5)]

    ensemble.fit_xgboost(
        train_data=data,
        train_indices=train_idx,
        train_tabular=tab[train_idx.numpy()],
        train_y=y[train_idx.numpy()],
        tabular_columns=cols,
        val_data=data,
        val_indices=val_idx,
        val_tabular=tab[val_idx.numpy()],
        val_y=y[val_idx.numpy()],
    )
    proba = ensemble.predict_proba(data, tab[val_idx.numpy()], target_indices=val_idx)
    assert proba.shape == (len(val_idx),)
    assert proba.min() >= 0
    assert proba.max() <= 1


def test_feature_columns_layout():
    """Embedding columns first, tabular columns second."""
    data = _toy_data()
    gnn = _toy_gnn(data)
    ensemble = FraudEnsemble(
        gnn=gnn,
        xgboost_model=XGBoostFraudModel(XGBoostConfig(n_estimators=10, early_stopping_rounds=None)),
    )
    n = data["transaction"].num_nodes
    train_idx = torch.arange(0, n)
    tab = np.random.default_rng(0).normal(size=(n, 4)).astype(np.float32)
    y = data["transaction"].y.numpy().astype(np.int8)
    cols = ["feat_a", "feat_b", "feat_c", "feat_d"]
    ensemble.fit_xgboost(
        train_data=data,
        train_indices=train_idx,
        train_tabular=tab,
        train_y=y,
        tabular_columns=cols,
    )
    # GNN embedding columns are gnn_emb_000..gnn_emb_007 (embedding_dim=8)
    assert ensemble.gnn_embedding_columns == [f"gnn_emb_{i:03d}" for i in range(8)]
    assert ensemble.feature_columns == ensemble.gnn_embedding_columns + cols


def test_save_load_roundtrip(tmp_path: Path):
    data = _toy_data()
    gnn = _toy_gnn(data)
    ensemble = FraudEnsemble(
        gnn=gnn,
        xgboost_model=XGBoostFraudModel(XGBoostConfig(n_estimators=10, early_stopping_rounds=None)),
    )
    n = data["transaction"].num_nodes
    idx = torch.arange(n)
    tab = np.random.default_rng(0).normal(size=(n, 3)).astype(np.float32)
    y = data["transaction"].y.numpy().astype(np.int8)
    ensemble.fit_xgboost(
        train_data=data,
        train_indices=idx,
        train_tabular=tab,
        train_y=y,
        tabular_columns=["a", "b", "c"],
    )
    init_kwargs = {
        "node_feature_dims": {nt: data[nt].num_node_features for nt in data.node_types},
        "edge_types": list(data.edge_types),
        "hidden_dim": 16,
        "embedding_dim": 8,
        "n_layers": 2,
        "heads": 2,
    }
    ensemble.save(tmp_path / "ens", gnn_init_kwargs=init_kwargs)

    loaded = FraudEnsemble.load(tmp_path / "ens")
    p1 = ensemble.predict_proba(data, tab, target_indices=idx)
    p2 = loaded.predict_proba(data, tab, target_indices=idx)
    np.testing.assert_allclose(p1, p2, atol=1e-6)
