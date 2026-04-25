#!/usr/bin/env python
"""Stage 2: train the GNN+XGBoost ensemble using the trained GNN.

Inputs:
    data/graphs/hetero.pt               (Phase 2)
    data/graphs/features.parquet        (Phase 2 -- 119 engineered features)
    data/models/gnn/state_dict.pt       (Phase 3 stage 1 GNN weights)

Outputs:
    data/models/ensemble/gnn.pt
    data/models/ensemble/xgb.pkl
    data/models/ensemble/artifacts.pkl
    data/models/ensemble/eval_*.png/json

Usage:
    python scripts/train_ensemble.py [--device cpu]
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import click  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import HeteroData  # noqa: E402
from torch_geometric.transforms import ToUndirected  # noqa: E402

from fraud_detection.models import FraudEnsemble, FraudHeteroGNN, XGBoostFraudModel  # noqa: E402
from fraud_detection.training import (  # noqa: E402
    ensure_temporal_masks,
    evaluate_predictions,
    write_evaluation_report,
)
from fraud_detection.utils.config import load_config  # noqa: E402
from fraud_detection.utils.logging import configure_logging, get_logger  # noqa: E402


def _slice_tabular(features: pd.DataFrame, mask_idx: torch.Tensor) -> tuple[np.ndarray, list[str]]:
    """Pull rows ``mask_idx`` from features.parquet and return as float32 ndarray."""
    df = features.iloc[mask_idx.cpu().numpy()].reset_index(drop=True)
    drop_cols = [c for c in ("TransactionID", "isFraud", "event_timestamp") if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df.to_numpy(dtype=np.float32), list(df.columns)


@click.command()
@click.option(
    "--gnn-state",
    type=click.Path(exists=True, dir_okay=False),
    default="data/models/gnn/state_dict.pt",
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="data/models/ensemble",
    show_default=True,
)
@click.option("--device", type=str, default="cpu", show_default=True)
def main(gnn_state: str, output_dir: str, device: str) -> None:
    cfg = load_config()
    configure_logging(level=cfg.logging.level, json=cfg.logging.use_json)
    log = get_logger("train_ensemble")

    # ----- Load Phase 2 + Phase 3 stage 1 artefacts -----
    graph_path = cfg.paths.data_graphs / "hetero.pt"
    features_path = cfg.paths.data_graphs / "features.parquet"
    if not graph_path.exists():
        msg = f"{graph_path} missing -- run `make build-graph` first."
        raise FileNotFoundError(msg)
    if not features_path.exists():
        msg = f"{features_path} missing -- run `make build-graph` first."
        raise FileNotFoundError(msg)

    data: HeteroData = torch.load(graph_path, weights_only=False)
    for nt in data.node_types:
        if hasattr(data[nt], "x") and data[nt].x is not None:
            data[nt].x = data[nt].x.float()
    data = ToUndirected()(data)
    data = ensure_temporal_masks(data, target_node_type="transaction")
    features = pd.read_parquet(features_path)
    log.info(
        "artifacts_loaded", graph_n_tx=data["transaction"].num_nodes, n_features=features.shape[1]
    )

    # Build GNN with the same schema and load state dict.
    node_feature_dims = {nt: data[nt].num_node_features for nt in data.node_types}
    gnn = FraudHeteroGNN(node_feature_dims=node_feature_dims, edge_types=data.edge_types)
    gnn.load_state_dict(torch.load(gnn_state, weights_only=True))
    gnn.eval()
    log.info("gnn_loaded", state_dict=gnn_state, n_params=gnn.n_parameters())

    # ----- Slice splits -----
    tx = data["transaction"]
    train_idx = tx.train_mask.nonzero(as_tuple=True)[0]
    val_idx = tx.val_mask.nonzero(as_tuple=True)[0]
    test_idx = tx.test_mask.nonzero(as_tuple=True)[0]

    y_train = tx.y[train_idx].numpy().astype(np.int8)
    y_val = tx.y[val_idx].numpy().astype(np.int8)
    y_test = tx.y[test_idx].numpy().astype(np.int8)

    X_train_tab, tab_cols = _slice_tabular(features, train_idx)
    X_val_tab, _ = _slice_tabular(features, val_idx)
    X_test_tab, _ = _slice_tabular(features, test_idx)
    log.info("tabular_columns", n=len(tab_cols))

    # ----- Train ensemble -----
    ensemble = FraudEnsemble(gnn=gnn, xgboost_model=XGBoostFraudModel())
    ensemble.fit_xgboost(
        train_data=data,
        train_indices=train_idx,
        train_tabular=X_train_tab,
        train_y=y_train,
        tabular_columns=tab_cols,
        val_data=data,
        val_indices=val_idx,
        val_tabular=X_val_tab,
        val_y=y_val,
        device=device,
    )

    # ----- Score val + test -----
    val_proba = ensemble.predict_proba(data, X_val_tab, target_indices=val_idx, device=device)
    test_proba = ensemble.predict_proba(data, X_test_tab, target_indices=test_idx, device=device)

    val_result = evaluate_predictions(y_val, val_proba)
    test_result = evaluate_predictions(y_test, test_proba)
    log.info(
        "ensemble_val_metrics",
        **{k: v for k, v in val_result.to_dict().items() if not isinstance(v, dict)},
    )
    log.info(
        "ensemble_test_metrics",
        **{k: v for k, v in test_result.to_dict().items() if not isinstance(v, dict)},
    )

    # ----- Persist -----
    out_dir = Path(output_dir)
    ensemble.save(
        out_dir,
        gnn_init_kwargs={
            "node_feature_dims": node_feature_dims,
            "edge_types": list(data.edge_types),
        },
    )
    eval_dir = out_dir / "eval"
    write_evaluation_report(y_val, val_proba, output_dir=eval_dir, name="val")
    write_evaluation_report(y_test, test_proba, output_dir=eval_dir, name="test")

    # Phase 3 acceptance.
    if val_result.auprc < 0.70:
        log.warning(
            "ensemble_val_auprc_below_acceptance",
            val_auprc=val_result.auprc,
            threshold=0.70,
        )
    else:
        log.info("ensemble_val_auprc_meets_acceptance", val_auprc=val_result.auprc)


if __name__ == "__main__":
    main()
