#!/usr/bin/env python
"""Evaluate a trained model (GNN-only or ensemble) on a held-out split.

Usage:
    python scripts/evaluate.py --model-dir data/models/gnn       --split test
    python scripts/evaluate.py --model-dir data/models/ensemble  --split test
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

from fraud_detection.models import FraudEnsemble, FraudHeteroGNN  # noqa: E402
from fraud_detection.training import (  # noqa: E402
    ensure_temporal_masks,
    write_evaluation_report,
)
from fraud_detection.utils.config import load_config  # noqa: E402
from fraud_detection.utils.logging import configure_logging, get_logger  # noqa: E402


@click.command()
@click.option(
    "--model-dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Directory with state_dict.pt (GNN-only) or gnn.pt+xgb.pkl+artifacts.pkl (ensemble).",
)
@click.option(
    "--split",
    type=click.Choice(["val", "test"]),
    default="test",
    show_default=True,
)
@click.option("--device", type=str, default="cpu", show_default=True)
def main(model_dir: str, split: str, device: str) -> None:
    cfg = load_config()
    configure_logging(level=cfg.logging.level, json=cfg.logging.use_json)
    log = get_logger("evaluate")

    model_dir = Path(model_dir)
    is_ensemble = (model_dir / "xgb.pkl").exists()

    graph_path = cfg.paths.data_graphs / "hetero.pt"
    data: HeteroData = torch.load(graph_path, weights_only=False)
    for nt in data.node_types:
        if hasattr(data[nt], "x") and data[nt].x is not None:
            data[nt].x = data[nt].x.float()
    data = ToUndirected()(data)
    data = ensure_temporal_masks(data, target_node_type="transaction")

    tx = data["transaction"]
    mask_idx = (tx.val_mask if split == "val" else tx.test_mask).nonzero(as_tuple=True)[0]
    y = tx.y[mask_idx].numpy().astype(np.int8)

    if is_ensemble:
        ensemble = FraudEnsemble.load(model_dir)
        features = pd.read_parquet(cfg.paths.data_graphs / "features.parquet")
        df = features.iloc[mask_idx.cpu().numpy()].reset_index(drop=True)
        drop_cols = [c for c in ("TransactionID", "isFraud", "event_timestamp") if c in df.columns]
        X_tab = df.drop(columns=drop_cols).to_numpy(dtype=np.float32)
        proba = ensemble.predict_proba(data, X_tab, target_indices=mask_idx, device=device)
        eval_name = f"{split}_ensemble"
    else:
        node_feature_dims = {nt: data[nt].num_node_features for nt in data.node_types}
        gnn = FraudHeteroGNN(node_feature_dims=node_feature_dims, edge_types=data.edge_types)
        gnn.load_state_dict(torch.load(model_dir / "state_dict.pt", weights_only=True))
        gnn.eval()
        gnn.to(device)
        data_dev = data.to(device)
        with torch.no_grad():
            logits = gnn(data_dev)
        proba_all = torch.sigmoid(logits).cpu().numpy()
        proba = proba_all[mask_idx.cpu().numpy()]
        eval_name = f"{split}_gnn"

    eval_dir = model_dir / "eval"
    paths = write_evaluation_report(y, proba, output_dir=eval_dir, name=eval_name)
    log.info(
        "evaluation_done",
        split=split,
        ensemble=is_ensemble,
        paths={k: str(v) for k, v in paths.items()},
    )


if __name__ == "__main__":
    main()
