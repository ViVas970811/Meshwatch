#!/usr/bin/env python
"""Train the heterogeneous fraud-detection GNN on the Phase 2 graph artefact.

Inputs:
    data/graphs/hetero.pt           -- HeteroData written by build_graph.py
    data/graphs/features.parquet    -- 119 engineered features (used in stage 2)

Outputs:
    data/models/gnn/state_dict.pt   -- trained model weights
    data/models/gnn/history.json    -- per-epoch metrics
    data/models/gnn/eval_*.png/json -- val + test report
    mlruns/                          -- MLflow tracking (if mlflow available)

Usage:
    python scripts/train.py [--epochs 100] [--batch-size 4096] [--device cpu]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import click  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import HeteroData  # noqa: E402
from torch_geometric.transforms import ToUndirected  # noqa: E402

from fraud_detection.models import FraudHeteroGNN  # noqa: E402
from fraud_detection.training import (  # noqa: E402
    Trainer,
    TrainerConfig,
    ensure_temporal_masks,
    write_evaluation_report,
)
from fraud_detection.utils.config import load_config  # noqa: E402
from fraud_detection.utils.logging import configure_logging, get_logger  # noqa: E402


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to a YAML config (defaults to configs/base.yaml).",
)
@click.option("--epochs", type=int, default=100, show_default=True)
@click.option("--batch-size", type=int, default=4096, show_default=True)
@click.option("--lr", type=float, default=1e-3, show_default=True)
@click.option("--patience", type=int, default=15, show_default=True)
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option("--no-mlflow", is_flag=True, help="Disable MLflow tracking")
@click.option(
    "--output-dir",
    type=click.Path(),
    default="data/models/gnn",
    show_default=True,
    help="Where to write model + eval artifacts.",
)
def main(
    config_path: str | None,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    device: str,
    num_workers: int,
    no_mlflow: bool,
    output_dir: str,
) -> None:
    cfg = load_config(config_path)
    configure_logging(level=cfg.logging.level, json=cfg.logging.use_json)
    log = get_logger("train")
    cfg.paths.ensure_exist()

    graph_path = cfg.paths.data_graphs / "hetero.pt"
    if not graph_path.exists():
        msg = f"Graph artifact not found at {graph_path}. Run `make build-graph` first."
        raise FileNotFoundError(msg)

    log.info("loading_hetero_graph", path=str(graph_path))
    data: HeteroData = torch.load(graph_path, weights_only=False)

    # Ensure all node features are float32 (PyG/Torch convention).
    for nt in data.node_types:
        if hasattr(data[nt], "x") and data[nt].x is not None:
            data[nt].x = data[nt].x.float()

    # Add reverse edges so message passing is bidirectional.
    log.info("adding_reverse_edges")
    data = ToUndirected()(data)

    # Build temporal splits if missing -- the Phase 2 builder may emit
    # all-True masks; we recompute by sorting the seed transactions by
    # their index (already temporally ordered upstream).
    data = ensure_temporal_masks(data, target_node_type="transaction")
    train_n = int(data["transaction"].train_mask.sum())
    val_n = int(data["transaction"].val_mask.sum())
    test_n = int(data["transaction"].test_mask.sum())
    log.info("split_summary", train=train_n, val=val_n, test=test_n)

    # Build model.
    node_feature_dims = {nt: data[nt].num_node_features for nt in data.node_types}
    model = FraudHeteroGNN(
        node_feature_dims=node_feature_dims,
        edge_types=data.edge_types,
    )
    log.info("model_built", n_params=model.n_parameters())

    # Train.
    tcfg = TrainerConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        early_stop_patience=patience,
        device=device,
        num_workers=num_workers,
        mlflow_enabled=not no_mlflow,
        mlflow_extra_params={"phase": "3", "tag": "v0.3.0-gnn-model"},
    )
    trainer = Trainer(model, tcfg)
    result = trainer.fit(data)

    # Persist model + history + final eval reports.
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state_dict.pt"
    torch.save(result["model"].state_dict(), state_path)
    log.info("state_dict_saved", path=str(state_path))

    history_path = out_dir / "history.json"
    history_path.write_text(json.dumps(result["history"], indent=2), encoding="utf-8")

    # Final eval reports on val + test.
    test_idx = data["transaction"].test_mask.nonzero(as_tuple=True)[0]
    test_result, y_test, p_test = trainer._evaluate_split(data, test_idx)
    log.info(
        "test_metrics",
        **{k: v for k, v in test_result.to_dict().items() if not isinstance(v, dict)},
    )

    eval_dir = out_dir / "eval"
    val_idx = data["transaction"].val_mask.nonzero(as_tuple=True)[0]
    val_result, y_val, p_val = trainer._evaluate_split(data, val_idx)
    write_evaluation_report(y_val, p_val, output_dir=eval_dir, name="val")
    write_evaluation_report(y_test, p_test, output_dir=eval_dir, name="test")

    # Phase 3 acceptance check.
    if val_result.auprc < 0.65:
        log.warning(
            "val_auprc_below_acceptance",
            val_auprc=val_result.auprc,
            threshold=0.65,
            note="Plan acceptance criterion is val AUPRC > 0.65 for GNN alone. "
            "Continue to ensemble stage; sometimes the ensemble gets there even when GNN-only doesn't.",
        )
    else:
        log.info("val_auprc_meets_acceptance", val_auprc=val_result.auprc)


if __name__ == "__main__":
    main()
