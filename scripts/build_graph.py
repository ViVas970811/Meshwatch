#!/usr/bin/env python
"""Build the Phase 2 heterogeneous graph + engineered features.

Reads:
    data/processed/train_processed.parquet
    data/splits/{train,val,test}.parquet  (for training_mask)

Writes:
    data/graphs/hetero.pt           -- PyG HeteroData blob
    data/graphs/graph_builder.pkl   -- fitted graph builder state
    data/graphs/features.parquet    -- 119 engineered features + id cols
    data/graphs/feature_pipeline.pkl -- fitted feature pipeline

Usage::

    python scripts/build_graph.py [--config path/to.yaml] [--nrows N]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import click  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from fraud_detection.data.graph_builder import HeteroGraphBuilder  # noqa: E402
from fraud_detection.features.pipeline import FeaturePipeline  # noqa: E402
from fraud_detection.utils.config import load_config  # noqa: E402
from fraud_detection.utils.logging import configure_logging, get_logger  # noqa: E402


def _load_splits(cfg) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load processed data and derive train/val/test boolean masks."""
    processed = pd.read_parquet(cfg.paths.data_processed / "train_processed.parquet")
    splits_dir = cfg.paths.data_splits
    train_ids: set[int] = set()
    val_ids: set[int] = set()
    test_ids: set[int] = set()
    for name, bucket in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        path = splits_dir / f"{name}.parquet"
        if path.exists():
            bucket.update(
                pd.read_parquet(path, columns=["TransactionID"])["TransactionID"].tolist()
            )
    train_mask = processed["TransactionID"].isin(train_ids)
    val_mask = processed["TransactionID"].isin(val_ids)
    test_mask = processed["TransactionID"].isin(test_ids)
    return processed, train_mask, val_mask, test_mask


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to a YAML config (defaults to configs/base.yaml).",
)
@click.option(
    "--nrows",
    type=int,
    default=None,
    help="Cap rows for quick iteration (entire df if None).",
)
@click.option(
    "--skip-features",
    is_flag=True,
    help="Only build the graph; skip the 119 engineered features.",
)
def main(config_path: str | None, nrows: int | None, skip_features: bool) -> None:
    cfg = load_config(config_path)
    configure_logging(level=cfg.logging.level, json=cfg.logging.use_json)
    log = get_logger("build_graph")
    cfg.paths.ensure_exist()

    t0 = time.time()
    df, train_mask, val_mask, test_mask = _load_splits(cfg)
    if nrows is not None:
        df = df.head(nrows)
        train_mask = train_mask.head(nrows)
        val_mask = val_mask.head(nrows)
        test_mask = test_mask.head(nrows)
    log.info(
        "build_graph_input_loaded",
        rows=len(df),
        cols=df.shape[1],
        n_train=int(train_mask.sum()),
        n_val=int(val_mask.sum()),
        n_test=int(test_mask.sum()),
    )

    # --- Heterogeneous graph -----------------------------------------------
    gb = HeteroGraphBuilder(cfg)
    data = gb.build_hetero_data(
        df,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    graph_path = cfg.paths.data_graphs / "hetero.pt"
    torch.save(data, graph_path)
    state_path = cfg.paths.data_graphs / "graph_builder.pkl"
    gb.save_state(state_path)
    log.info(
        "build_graph_hetero_saved",
        path=str(graph_path),
        bytes=graph_path.stat().st_size,
        elapsed=time.time() - t0,
    )

    if skip_features:
        log.info("build_graph_skipping_features")
        return

    # --- 119 engineered features -------------------------------------------
    t1 = time.time()
    pipeline = FeaturePipeline()
    features = pipeline.fit_transform(df, training_mask=train_mask)
    # Add an ``event_timestamp`` column derived from TransactionDT (seconds
    # since an unknown reference date in IEEE-CIS). Feast requires a
    # datetime column on every feature view. We anchor at 2017-01-01 UTC
    # (close to IEEE-CIS's competition window) so the relative ordering is
    # preserved.
    if "TransactionDT" in features.columns:
        features["event_timestamp"] = pd.to_datetime(
            features["TransactionDT"].astype(np.int64), unit="s", origin="2017-01-01"
        )
    features_path = cfg.paths.data_graphs / "features.parquet"
    features.to_parquet(features_path, index=False)
    pipeline_path = cfg.paths.data_graphs / "feature_pipeline.pkl"
    pipeline.save(pipeline_path)
    log.info(
        "build_graph_features_saved",
        path=str(features_path),
        rows=len(features),
        cols=features.shape[1],
        elapsed=time.time() - t1,
    )

    log.info("build_graph_total_elapsed", seconds=time.time() - t0)


if __name__ == "__main__":
    main()
