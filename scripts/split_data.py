#!/usr/bin/env python
"""Create temporal train/val/test splits from processed data.

Inputs:
    data/processed/train_processed.parquet

Outputs:
    data/splits/train.parquet
    data/splits/val.parquet
    data/splits/test.parquet
    data/splits/split_summary.json

Usage::

    python scripts/split_data.py [--config path/to.yaml]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import click  # noqa: E402
import pandas as pd  # noqa: E402

from fraud_detection.data.splits import TemporalSplitter  # noqa: E402
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
@click.option(
    "--input-name",
    default="train_processed.parquet",
    help="Filename under data/processed/.",
)
def main(config_path: str | None, input_name: str) -> None:
    cfg = load_config(config_path)
    configure_logging(level=cfg.logging.level, json=cfg.logging.use_json)
    log = get_logger("split")
    cfg.paths.ensure_exist()

    in_path = cfg.paths.data_processed / input_name
    if not in_path.exists():
        msg = f"Processed file not found: {in_path}. Run `make preprocess` first."
        raise FileNotFoundError(msg)

    df = pd.read_parquet(in_path)
    log.info("processed_loaded", path=str(in_path), rows=len(df))

    splitter = TemporalSplitter(cfg)
    result = splitter.split(df)
    paths = result.save_parquet(cfg.paths.data_splits)

    summary = result.summary()
    summary["paths"] = {k: str(v) for k, v in paths.items()}
    summary_path = cfg.paths.data_splits / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log.info("split_summary_written", path=str(summary_path), **summary)


if __name__ == "__main__":
    main()
