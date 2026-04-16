#!/usr/bin/env python
"""Run :class:`IEEECISPreprocessor` over the raw data.

Outputs:
    data/processed/train_processed.parquet  (the full processed frame)
    data/processed/preprocessor.pkl         (fitted state for reuse at serve)
    data/processed/preprocessor.summary.json

Usage::

    python scripts/preprocess.py [--config path/to.yaml] [--nrows 50000]
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import click  # noqa: E402

from fraud_detection.data.preprocessing import IEEECISPreprocessor  # noqa: E402
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
    "--nrows",
    type=int,
    default=None,
    help="Optional row cap for quick iteration.",
)
@click.option(
    "--output-name",
    default="train_processed.parquet",
    help="Filename written under data/processed/.",
)
def main(config_path: str | None, nrows: int | None, output_name: str) -> None:
    cfg = load_config(config_path)
    configure_logging(level=cfg.logging.level, json=cfg.logging.use_json)
    log = get_logger("preprocess")
    cfg.paths.ensure_exist()

    pp = IEEECISPreprocessor(cfg)
    df = pp.load_raw(nrows=nrows)
    log.info("raw_loaded", rows=len(df), cols=df.shape[1])
    processed = pp.fit_transform(df)

    out_parquet = cfg.paths.data_processed / output_name
    processed.to_parquet(out_parquet, index=False)
    log.info("processed_written", path=str(out_parquet), rows=len(processed))

    state_path = cfg.paths.data_processed / "preprocessor.pkl"
    pp.save(state_path)


if __name__ == "__main__":
    main()
