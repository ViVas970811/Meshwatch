#!/usr/bin/env python
"""Replay transactions to the fraud-detection API for demo / load testing.

Reads rows from ``data/graphs/features.parquet`` (or any IEEE-CIS-shaped
parquet) and POSTs them to ``/api/v1/predict`` at a configurable rate.
Useful for:

* Driving a populated dashboard during a demo.
* Producing latency stats (P50/P95/P99) under realistic load.
* Stress-testing the alert pipeline.

Examples::

    # Send 200 transactions at 5 RPS to localhost:
    python scripts/demo_stream.py --n 200 --rps 5

    # Stress test: 2000 txns, 50 RPS, batch endpoint:
    python scripts/demo_stream.py --n 2000 --rps 50 --batch 50

    # Hit a remote API:
    python scripts/demo_stream.py --url http://meshwatch.dev/api/v1/predict
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

from fraud_detection.utils.config import load_config  # noqa: E402
from fraud_detection.utils.logging import configure_logging, get_logger  # noqa: E402


def _row_to_request(row: pd.Series) -> dict:
    """Map a parquet row into a TransactionRequest payload.

    Forwards two layers:
    * The few well-known fields the API expects by name
      (transaction_id / dt / amt / product_cd / card / addr / dist).
    * Any other engineered ``feat_*`` columns the row carries. The
      TransactionRequest schema allows extras (``extra="allow"``) and
      the predictor maps them by name into the trained XGBoost feature
      matrix -- so if the source parquet is ``data/graphs/features.parquet``
      the request carries all 119 engineered features and the model
      actually scores against them.
    """
    # When the source parquet is features.parquet, the raw TransactionAmt has
    # been replaced by engineered feat_log_amt / feat_amt_cum_* columns. Fall
    # back to a synthesized log-uniform amount in [$5, $5000] so the alert
    # feed and downstream UI show realistic dollar values. The amount is NOT
    # a model feature on this pipeline (XGBoost uses feat_* only), so this
    # is purely cosmetic.
    raw_amt = row.get("TransactionAmt")
    if raw_amt is None or (isinstance(raw_amt, float) and np.isnan(raw_amt)):
        # Stable per-row pseudo-random so the same row always gets the same amount.
        tx_id = int(row.get("TransactionID", 0)) or 1
        rng = np.random.default_rng(tx_id)
        amt = float(np.exp(rng.uniform(np.log(5.0), np.log(5000.0))))
    else:
        amt = float(raw_amt)
    out: dict = {
        "transaction_id": int(row.get("TransactionID", 0)),
        "transaction_dt": int(row.get("TransactionDT", 0)),
        "transaction_amt": round(amt, 2),
        "product_cd": "W",
    }
    # Optional well-known fields.
    for col, dest in [
        ("card1", "card1"),
        ("card2", "card2"),
        ("card3", "card3"),
        ("card5", "card5"),
        ("addr1", "addr1"),
        ("addr2", "addr2"),
        ("dist1", "dist1"),
        ("dist2", "dist2"),
    ]:
        v = row.get(col)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            out[dest] = float(v) if dest != "card1" else int(v)

    # Forward every other numeric column verbatim (feat_*, V1..V339,
    # D1..D15, C1..C14, id_01..id_38, etc.). The predictor's
    # _build_tabular_row maps these by column name; unknown keys are
    # silently dropped, so passing extras is safe.
    skip = {"TransactionID", "TransactionDT", "TransactionAmt", "isFraud", "event_timestamp"}
    for col, v in row.items():
        if col in out or col in skip:
            continue
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        try:
            out[col] = float(v)
        except (TypeError, ValueError):
            # Non-numeric values (strings, dates) -- the predictor expects
            # everything pre-encoded, so we drop these silently.
            continue
    return out


@click.command()
@click.option(
    "--url",
    default="http://127.0.0.1:8000/api/v1/predict",
    show_default=True,
    help="Predict endpoint (single tx). Use --batch to switch to batch endpoint.",
)
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Parquet to draw rows from (default: data/graphs/features.parquet).",
)
@click.option("--n", default=100, show_default=True, type=int, help="Total tx to send.")
@click.option("--rps", default=10.0, show_default=True, type=float, help="Target RPS.")
@click.option(
    "--batch",
    default=0,
    show_default=True,
    type=int,
    help="If >0, switch to /predict/batch with this batch size.",
)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option(
    "--summary-only",
    is_flag=True,
    help="Suppress per-request logging; only print latency summary.",
)
def main(
    url: str,
    input_path: str | None,
    n: int,
    rps: float,
    batch: int,
    seed: int,
    summary_only: bool,
) -> None:
    cfg = load_config()
    configure_logging(level=cfg.logging.level, json=cfg.logging.use_json)
    log = get_logger("demo_stream")

    import requests

    parquet = Path(input_path) if input_path else cfg.paths.data_graphs / "features.parquet"
    if not parquet.exists():
        msg = f"Input parquet not found: {parquet}. Run `make build-graph` first."
        raise FileNotFoundError(msg)

    df = pd.read_parquet(parquet)
    rng = np.random.default_rng(seed)
    if len(df) > n:
        df = df.iloc[rng.choice(len(df), size=n, replace=False)]
    df = df.reset_index(drop=True)

    if batch > 0:
        url = url.rstrip("/").replace("/api/v1/predict", "/api/v1/predict/batch")
        log.info("demo_stream_start_batch", url=url, n=len(df), batch=batch, rps=rps)
    else:
        log.info("demo_stream_start_single", url=url, n=len(df), rps=rps)

    delay = 1.0 / rps if rps > 0 else 0.0
    latencies_ms: list[float] = []
    n_alerts = 0
    n_errors = 0

    sent = 0
    while sent < len(df):
        chunk = df.iloc[sent : sent + (batch or 1)]
        sent += len(chunk)
        payload: dict
        if batch > 0:
            payload = {"transactions": [_row_to_request(r) for _, r in chunk.iterrows()]}
        else:
            payload = _row_to_request(chunk.iloc[0])

        t0 = time.perf_counter()
        try:
            r = requests.post(url, json=payload, timeout=10.0)
            elapsed = (time.perf_counter() - t0) * 1000
            latencies_ms.append(elapsed)
            if r.status_code != 200:
                n_errors += 1
                if not summary_only:
                    log.warning("predict_error", status=r.status_code, body=r.text[:200])
            else:
                body = r.json()
                if batch > 0:
                    n_alerts += int(body.get("n_alerts", 0))
                    if not summary_only:
                        log.info(
                            "batch_response",
                            n_processed=body.get("n_processed"),
                            n_alerts=body.get("n_alerts"),
                            elapsed_ms=round(elapsed, 2),
                        )
                else:
                    if body.get("is_fraud_predicted"):
                        n_alerts += 1
                    if not summary_only:
                        log.info(
                            "predict_response",
                            tx=body.get("transaction_id"),
                            score=round(body.get("fraud_score", 0.0), 4),
                            risk=body.get("risk_level"),
                            elapsed_ms=round(elapsed, 2),
                        )
        except Exception as exc:
            n_errors += 1
            if not summary_only:
                log.warning("predict_request_failed", error=str(exc))

        if delay > 0 and sent < len(df):
            time.sleep(delay)

    if latencies_ms:
        arr = np.array(latencies_ms)
        log.info(
            "demo_stream_complete",
            n_sent=len(df),
            n_alerts=n_alerts,
            n_errors=n_errors,
            p50_ms=round(float(np.percentile(arr, 50)), 2),
            p95_ms=round(float(np.percentile(arr, 95)), 2),
            p99_ms=round(float(np.percentile(arr, 99)), 2),
            mean_ms=round(float(arr.mean()), 2),
            max_ms=round(float(arr.max()), 2),
        )
    else:
        log.warning("demo_stream_complete_no_responses")


if __name__ == "__main__":
    main()
