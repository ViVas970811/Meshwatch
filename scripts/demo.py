#!/usr/bin/env python
"""End-to-end demo of the Meshwatch stack (Phase 8).

This is the one-command tour from the plan (page 13):

    1. Verify the API is up.
    2. Replay 1000 transactions through /api/v1/predict.
    3. Surface latency stats (P50/P95/P99).
    4. Pick the most-fraudulent alert and trigger the LangGraph
       investigator via /api/v1/investigate.
    5. Print the structured investigation report.
    6. Snapshot the live monitoring surface (drift, performance,
       active alerts).

Designed so the *only* thing the operator does is::

    docker compose up -d        # starts API + Redis + Kafka + Grafana
    python scripts/demo.py      # runs the demo end-to-end

When no IEEE-CIS parquet is available the demo synthesises a small
distribution that exercises every code path (a mix of low-risk and
fraud-flavoured rows). The synthetic mode keeps the demo runnable on a
freshly cloned repo with no preprocessing.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import click  # noqa: E402
import numpy as np  # noqa: E402

from fraud_detection.utils.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("demo")


# ---------------------------------------------------------------------------
# Console helpers (no external rich/colorama dependency)
# ---------------------------------------------------------------------------


def _hr(title: str = "") -> None:
    bar = "=" * 78
    if title:
        click.echo(f"\n{bar}\n  {title}\n{bar}")
    else:
        click.echo(bar)


def _kv(label: str, value: object, *, indent: int = 2) -> None:
    pad = " " * indent
    click.echo(f"{pad}{label:<24}{value}")


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------


def _synthetic_transactions(n: int, *, seed: int = 42) -> list[dict]:
    """A small deterministic distribution that exercises every risk bucket.

    Roughly 60% low-amount low-risk, 25% medium amounts, 10% large amounts
    that often score MEDIUM/HIGH, and 5% obviously-fraudulent (huge
    amounts on shared cards with unusual product codes).
    """
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for i in range(n):
        kind = rng.choice(
            ["low", "medium", "high", "critical"],
            p=[0.60, 0.25, 0.10, 0.05],
        )
        if kind == "low":
            amt = float(rng.normal(40, 15).clip(2, 200))
            card1 = int(rng.integers(1000, 19000))
        elif kind == "medium":
            amt = float(rng.normal(180, 50).clip(50, 600))
            card1 = int(rng.integers(1000, 19000))
        elif kind == "high":
            amt = float(rng.normal(900, 200).clip(300, 2500))
            card1 = int(rng.integers(1000, 19000))
        else:  # critical
            amt = float(rng.normal(4500, 800).clip(2000, 9999))
            card1 = int(rng.choice([1234, 4242, 8675]))  # repeat hot cards
        out.append(
            {
                "transaction_id": f"demo-{i:04d}",
                "transaction_dt": int(time.time()) + i,
                "transaction_amt": round(amt, 2),
                "product_cd": rng.choice(["W", "C", "H", "S", "R"]).item(),
                "card1": card1,
            }
        )
    return out


def _load_parquet_rows(path: Path, n: int, seed: int) -> list[dict] | None:
    import pandas as pd

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        log.warning("demo_parquet_load_failed", path=str(path), error=str(exc))
        return None
    rng = np.random.default_rng(seed)
    if len(df) > n:
        df = df.iloc[rng.choice(len(df), size=n, replace=False)]
    df = df.reset_index(drop=True)

    rows: list[dict] = []
    for _, r in df.iterrows():
        row = {
            "transaction_id": str(r.get("TransactionID", _)),
            "transaction_dt": int(r.get("TransactionDT", 0)),
            "transaction_amt": float(r.get("TransactionAmt", 0.0)),
            "product_cd": str(r.get("ProductCD", "W")),
        }
        card = r.get("card1")
        if card is not None and not (isinstance(card, float) and np.isnan(card)):
            row["card1"] = int(card)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# API client wrappers
# ---------------------------------------------------------------------------


def _headers(api_key: str | None) -> dict[str, str]:
    return {"X-API-Key": api_key} if api_key else {}


def _wait_for_health(base_url: str, *, api_key: str | None, timeout_s: float = 30.0) -> dict:
    """Poll /api/v1/health until it returns 200 or we time out."""
    import requests

    deadline = time.time() + timeout_s
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/api/v1/health", timeout=5.0, headers=_headers(api_key))
            if r.status_code == 200:
                return r.json()
        except Exception as exc:
            last_exc = exc
        time.sleep(1.0)
    msg = f"API at {base_url} did not become healthy in {timeout_s}s"
    if last_exc is not None:
        msg += f" (last error: {last_exc!s})"
    raise click.ClickException(msg)


def _replay(
    base_url: str,
    *,
    rows: list[dict],
    rps: float,
    api_key: str | None,
) -> tuple[list[dict], list[float], int, int]:
    """Send each row to /api/v1/predict at ``rps``. Returns (predictions, latencies, alerts, errors)."""
    import requests

    delay = 1.0 / rps if rps > 0 else 0.0
    latencies_ms: list[float] = []
    predictions: list[dict] = []
    n_alerts = 0
    n_errors = 0

    headers = _headers(api_key)
    url = f"{base_url}/api/v1/predict"
    for i, payload in enumerate(rows):
        t0 = time.perf_counter()
        try:
            r = requests.post(url, json=payload, timeout=10.0, headers=headers)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
            if r.status_code == 200:
                body = r.json()
                predictions.append(body)
                if body.get("is_fraud_predicted"):
                    n_alerts += 1
            else:
                n_errors += 1
                if n_errors <= 3:  # show a few, then stay quiet
                    log.warning(
                        "demo_predict_error",
                        status=r.status_code,
                        body=r.text[:200],
                    )
        except Exception as exc:
            n_errors += 1
            log.warning("demo_predict_request_failed", error=str(exc))

        if delay > 0 and i + 1 < len(rows):
            time.sleep(delay)
    return predictions, latencies_ms, n_alerts, n_errors


def _trigger_investigation(base_url: str, *, prediction: dict, api_key: str | None) -> dict | None:
    import requests

    payload = {"prediction": prediction, "alert_id": f"demo-{prediction['transaction_id']}"}
    try:
        r = requests.post(
            f"{base_url}/api/v1/investigate",
            json=payload,
            timeout=60.0,
            headers=_headers(api_key),
        )
    except Exception as exc:
        log.warning("demo_investigate_request_failed", error=str(exc))
        return None
    if r.status_code != 200:
        log.warning("demo_investigate_unavailable", status=r.status_code, body=r.text[:200])
        return None
    return r.json()


def _fetch_monitoring(base_url: str, *, api_key: str | None) -> dict:
    import requests

    out: dict = {}
    headers = _headers(api_key)
    for name, path in (
        ("performance", "/api/v1/monitoring/performance"),
        ("drift", "/api/v1/monitoring/drift"),
        ("alerts", "/api/v1/monitoring/alerts"),
        ("shadow", "/api/v1/monitoring/shadow"),
    ):
        try:
            r = requests.get(f"{base_url}{path}", timeout=5.0, headers=headers)
            if r.status_code == 200:
                out[name] = r.json()
        except Exception as exc:
            log.warning("demo_monitoring_unreachable", path=path, error=str(exc))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--url",
    default="http://127.0.0.1:8000",
    show_default=True,
    help="Base URL of the Meshwatch API.",
)
@click.option("--n", default=1000, show_default=True, type=int, help="Transactions to replay.")
@click.option("--rps", default=50.0, show_default=True, type=float, help="Target requests/sec.")
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional parquet of real transactions (falls back to synthetic).",
)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option(
    "--api-key",
    envvar="FRAUD_API_KEY",
    default=None,
    help="API key, when the server is locked down with FRAUD_API_KEYS.",
)
@click.option(
    "--health-timeout",
    default=30.0,
    show_default=True,
    type=float,
    help="Seconds to wait for /api/v1/health.",
)
@click.option(
    "--skip-investigate",
    is_flag=True,
    help="Skip the agent investigation step (still prints monitoring summary).",
)
def main(
    url: str,
    n: int,
    rps: float,
    input_path: Path | None,
    seed: int,
    api_key: str | None,
    health_timeout: float,
    skip_investigate: bool,
) -> None:
    """Run the canonical Meshwatch end-to-end demo."""
    configure_logging(level="INFO")
    base = url.rstrip("/")
    t_demo_start = time.perf_counter()

    # ---- Step 1: health ----------------------------------------------------
    _hr("1. Health check")
    health = _wait_for_health(base, api_key=api_key, timeout_s=health_timeout)
    _kv("status", health.get("status"))
    _kv("model_loaded", health.get("model_loaded"))
    _kv("redis_connected", health.get("redis_connected"))
    _kv("kafka_connected", health.get("kafka_connected"))
    _kv("ray_serve_active", health.get("ray_serve_active"))
    _kv("uptime_seconds", round(float(health.get("uptime_seconds", 0.0)), 1))

    # ---- Step 2: build the transaction stream ------------------------------
    _hr(f"2. Building {n} transactions")
    rows: list[dict] | None = None
    if input_path is not None:
        rows = _load_parquet_rows(input_path, n=n, seed=seed)
    if rows is None:
        rows = _synthetic_transactions(n, seed=seed)
        click.echo("  Using synthetic distribution (no parquet supplied).")
    else:
        click.echo(f"  Loaded {len(rows)} rows from {input_path}.")

    # ---- Step 3: replay ----------------------------------------------------
    _hr(f"3. Replaying {len(rows)} transactions at {rps:.0f} RPS")
    t0 = time.perf_counter()
    predictions, latencies, n_alerts, n_errors = _replay(base, rows=rows, rps=rps, api_key=api_key)
    elapsed_s = time.perf_counter() - t0
    if not latencies:
        raise click.ClickException("No successful predictions -- aborting demo.")
    arr = np.array(latencies, dtype=np.float64)
    throughput = len(latencies) / max(elapsed_s, 1e-6)

    _kv("n_sent", len(rows))
    _kv("n_responses", len(predictions))
    _kv("n_alerts (>= threshold)", n_alerts)
    _kv("n_errors", n_errors)
    _kv("elapsed_seconds", round(elapsed_s, 2))
    _kv("throughput rps", round(throughput, 1))
    _kv("latency p50_ms", round(float(np.percentile(arr, 50)), 2))
    _kv("latency p95_ms", round(float(np.percentile(arr, 95)), 2))
    _kv("latency p99_ms", round(float(np.percentile(arr, 99)), 2))
    _kv("latency mean_ms", round(float(arr.mean()), 2))
    _kv("latency max_ms", round(float(arr.max()), 2))

    risk_counts = Counter(p.get("risk_level", "?") for p in predictions)
    risk_summary = ", ".join(f"{lvl}: {count}" for lvl, count in sorted(risk_counts.items()))
    _kv("risk distribution", risk_summary)

    # ---- Step 4: pick the most fraudulent alert + run agent ----------------
    if predictions and not skip_investigate:
        _hr("4. Triggering investigator on the highest-scoring prediction")
        ranked = sorted(predictions, key=lambda p: -float(p.get("fraud_score", 0.0)))
        top = ranked[0]
        _kv("transaction_id", top.get("transaction_id"))
        _kv("fraud_score", round(float(top.get("fraud_score", 0.0)), 4))
        _kv("risk_level", top.get("risk_level"))

        report = _trigger_investigation(base, prediction=top, api_key=api_key)
        if report is None:
            click.echo("\n  (agent unavailable -- skipping report)")
        else:
            click.echo()
            click.echo(f"  Summary:     {report.get('summary', '')}")
            click.echo(f"  Action:      {report.get('recommended_action', '')}")
            click.echo(f"  Confidence:  {report.get('confidence', '')}")
            click.echo(f"  Elapsed ms:  {report.get('elapsed_ms', '')}")
            click.echo(f"  Tools used:  {', '.join(report.get('tools_used') or [])}")
            narrative = (report.get("narrative") or "").strip()
            if narrative:
                click.echo("\n  Narrative:")
                for line in narrative.splitlines():
                    click.echo(f"    {line}")
    elif skip_investigate:
        _hr("4. Investigator step skipped (--skip-investigate)")

    # ---- Step 5: monitoring snapshot --------------------------------------
    _hr("5. Monitoring snapshot")
    mon = _fetch_monitoring(base, api_key=api_key)
    perf = mon.get("performance", {})
    drift = mon.get("drift", {})
    alerts = mon.get("alerts", {})
    shadow = mon.get("shadow", {})
    _kv("performance n_total", perf.get("n_total", 0))
    _kv("performance n_labelled", perf.get("n_labelled", 0))
    _kv("performance precision", perf.get("precision", 0))
    _kv("performance recall", perf.get("recall", 0))
    _kv("drift status", drift.get("severity") or drift.get("status") or "n/a")
    _kv("active alerts", alerts.get("n_active", 0))
    if alerts.get("alerts"):
        for a in alerts["alerts"]:
            click.echo(f"    - {a['name']} ({a['severity']}): {a['reason']}")
    if shadow.get("status") != "disabled":
        _kv("shadow n_total", shadow.get("n_total", 0))
        _kv("shadow agreement", shadow.get("agreement_rate", "n/a"))

    _hr(f"Demo complete in {time.perf_counter() - t_demo_start:.1f}s")


if __name__ == "__main__":
    main()
