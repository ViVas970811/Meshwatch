"""CLI for running the fraud-investigation agent on a synthetic alert.

Examples
--------

Quick CRITICAL alert (uses the deterministic stub LLM):

    python scripts/investigate.py --score 0.95 --amount 4210

Same, against a running Ollama daemon:

    OLLAMA_BASE_URL=http://localhost:11434 \\
        python scripts/investigate.py --score 0.95 --amount 4210 --use-ollama

Replay an alert against a live API:

    python scripts/investigate.py --score 0.95 --amount 4210 --api http://localhost:8000
"""

from __future__ import annotations

import json
import sys
from typing import Any

import click

from fraud_detection.agent import (
    AgentDeps,
    StubProvider,
    get_llm,
    investigate,
    new_state,
)
from fraud_detection.serving.schemas import FraudPrediction, risk_level
from fraud_detection.utils.logging import configure_logging, get_logger

log = get_logger(__name__)


@click.command(help="Run the LangGraph fraud investigator on a synthetic alert.")
@click.option("--transaction-id", default="demo-001", help="Synthetic transaction id.")
@click.option("--card", "card_id", default=9999, type=int, help="Synthetic card id.")
@click.option("--amount", default=420.0, type=float, help="Transaction amount in USD.")
@click.option("--score", default=0.85, type=float, help="Fraud score in [0,1].")
@click.option("--use-ollama", is_flag=True, help="Try the real Ollama provider.")
@click.option(
    "--api",
    default=None,
    type=str,
    help="If set, POST the request to a running API instead of running locally.",
)
@click.option("--json-output", "as_json", is_flag=True, help="Print the report as JSON.")
def main(
    transaction_id: str,
    card_id: int,
    amount: float,
    score: float,
    use_ollama: bool,
    api: str | None,
    as_json: bool,
) -> int:
    """Entry point."""
    configure_logging(level="INFO", json=False)

    pred = FraudPrediction(
        transaction_id=transaction_id,
        fraud_probability=score,
        fraud_score=score,
        risk_level=risk_level(score),
        is_fraud_predicted=score >= 0.7,
        threshold=0.7,
    )
    request: dict[str, Any] = {
        "transaction_id": transaction_id,
        "transaction_dt": 1_500_000,
        "transaction_amt": amount,
        "card1": card_id,
        "product_cd": "W",
    }

    if api:
        report = _hit_api(api, request, pred)
    else:
        llm = get_llm(prefer_ollama=use_ollama) if use_ollama else StubProvider()
        deps = AgentDeps(llm=llm)
        state = new_state(
            transaction_id=transaction_id,
            prediction=pred,
            request=request,
        )
        report = investigate(state, deps=deps).model_dump(mode="json")

    if as_json:
        click.echo(json.dumps(report, indent=2, default=str))
    else:
        _print_pretty(report)
    return 0


def _hit_api(base_url: str, request: dict[str, Any], pred: FraudPrediction) -> dict[str, Any]:
    import httpx  # type: ignore[import-not-found]

    payload = {
        "transaction": request,
        "prediction": pred.model_dump(mode="json"),
    }
    log.info("investigate_via_api", url=f"{base_url}/api/v1/investigate")
    resp = httpx.post(f"{base_url.rstrip('/')}/api/v1/investigate", json=payload, timeout=60.0)
    resp.raise_for_status()
    return resp.json()


def _print_pretty(report: dict[str, Any]) -> None:
    click.echo("=" * 60)
    click.echo(f"ALERT  {report.get('alert_id')}  (transaction {report.get('transaction_id')})")
    click.echo(
        f"  risk_level={report.get('risk_level')}  "
        f"depth={report.get('depth')}  "
        f"score={report.get('fraud_score'):.3f}"
    )
    click.echo(
        f"  recommended_action={report.get('recommended_action').upper()}  "
        f"confidence={report.get('confidence'):.2f}"
    )
    click.echo(
        f"  human_review={report.get('requires_human_review')}  "
        f"model={report.get('model')}  "
        f"elapsed={report.get('elapsed_ms'):.1f} ms"
    )
    click.echo("-" * 60)
    click.echo("SUMMARY:")
    click.echo(f"  {report.get('summary')}")
    click.echo("NARRATIVE:")
    click.echo(f"  {report.get('narrative')}")
    click.echo("MATCHED PATTERNS:")
    for p in report.get("matched_patterns", []):
        click.echo(f"  - {p['name']:<18s} conf={p['confidence']:.2f}  {p['rationale']}")
    click.echo("ENTITY RISKS:")
    for e in report.get("entity_risks", []):
        click.echo(
            f"  - {e['entity_type']:<8s} {e['entity_id']:<16s} "
            f"score={e['risk_score']:.2f}  factors={e['contributing_factors']}"
        )
    click.echo("SIMILAR CASES:")
    for c in report.get("similar_cases", []):
        click.echo(
            f"  - {c['case_id']:<10s} sim={c['similarity']:.3f}  "
            f"pattern={c['pattern']}  -- {c['summary']}"
        )
    click.echo("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
