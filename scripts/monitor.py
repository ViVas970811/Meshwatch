"""CLI: compute a drift report between a reference (training) split and
a current (recent production) split.

Examples
--------

# Default paths: data/splits/train.parquet vs data/splits/test.parquet
python scripts/monitor.py

# Use an explicit current window (e.g. recent production capture)
python scripts/monitor.py \
    --reference data/splits/train.parquet \
    --current   data/captures/2026-05-01.parquet \
    --output-dir data/reports/2026-05-01 \
    --top-k 25

# Restrict the comparison to a feature subset
python scripts/monitor.py --features TransactionAmt,card1,P_emaildomain

Writes both a JSON and an HTML report under ``--output-dir`` and -- when
MLflow is available -- mirrors the summary metrics under the
``meshwatch-production`` experiment.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path

import click

from fraud_detection.monitoring import (
    DriftDetector,
    DriftDetectorConfig,
    DriftReport,
    evidently_html_report,
    label_drift,
    monitoring_metrics,
    prediction_distribution_drift,
    write_html,
    write_json,
)
from fraud_detection.utils.logging import configure_logging, get_logger

log = get_logger(__name__)


def _load_frame(path: Path) -> dict[str, list]:
    """Load a Parquet / CSV file into ``dict[col, list[value]]``."""
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix in {".csv", ".tsv", ".txt"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    else:
        raise click.BadParameter(f"Unsupported file format: {path.suffix}")
    return {col: df[col].tolist() for col in df.columns}


def _filter_features(
    frame: dict[str, list],
    features: Iterable[str] | None,
) -> dict[str, list]:
    if not features:
        return frame
    keep = set(features)
    return {k: v for k, v in frame.items() if k in keep}


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--reference",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("data/splits/train.parquet"),
    show_default=True,
    help="Reference distribution (typically the training split).",
)
@click.option(
    "--current",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("data/splits/test.parquet"),
    show_default=True,
    help="Current distribution (typically a recent production capture).",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/reports/latest"),
    show_default=True,
    help="Where to write drift.json + drift.html.",
)
@click.option(
    "--features",
    type=str,
    default=None,
    help="Comma-separated subset of features to compare (default: all overlapping cols).",
)
@click.option(
    "--top-k",
    type=int,
    default=20,
    show_default=True,
    help="Print this many top-drifted features to stdout.",
)
@click.option(
    "--psi-warn", type=float, default=0.10, show_default=True, help="Warn threshold for PSI."
)
@click.option(
    "--psi-alert", type=float, default=0.25, show_default=True, help="Alert threshold for PSI."
)
@click.option(
    "--with-evidently/--no-evidently",
    default=False,
    show_default=True,
    help="Also write an Evidently HTML report (requires the [monitor] extra).",
)
@click.option(
    "--mlflow/--no-mlflow",
    default=False,
    show_default=True,
    help="Log summary metrics to MLflow under experiment 'meshwatch-production'.",
)
def main(
    reference: Path,
    current: Path,
    output_dir: Path,
    features: str | None,
    top_k: int,
    psi_warn: float,
    psi_alert: float,
    with_evidently: bool,
    mlflow: bool,
) -> None:
    """Compute drift between two snapshots and write a JSON + HTML report."""
    configure_logging(level="INFO")
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("monitor_loading_reference", path=str(reference))
    ref_frame = _load_frame(reference)
    log.info("monitor_loading_current", path=str(current))
    cur_frame = _load_frame(current)

    feature_list = (
        [f.strip() for f in features.split(",") if f.strip()] if features is not None else None
    )
    ref_frame = _filter_features(ref_frame, feature_list)
    cur_frame = _filter_features(cur_frame, feature_list)

    common = sorted(set(ref_frame) & set(cur_frame))
    if not common:
        click.echo("No overlapping features between reference and current.", err=True)
        sys.exit(2)
    ref_frame = {k: ref_frame[k] for k in common}
    cur_frame = {k: cur_frame[k] for k in common}

    detector = DriftDetector(
        ref_frame,
        config=DriftDetectorConfig(psi_warn=psi_warn, psi_alert=psi_alert),
    )
    report: DriftReport = detector.detect(cur_frame)

    json_path = write_json(report, output_dir / "drift.json")
    html_path = write_html(report, output_dir / "drift.html")
    log.info(
        "monitor_report_written",
        json=str(json_path),
        html=str(html_path),
        overall_psi=report.overall_psi,
        severity=report.severity,
        n_features=report.n_features,
        n_severe=report.n_severe,
    )

    # Print the top-k offenders to stdout so this script doubles as a
    # human-readable diagnostic without anyone opening the HTML.
    click.echo("")
    click.echo(f"Overall PSI: {report.overall_psi:.4f}  ({report.severity})")
    click.echo(f"Features evaluated: {report.n_features}  |  severe: {report.n_severe}")
    click.echo("")
    click.echo(f"{'Feature':<32} {'Kind':<12} {'PSI':>8} {'Severity':<10}")
    click.echo("-" * 64)
    for f in report.top(top_k):
        click.echo(f"{f.feature[:31]:<32} {f.kind:<12} {f.psi:>8.4f} {f.severity:<10}")
    click.echo("")

    # Optional Evidently HTML (richer, but bigger and slower).
    if with_evidently:
        ev_path = output_dir / "drift_evidently.html"
        result = evidently_html_report(ref_frame, cur_frame, output_path=str(ev_path))
        if result is None:
            log.warning(
                "monitor_evidently_skipped",
                reason="evidently not installed (install with `pip install -e .[monitor]`)",
            )
        else:
            log.info("monitor_evidently_written", path=str(ev_path))

    # Update Prometheus gauges so any running scraper sees the latest values.
    monitoring_metrics.update_drift(report)

    # Label drift -- if both sides have ``isFraud`` (the IEEE-CIS target), we
    # can also surface absolute label drift onto the registry.
    if "isFraud" in ref_frame and "isFraud" in cur_frame:
        ref_labels = [v for v in ref_frame["isFraud"] if v is not None]
        ref_rate = (
            (sum(int(bool(int(v))) for v in ref_labels) / len(ref_labels)) if ref_labels else 0.035
        )
        ld = label_drift(cur_frame["isFraud"], reference_rate=ref_rate)
        monitoring_metrics.update_label_drift(
            production_rate=ld["production_rate"],
            reference_rate=ld["reference_rate"],
        )
        click.echo(
            f"Label drift: production fraud rate {ld['production_rate']:.4f}, "
            f"reference {ld['reference_rate']:.4f}, "
            f"|delta| {ld['absolute_drift']:.4f}"
        )

    # Prediction-distribution drift -- when both sides expose a ``fraud_score``
    # column (i.e. you exported the predictor's responses), measure PSI on the
    # score distribution.
    if "fraud_score" in ref_frame and "fraud_score" in cur_frame:
        psi = prediction_distribution_drift(
            production_scores=cur_frame["fraud_score"],
            reference_scores=ref_frame["fraud_score"],
        )
        monitoring_metrics.update_prediction_distribution_drift(psi)
        click.echo(f"Prediction-distribution PSI: {psi:.4f}")

    # Optional MLflow logging.
    if mlflow:
        try:
            import mlflow as mlf

            mlf.set_experiment("meshwatch-production")
            with mlf.start_run(run_name="drift-snapshot"):
                mlf.log_param("reference", str(reference))
                mlf.log_param("current", str(current))
                mlf.log_param("n_features", report.n_features)
                mlf.log_metric("overall_psi", report.overall_psi)
                mlf.log_metric("n_severe_features", report.n_severe)
                mlf.log_metric("n_moderate_features", report.n_moderate)
                mlf.log_artifact(str(json_path))
                mlf.log_artifact(str(html_path))
            log.info("monitor_mlflow_logged", experiment="meshwatch-production")
        except Exception as exc:
            log.warning("monitor_mlflow_unavailable", error=str(exc))


if __name__ == "__main__":
    main()
