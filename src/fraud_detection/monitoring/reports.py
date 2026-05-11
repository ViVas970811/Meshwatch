"""Drift report serialisation -- JSON + self-contained HTML.

The HTML renderer ships with all assets inlined so reports can be emailed,
attached to tickets, or served behind ``nginx`` without rebuilding the
asset pipeline. We deliberately keep the template tiny (no JS, no
external fonts) so the bundle is well under 50 KB even for the full
119-feature snapshot.
"""

from __future__ import annotations

import html as _html
import json
from pathlib import Path
from typing import Any

from fraud_detection.monitoring.drift import DriftReport, FeatureDrift

# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def report_to_json(report: DriftReport, *, indent: int = 2) -> str:
    return json.dumps(report.to_dict(), indent=indent, sort_keys=False)


def write_json(report: DriftReport, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(report_to_json(report), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_SEVERITY_CLASS = {
    "none": "ok",
    "moderate": "warn",
    "severe": "alert",
}

_HTML_STYLES = """
:root {
    --bg: #0b1020;
    --panel: #131a30;
    --panel-2: #1b2440;
    --text: #e6e8ef;
    --muted: #8892a6;
    --ok: #16a34a;
    --warn: #d97706;
    --alert: #dc2626;
    --accent: #6366f1;
}
* { box-sizing: border-box; }
body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
header {
    padding: 24px 32px;
    background: linear-gradient(180deg, #182142 0%, #0f1530 100%);
    border-bottom: 1px solid #1e2848;
}
h1 { margin: 0 0 6px 0; font-size: 20px; }
.subtitle { color: var(--muted); font-size: 13px; }
.container { padding: 24px 32px; }
.summary {
    display: grid;
    grid-template-columns: repeat(4, minmax(160px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
}
.card {
    background: var(--panel);
    border: 1px solid #1e2848;
    border-radius: 8px;
    padding: 14px 16px;
}
.card .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
.card .value { font-size: 22px; font-weight: 600; margin-top: 4px; }
.severity {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
}
.severity.ok    { background: rgba(22, 163, 74, 0.15);  color: var(--ok); }
.severity.warn  { background: rgba(217, 119, 6, 0.18);  color: var(--warn); }
.severity.alert { background: rgba(220, 38, 38, 0.18);  color: var(--alert); }
table { width: 100%; border-collapse: collapse; background: var(--panel); border-radius: 8px; overflow: hidden; }
th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid #1e2848; font-variant-numeric: tabular-nums; }
th { background: var(--panel-2); color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; font-size: 11px; }
tr:last-child td { border-bottom: none; }
.bar { display: inline-block; height: 6px; background: linear-gradient(90deg, var(--ok) 0%, var(--warn) 60%, var(--alert) 100%); border-radius: 3px; }
.muted { color: var(--muted); }
footer { padding: 16px 32px; color: var(--muted); font-size: 12px; }
""".strip()


def _fmt(v: Any, *, digits: int = 4) -> str:
    if v is None:
        return "<span class='muted'>--</span>"
    if isinstance(v, float):
        if v != v:  # NaN
            return "<span class='muted'>--</span>"
        return f"{v:.{digits}f}"
    return _html.escape(str(v))


def _feature_row(f: FeatureDrift, *, max_psi: float) -> str:
    klass = _SEVERITY_CLASS.get(f.severity, "ok")
    width = min(100, int((f.psi / max(max_psi, 1e-9)) * 100)) if f.psi > 0 else 1
    return (
        "<tr>"
        f"<td>{_html.escape(f.feature)}</td>"
        f"<td>{_html.escape(f.kind)}</td>"
        f"<td>{_fmt(f.psi)}</td>"
        f"<td><span class='bar' style='width:{width}%'></span></td>"
        f"<td>{_fmt(f.ks_statistic)}</td>"
        f"<td>{_fmt(f.ks_pvalue)}</td>"
        f"<td>{_fmt(f.chi2_statistic, digits=2)}</td>"
        f"<td>{_fmt(f.js_divergence)}</td>"
        f"<td>{_fmt(f.mean_shift)}</td>"
        f"<td><span class='severity {klass}'>{_html.escape(f.severity)}</span></td>"
        f"<td class='muted'>{f.n_reference} / {f.n_current}</td>"
        "</tr>"
    )


def report_to_html(report: DriftReport, *, title: str = "Meshwatch drift report") -> str:
    """Render a self-contained HTML report. No external assets, no JS."""
    klass = _SEVERITY_CLASS.get(report.severity, "ok")
    max_psi = max((f.psi for f in report.features), default=0.0)
    feature_rows = "\n".join(_feature_row(f, max_psi=max_psi) for f in report.top(50))
    body = f"""
<header>
    <h1>{_html.escape(title)}</h1>
    <div class="subtitle">
        Reference: <strong>{_html.escape(report.reference_label)}</strong> ({report.n_reference} rows)
        &nbsp;|&nbsp; Current: <strong>{_html.escape(report.current_label)}</strong> ({report.n_current} rows)
        &nbsp;|&nbsp; Generated {report.generated_at.isoformat()}
    </div>
</header>
<div class="container">
    <section class="summary">
        <div class="card">
            <div class="label">Overall PSI</div>
            <div class="value">{report.overall_psi:.4f}</div>
        </div>
        <div class="card">
            <div class="label">Severity</div>
            <div class="value"><span class="severity {klass}">{_html.escape(report.severity)}</span></div>
        </div>
        <div class="card">
            <div class="label">Moderate features</div>
            <div class="value">{report.n_moderate}</div>
        </div>
        <div class="card">
            <div class="label">Severe features</div>
            <div class="value">{report.n_severe}</div>
        </div>
    </section>
    <h2 style="margin: 8px 0 12px 0;">Top {min(50, report.n_features)} features by PSI</h2>
    <table>
        <thead>
            <tr>
                <th>Feature</th>
                <th>Kind</th>
                <th>PSI</th>
                <th>Magnitude</th>
                <th>KS</th>
                <th>KS p-value</th>
                <th>χ²</th>
                <th>JSD</th>
                <th>Mean Δ</th>
                <th>Severity</th>
                <th>n ref / cur</th>
            </tr>
        </thead>
        <tbody>{feature_rows}</tbody>
    </table>
</div>
<footer>
    PSI warn threshold {report.psi_warn:.2f} &middot; PSI alert threshold {report.psi_alert:.2f}
    &middot; Meshwatch v0.7.0-mlops
</footer>
""".strip()
    return (
        "<!doctype html>\n"
        "<html lang='en'>\n"
        "<head>\n"
        f"<meta charset='utf-8'><title>{_html.escape(title)}</title>\n"
        f"<style>{_HTML_STYLES}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )


def write_html(report: DriftReport, path: str | Path, *, title: str | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        report_to_html(report, title=title or "Meshwatch drift report"),
        encoding="utf-8",
    )
    return p


__all__ = [
    "report_to_html",
    "report_to_json",
    "write_html",
    "write_json",
]
