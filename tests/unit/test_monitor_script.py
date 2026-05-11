"""Unit tests for scripts/monitor.py (Phase 7 CLI)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from click.testing import CliRunner
from scripts.monitor import main as monitor_cli


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _make_dataframes():
    rng = np.random.default_rng(99)
    ref = pd.DataFrame(
        {
            "amount": rng.normal(loc=0, scale=1, size=300),
            "country": rng.choice(["US", "GB", "CA"], size=300),
        }
    )
    cur = pd.DataFrame(
        {
            "amount": rng.normal(loc=4, scale=1, size=300),  # severe drift
            "country": rng.choice(["US", "GB", "CA"], size=300),
        }
    )
    return ref, cur


def test_monitor_cli_writes_json_and_html(tmp_path):
    ref_path = tmp_path / "ref.parquet"
    cur_path = tmp_path / "cur.parquet"
    out_dir = tmp_path / "out"
    ref, cur = _make_dataframes()
    _write_parquet(ref_path, ref)
    _write_parquet(cur_path, cur)

    runner = CliRunner()
    result = runner.invoke(
        monitor_cli,
        [
            "--reference",
            str(ref_path),
            "--current",
            str(cur_path),
            "--output-dir",
            str(out_dir),
            "--top-k",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "drift.json").exists()
    assert (out_dir / "drift.html").exists()

    payload = json.loads((out_dir / "drift.json").read_text())
    assert payload["n_features"] == 2
    # Severe shift on `amount` -> overall severity is "severe".
    assert payload["severity"] == "severe"


def test_monitor_cli_accepts_csv_input(tmp_path):
    ref_path = tmp_path / "ref.csv"
    cur_path = tmp_path / "cur.csv"
    out_dir = tmp_path / "out"
    ref, cur = _make_dataframes()
    ref.to_csv(ref_path, index=False)
    cur.to_csv(cur_path, index=False)

    runner = CliRunner()
    result = runner.invoke(
        monitor_cli,
        [
            "--reference",
            str(ref_path),
            "--current",
            str(cur_path),
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "drift.json").exists()


def test_monitor_cli_respects_feature_subset(tmp_path):
    ref_path = tmp_path / "ref.parquet"
    cur_path = tmp_path / "cur.parquet"
    out_dir = tmp_path / "out"
    ref, cur = _make_dataframes()
    _write_parquet(ref_path, ref)
    _write_parquet(cur_path, cur)

    runner = CliRunner()
    result = runner.invoke(
        monitor_cli,
        [
            "--reference",
            str(ref_path),
            "--current",
            str(cur_path),
            "--output-dir",
            str(out_dir),
            "--features",
            "amount",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads((out_dir / "drift.json").read_text())
    feats = {f["feature"] for f in payload["features"]}
    assert feats == {"amount"}


def test_monitor_cli_exits_with_two_when_no_overlap(tmp_path):
    ref_path = tmp_path / "ref.parquet"
    cur_path = tmp_path / "cur.parquet"
    out_dir = tmp_path / "out"
    _write_parquet(ref_path, pd.DataFrame({"only_ref": [1, 2, 3]}))
    _write_parquet(cur_path, pd.DataFrame({"only_cur": [4, 5, 6]}))

    runner = CliRunner()
    result = runner.invoke(
        monitor_cli,
        [
            "--reference",
            str(ref_path),
            "--current",
            str(cur_path),
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 2
    assert "No overlapping features" in result.output
