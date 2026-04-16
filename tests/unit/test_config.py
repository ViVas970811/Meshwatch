"""Tests for ``fraud_detection.utils.config``."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

from fraud_detection.utils.config import (
    DEFAULT_CONFIG_PATH,
    AppConfig,
    SplitsConfig,
    load_config,
)


def test_defaults_are_valid():
    cfg = AppConfig()
    assert cfg.project.name == "fraud-detection-gnn"
    assert cfg.splits.train_frac == pytest.approx(0.60)
    # fractions sum to 1.0
    s = cfg.splits
    assert s.train_frac + s.val_frac + s.test_frac == pytest.approx(1.0)


def test_paths_resolve_to_project_root():
    cfg = AppConfig()
    # Default is "data/raw" which should be resolved against PROJECT_ROOT.
    assert cfg.paths.data_raw.is_absolute()
    assert cfg.paths.data_raw.name == "raw"


def test_paths_absolute_are_kept_as_is(tmp_path: Path):
    # Absolute path should bypass the PROJECT_ROOT join.
    cfg = AppConfig(paths={"data_raw": str(tmp_path / "custom_raw")})
    assert cfg.paths.data_raw == tmp_path / "custom_raw"


def test_ensure_exist_creates_dirs(tmp_path: Path):
    cfg = AppConfig(
        paths={
            "data_raw": str(tmp_path / "r"),
            "data_processed": str(tmp_path / "p"),
            "data_splits": str(tmp_path / "s"),
            "data_graphs": str(tmp_path / "g"),
        }
    )
    cfg.paths.ensure_exist()
    for p in (tmp_path / "r", tmp_path / "p", tmp_path / "s", tmp_path / "g"):
        assert p.exists()


def test_splits_fractions_must_sum_to_one():
    with pytest.raises(ValidationError) as excinfo:
        SplitsConfig(train_frac=0.7, val_frac=0.2, test_frac=0.2)  # sums to 1.1
    assert "sum to 1.0" in str(excinfo.value)


def test_from_yaml_roundtrip(tmp_path: Path):
    yaml_path = tmp_path / "custom.yaml"
    yaml_path.write_text(
        dedent(
            """
            project:
              name: fraud-detection-gnn
              version: "0.1.0-test"
              seed: 7
            splits:
              strategy: temporal
              train_frac: 0.5
              val_frac: 0.25
              test_frac: 0.25
            """
        ).strip(),
        encoding="utf-8",
    )
    cfg = AppConfig.from_yaml(yaml_path)
    assert cfg.project.version == "0.1.0-test"
    assert cfg.project.seed == 7
    assert cfg.splits.train_frac == pytest.approx(0.5)


def test_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    yaml_path = tmp_path / "c.yaml"
    yaml_path.write_text("project:\n  seed: 1\n", encoding="utf-8")
    monkeypatch.setenv("FRAUD_PROJECT__SEED", "999")
    cfg = AppConfig.from_yaml(yaml_path)
    assert cfg.project.seed == 999


def test_load_config_uses_default_when_file_missing(tmp_path: Path):
    # Point at a non-existent config file -> falls back to defaults.
    missing = tmp_path / "does_not_exist.yaml"
    cfg = load_config(missing)
    assert isinstance(cfg, AppConfig)


def test_load_config_reads_default_yaml():
    # The default YAML shipped with the repo must always parse cleanly.
    assert DEFAULT_CONFIG_PATH.exists(), "configs/base.yaml should exist"
    cfg = load_config()
    assert cfg.project.name == "fraud-detection-gnn"
    assert cfg.dataset.slug == "ieee-fraud-detection"
    assert cfg.preprocessing.missing_strategy.v_features.drop_threshold == pytest.approx(0.75)


def test_missing_strategy_defaults_match_plan():
    cfg = AppConfig()
    ms = cfg.preprocessing.missing_strategy
    # These values are taken verbatim from the implementation plan page 4.
    assert ms.v_features.drop_threshold == pytest.approx(0.75)
    assert ms.v_features.fill_value == -999
    assert ms.v_features.add_indicator is True
    assert ms.d_features.fill_value == -1
    assert ms.d_features.add_indicator is True
    assert ms.c_features.fill_value == 0
    assert ms.c_features.add_indicator is False
    assert ms.m_features.fill_value == "missing"
    assert ms.id_numeric.fill_value == -999
    assert ms.id_categorical.fill_value == "unknown"
    assert ms.email_domains.fill_value == "unknown"
