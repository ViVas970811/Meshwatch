"""Tests for ``fraud_detection.data.download``.

We stub out the Kaggle CLI since these must stay offline/fast.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fraud_detection.data import download as dl
from fraud_detection.utils.config import AppConfig

# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------


def test_resolve_credentials_prefers_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "me")
    monkeypatch.setenv("KAGGLE_KEY", "secret-key")
    creds = dl.resolve_credentials()
    assert creds.username == "me"
    assert creds.key == "secret-key"


def test_resolve_credentials_falls_back_to_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # No env vars.
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    # Fake kaggle.json in a custom home dir.
    fake_home = tmp_path / "home"
    (fake_home / ".kaggle").mkdir(parents=True)
    (fake_home / ".kaggle" / "kaggle.json").write_text(
        json.dumps({"username": "file-user", "key": "file-key"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(dl.Path, "home", classmethod(lambda cls: fake_home))

    creds = dl.resolve_credentials()
    assert creds.username == "file-user"
    assert creds.key == "file-key"


def test_resolve_credentials_raises_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    # Isolated home with no kaggle.json.
    monkeypatch.setattr(dl.Path, "home", classmethod(lambda cls: tmp_path))
    with pytest.raises(dl.KaggleCredentialsError):
        dl.resolve_credentials()


# ---------------------------------------------------------------------------
# Downloader short-circuits when files already exist
# ---------------------------------------------------------------------------


def test_download_short_circuits_when_files_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Point config at a custom raw dir that already contains the expected CSVs.
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cfg = AppConfig(paths={"data_raw": str(raw_dir)})
    for name in (
        cfg.dataset.transaction_file,
        cfg.dataset.identity_file,
        cfg.dataset.test_transaction_file,
        cfg.dataset.test_identity_file,
    ):
        (raw_dir / name).write_text("TransactionID,isFraud\n1,0\n", encoding="utf-8")

    downloader = dl.IEEECISDownloader(cfg)
    # Fail loudly if the Kaggle API is ever constructed -- it shouldn't be.
    monkeypatch.setattr(
        dl.IEEECISDownloader,
        "_kaggle_api",
        staticmethod(lambda: (_ for _ in ()).throw(AssertionError("should not be called"))),
    )
    files = downloader.download(force=False, unzip=False)
    assert len(files) == 4


def test_download_calls_kaggle_api_when_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cfg = AppConfig(paths={"data_raw": str(raw_dir)})
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")

    fake_api = MagicMock()

    def _fake_download(*, competition, path, force, quiet):
        # Simulate Kaggle writing the expected CSVs.
        for name in (
            cfg.dataset.transaction_file,
            cfg.dataset.identity_file,
            cfg.dataset.test_transaction_file,
            cfg.dataset.test_identity_file,
        ):
            (Path(path) / name).write_text("TransactionID,isFraud\n1,0\n", encoding="utf-8")

    fake_api.competition_download_files.side_effect = _fake_download
    monkeypatch.setattr(dl.IEEECISDownloader, "_kaggle_api", staticmethod(lambda: fake_api))

    downloader = dl.IEEECISDownloader(cfg)
    files = downloader.download(force=True, unzip=False)
    assert len(files) == 4
    fake_api.competition_download_files.assert_called_once()


def test_download_raises_when_kaggle_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cfg = AppConfig(paths={"data_raw": str(raw_dir)})
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")

    fake_api = MagicMock()
    fake_api.competition_download_files.side_effect = Exception("403 Forbidden")
    monkeypatch.setattr(dl.IEEECISDownloader, "_kaggle_api", staticmethod(lambda: fake_api))

    downloader = dl.IEEECISDownloader(cfg)
    with pytest.raises(dl.DownloadError, match="Failed to download"):
        downloader.download(force=True)
