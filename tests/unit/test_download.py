"""Tests for ``fraud_detection.data.download``.

We stub out Kaggle (legacy CLI + Bearer HTTP) since these must stay
offline/fast.
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


def test_resolve_credentials_prefers_bearer_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_deadbeef")
    # Also set legacy; Bearer should still win.
    monkeypatch.setenv("KAGGLE_USERNAME", "ignored")
    monkeypatch.setenv("KAGGLE_KEY", "ignored")
    creds = dl.resolve_credentials()
    assert creds.is_bearer
    assert creds.bearer_token == "KGAT_deadbeef"
    assert creds.username is None
    assert creds.key is None


def test_resolve_credentials_uses_legacy_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setenv("KAGGLE_USERNAME", "me")
    monkeypatch.setenv("KAGGLE_KEY", "secret-key")
    creds = dl.resolve_credentials()
    assert not creds.is_bearer
    assert creds.username == "me"
    assert creds.key == "secret-key"


def test_resolve_credentials_falls_back_to_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for var in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(var, raising=False)

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


def test_resolve_credentials_reads_kgat_from_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for var in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(var, raising=False)

    fake_home = tmp_path / "home"
    (fake_home / ".kaggle").mkdir(parents=True)
    (fake_home / ".kaggle" / "kaggle.json").write_text(
        json.dumps({"api_token": "KGAT_xyz"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(dl.Path, "home", classmethod(lambda cls: fake_home))

    creds = dl.resolve_credentials()
    assert creds.is_bearer
    assert creds.bearer_token == "KGAT_xyz"


def test_resolve_credentials_raises_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for var in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(var, raising=False)
    # Isolated home with no kaggle.json.
    monkeypatch.setattr(dl.Path, "home", classmethod(lambda cls: tmp_path))
    with pytest.raises(dl.KaggleCredentialsError):
        dl.resolve_credentials()


# ---------------------------------------------------------------------------
# Downloader short-circuits when files already exist
# ---------------------------------------------------------------------------


def _seed_expected_files(raw_dir: Path, cfg: AppConfig) -> None:
    for name in (
        cfg.dataset.transaction_file,
        cfg.dataset.identity_file,
        cfg.dataset.test_transaction_file,
        cfg.dataset.test_identity_file,
    ):
        (raw_dir / name).write_text("TransactionID,isFraud\n1,0\n", encoding="utf-8")


def test_download_short_circuits_when_files_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cfg = AppConfig(paths={"data_raw": str(raw_dir)})
    _seed_expected_files(raw_dir, cfg)

    downloader = dl.IEEECISDownloader(cfg)
    # Guard: both backends must NOT be invoked.
    monkeypatch.setattr(
        dl.IEEECISDownloader,
        "_kaggle_api",
        staticmethod(lambda: (_ for _ in ()).throw(AssertionError("legacy CLI called"))),
    )
    monkeypatch.setattr(
        dl.IEEECISDownloader,
        "_download_via_bearer",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("bearer called")),
    )
    files = downloader.download(force=False, unzip=False)
    assert len(files) == 4


# ---------------------------------------------------------------------------
# Legacy CLI backend
# ---------------------------------------------------------------------------


def test_download_uses_legacy_cli_for_username_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cfg = AppConfig(paths={"data_raw": str(raw_dir)})
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")

    fake_api = MagicMock()

    def _fake_download(*, competition, path, force, quiet):
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


def test_download_raises_when_legacy_kaggle_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cfg = AppConfig(paths={"data_raw": str(raw_dir)})
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")

    fake_api = MagicMock()
    fake_api.competition_download_files.side_effect = Exception("403 Forbidden")
    monkeypatch.setattr(dl.IEEECISDownloader, "_kaggle_api", staticmethod(lambda: fake_api))

    downloader = dl.IEEECISDownloader(cfg)
    with pytest.raises(dl.DownloadError, match="Failed to download"):
        downloader.download(force=True)


# ---------------------------------------------------------------------------
# Bearer backend
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        json_body: dict | None = None,
        chunks: list[bytes] | None = None,
    ) -> None:
        self.status_code = status_code
        self._json = json_body or {}
        self._chunks = chunks or []
        self.text = json.dumps(self._json) if json_body else ""
        self.ok = 200 <= status_code < 300

    def json(self) -> dict:
        return self._json

    def iter_content(self, chunk_size: int = 0):
        return iter(self._chunks)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeSession:
    """Minimal ``requests.Session`` stand-in."""

    def __init__(self, list_payload: dict, file_chunks: bytes):
        self.headers: dict = {}
        self.calls: list[tuple[str, str]] = []
        self._list_payload = list_payload
        self._file_chunks = file_chunks

    def update_headers(self, d):
        self.headers.update(d)

    def get(self, url: str, *, params=None, stream=False, allow_redirects=True):
        self.calls.append(("GET", url))
        if "/list/" in url:
            return _FakeResponse(status_code=200, json_body=self._list_payload)
        if "/download/" in url:
            return _FakeResponse(status_code=200, chunks=[self._file_chunks])
        return _FakeResponse(status_code=404, json_body={"message": "nope"})


def _patch_requests(monkeypatch: pytest.MonkeyPatch, session) -> None:
    """Replace ``requests`` inside the download module with a fake."""
    import sys

    fake_requests = MagicMock()
    fake_requests.Session.return_value = session
    monkeypatch.setitem(sys.modules, "requests", fake_requests)


def test_download_uses_bearer_when_token_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cfg = AppConfig(paths={"data_raw": str(raw_dir)})
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_test")

    expected = [
        cfg.dataset.transaction_file,
        cfg.dataset.identity_file,
        cfg.dataset.test_transaction_file,
        cfg.dataset.test_identity_file,
    ]
    fake_session = _FakeSession(
        list_payload={"files": [{"name": n} for n in expected]},
        file_chunks=b"TransactionID,isFraud\n1,0\n",
    )
    _patch_requests(monkeypatch, fake_session)

    downloader = dl.IEEECISDownloader(cfg)
    files = downloader.download(force=True, unzip=False)

    assert fake_session.headers.get("Authorization") == "Bearer KGAT_test"
    # 1 list call + 4 download calls.
    assert len(fake_session.calls) == 5
    assert len(files) == 4


def test_bearer_backend_raises_rules_error_on_403(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cfg = AppConfig(paths={"data_raw": str(raw_dir)})
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_test")

    class _RulesSession(_FakeSession):
        def get(self, url, *, params=None, stream=False, allow_redirects=True):
            return _FakeResponse(
                status_code=403,
                json_body={"message": "You must accept this competition's rules"},
            )

    fake_session = _RulesSession(list_payload={}, file_chunks=b"")
    _patch_requests(monkeypatch, fake_session)

    downloader = dl.IEEECISDownloader(cfg)
    with pytest.raises(dl.DownloadError, match="accept the competition rules"):
        downloader.download(force=True)


def test_bearer_skips_existing_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cfg = AppConfig(paths={"data_raw": str(raw_dir)})
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_test")

    # One of the four expected files already exists -> download should skip it.
    # We seed just one so the short-circuit check (all four exist) is false,
    # but the per-file "already present" path is exercised.
    already = raw_dir / cfg.dataset.transaction_file
    already.write_text("x", encoding="utf-8")

    expected = [
        cfg.dataset.transaction_file,
        cfg.dataset.identity_file,
        cfg.dataset.test_transaction_file,
        cfg.dataset.test_identity_file,
    ]
    fake_session = _FakeSession(
        list_payload={"files": [{"name": n} for n in expected]},
        file_chunks=b"TransactionID,isFraud\n1,0\n",
    )
    _patch_requests(monkeypatch, fake_session)

    downloader = dl.IEEECISDownloader(cfg)
    downloader.download(force=False, unzip=False)
    # 1 list + 3 downloads (the pre-existing file was skipped).
    assert len(fake_session.calls) == 4
