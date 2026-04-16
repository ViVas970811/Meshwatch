"""Download the IEEE-CIS Fraud Detection dataset from Kaggle.

Two auth schemes are supported:

1. **Bearer token (``KAGGLE_API_TOKEN``)** -- Kaggle's newer token format
   (``KGAT_...``). Sent as ``Authorization: Bearer <token>`` against the
   public REST API. This path bypasses the legacy Python ``kaggle`` CLI
   entirely, which as of 1.7.4.5 does not understand the new format.

2. **Legacy ``KAGGLE_USERNAME`` + ``KAGGLE_KEY``** -- classic kaggle.json /
   username-and-key flow. Delegates to ``kaggle.api.kaggle_api_extended``.

Credentials
-----------
* Prefer env vars (``.env`` file at project root is auto-loaded by the
  config system). ``KAGGLE_API_TOKEN`` takes precedence over the legacy
  pair if both are set.
* Or place ``~/.kaggle/kaggle.json`` with ``{"username": "...", "key": "..."}``.

Kaggle also requires you to accept the IEEE-CIS competition rules at
https://www.kaggle.com/c/ieee-fraud-detection/rules before downloads work.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from fraud_detection.utils.config import AppConfig, load_config
from fraud_detection.utils.logging import configure_logging, get_logger

if TYPE_CHECKING:  # pragma: no cover
    import requests  # noqa: F401

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class KaggleCredentialsError(RuntimeError):
    """Raised when Kaggle credentials cannot be located."""


class DownloadError(RuntimeError):
    """Raised when the Kaggle download fails or produces no files."""


# ---------------------------------------------------------------------------
# Credential discovery
# ---------------------------------------------------------------------------


@dataclass
class KaggleCredentials:
    """Represents a resolved credential bundle.

    Exactly one of ``bearer_token`` or ``(username, key)`` is populated.
    """

    username: str | None = None
    key: str | None = None
    bearer_token: str | None = None

    @property
    def is_bearer(self) -> bool:
        return self.bearer_token is not None

    def to_env(self) -> None:
        """Export credentials to the env vars the Kaggle CLI expects."""
        if self.is_bearer:
            os.environ["KAGGLE_API_TOKEN"] = self.bearer_token or ""
        if self.username:
            os.environ["KAGGLE_USERNAME"] = self.username
        if self.key:
            os.environ["KAGGLE_KEY"] = self.key


def _load_creds_from_env() -> KaggleCredentials | None:
    bearer = os.environ.get("KAGGLE_API_TOKEN")
    if bearer:
        return KaggleCredentials(bearer_token=bearer)
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if user and key:
        return KaggleCredentials(username=user, key=key)
    return None


def _load_creds_from_file() -> KaggleCredentials | None:
    import json

    candidate = Path.home() / ".kaggle" / "kaggle.json"
    if not candidate.exists():
        return None
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover -- defensive
        msg = f"Malformed {candidate}: {exc}"
        raise KaggleCredentialsError(msg) from exc
    # Newer kaggle.json format may also carry an api_token field.
    api_token = data.get("api_token") or data.get("token")
    if api_token and str(api_token).startswith("KGAT_"):
        return KaggleCredentials(bearer_token=api_token)
    user = data.get("username")
    key = data.get("key")
    if user and key:
        return KaggleCredentials(username=user, key=key)
    return None


def resolve_credentials() -> KaggleCredentials:
    """Find Kaggle credentials or raise.

    Search order: env vars (``KAGGLE_API_TOKEN`` > ``KAGGLE_USERNAME``/
    ``KAGGLE_KEY``) -> ``~/.kaggle/kaggle.json``.
    """
    creds = _load_creds_from_env() or _load_creds_from_file()
    if creds is None:
        msg = (
            "Kaggle credentials not found. Either set KAGGLE_API_TOKEN "
            "(new-format 'KGAT_...' tokens), or KAGGLE_USERNAME + KAGGLE_KEY "
            "env vars, or place kaggle.json at ~/.kaggle/kaggle.json "
            "(see https://www.kaggle.com/settings -> 'Create New API Token')."
        )
        raise KaggleCredentialsError(msg)
    creds.to_env()
    return creds


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------


class IEEECISDownloader:
    """Downloads the IEEE-CIS competition files.

    Dispatches between two backends:

    * :meth:`_download_via_bearer` -- pure ``requests`` with Bearer auth.
    * :meth:`_download_via_legacy_cli` -- kaggle-api Python package.
    """

    API_BASE = "https://www.kaggle.com/api/v1"

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.slug = self.config.dataset.slug
        self.target_dir = self.config.paths.data_raw

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def download(self, *, force: bool = False, unzip: bool = True) -> list[Path]:
        """Download the competition files.

        Parameters
        ----------
        force
            Re-download even if CSVs already exist locally.
        unzip
            Extract any downloaded ``.zip`` archives in place.

        Returns
        -------
        list[Path]
            Paths of the files now present in ``data_raw``.
        """
        self.target_dir.mkdir(parents=True, exist_ok=True)

        if not force and self._has_expected_files():
            log.info(
                "dataset_already_present",
                target_dir=str(self.target_dir),
                slug=self.slug,
            )
            return sorted(self.target_dir.glob("*.csv"))

        creds = resolve_credentials()
        log.info(
            "dataset_download_start",
            target_dir=str(self.target_dir),
            slug=self.slug,
            auth="bearer" if creds.is_bearer else "legacy",
        )
        if creds.is_bearer:
            self._download_via_bearer(creds.bearer_token, force=force)
        else:
            self._download_via_legacy_cli(force=force)

        if unzip:
            self._extract_zips()

        files = sorted(self.target_dir.glob("*.csv"))
        if not files:
            msg = f"No CSVs found in {self.target_dir} after download."
            raise DownloadError(msg)

        log.info(
            "dataset_download_complete",
            target_dir=str(self.target_dir),
            files=[f.name for f in files],
        )
        return files

    # ------------------------------------------------------------------
    # Bearer backend
    # ------------------------------------------------------------------
    def _download_via_bearer(self, token: str | None, *, force: bool) -> None:
        """Download each competition file via Bearer auth + streaming."""
        import requests

        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {token}"})

        filenames = self._list_files_bearer(session)
        log.info("bearer_list_files", count=len(filenames))

        for name in filenames:
            dest = self.target_dir / name
            if not force and dest.exists():
                log.info("file_already_present", name=name)
                continue
            self._download_one_bearer(session, name, dest)

    def _list_files_bearer(self, session: Any) -> list[str]:
        url = f"{self.API_BASE}/competitions/data/list/{self.slug}"
        r = session.get(url, params={"pagesize": 100})
        self._raise_for_kaggle_error(r)
        payload = r.json()
        files = payload.get("files", []) or []
        names = [f["name"] for f in files if f.get("name")]
        if not names:
            msg = f"No files listed for competition '{self.slug}'."
            raise DownloadError(msg)
        return names

    def _download_one_bearer(self, session: Any, name: str, dest: Path) -> None:
        url = f"{self.API_BASE}/competitions/data/download/{self.slug}/{name}"
        log.info("bearer_download_file_start", name=name)
        with session.get(url, stream=True, allow_redirects=True) as r:
            self._raise_for_kaggle_error(r)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Kaggle wraps individual-file downloads in a ZIP archive, even
            # when the requested filename ends in .csv. Detect by magic bytes
            # and extract in place so the on-disk layout matches expectations.
            if self._is_zip(tmp):
                self._extract_and_replace(tmp, dest)
            else:
                shutil.move(str(tmp), str(dest))
        log.info(
            "bearer_download_file_done",
            name=name,
            bytes=dest.stat().st_size,
        )

    @staticmethod
    def _is_zip(path: Path) -> bool:
        try:
            with path.open("rb") as f:
                return f.read(4) == b"PK\x03\x04"
        except OSError:  # pragma: no cover -- defensive
            return False

    @staticmethod
    def _extract_and_replace(tmp_zip: Path, dest: Path) -> None:
        """Unzip ``tmp_zip`` in place and rename the single inner file to ``dest``.

        Kaggle's per-file downloads always contain exactly one inner file whose
        name matches ``dest.name`` (e.g. ``train_transaction.csv``). If the zip
        has multiple entries we keep the one matching ``dest.name`` and extract
        the rest alongside it.
        """
        with zipfile.ZipFile(tmp_zip) as zf:
            members = zf.namelist()
            if not members:  # pragma: no cover -- defensive
                msg = f"Empty zip archive for {dest.name}"
                raise DownloadError(msg)
            # Extract everything to the parent dir; rename mismatches to dest.
            zf.extractall(dest.parent)
        tmp_zip.unlink()  # discard the wrapper
        # If the inner filename doesn't match dest (rare), find it and rename.
        if not dest.exists():
            for member in members:
                extracted = dest.parent / member
                if extracted.exists():
                    extracted.rename(dest)
                    break

    @staticmethod
    def _raise_for_kaggle_error(response: Any) -> None:
        if response.ok:
            return
        try:
            body = response.json()
            message = body.get("message") or response.text[:200]
        except ValueError:
            message = response.text[:200]
        if response.status_code == 403 and "rules" in str(message).lower():
            msg = (
                "Kaggle returned 403: you must accept the competition rules "
                "at https://www.kaggle.com/c/ieee-fraud-detection/rules "
                "before downloading any files."
            )
            raise DownloadError(msg)
        msg = f"Kaggle API error {response.status_code}: {message}"
        raise DownloadError(msg)

    # ------------------------------------------------------------------
    # Legacy CLI backend
    # ------------------------------------------------------------------
    def _download_via_legacy_cli(self, *, force: bool) -> None:
        api = self._kaggle_api()
        try:
            api.competition_download_files(
                competition=self.slug,
                path=str(self.target_dir),
                force=force,
                quiet=False,
            )
        except Exception as exc:
            msg = (
                f"Failed to download competition '{self.slug}'. "
                "Ensure you have accepted the competition rules at "
                f"https://www.kaggle.com/c/{self.slug}/rules. "
                f"Original error: {exc}"
            )
            raise DownloadError(msg) from exc

    @staticmethod
    def _kaggle_api():
        """Import and authenticate the Kaggle API lazily."""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover -- integration concern
            msg = (
                "The 'kaggle' package is required to download the dataset. "
                "Install with: pip install -e '.[dev]' (it's in the base deps)."
            )
            raise DownloadError(msg) from exc

        api = KaggleApi()
        api.authenticate()
        return api

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _expected_filenames(self) -> list[str]:
        d = self.config.dataset
        return [
            d.transaction_file,
            d.identity_file,
            d.test_transaction_file,
            d.test_identity_file,
        ]

    def _has_expected_files(self) -> bool:
        return all((self.target_dir / name).exists() for name in self._expected_filenames())

    def _extract_zips(self) -> None:
        for zip_path in self.target_dir.glob("*.zip"):
            log.info("extracting_zip", zip_path=str(zip_path))
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(self.target_dir)
            # Keep the zip around in case of network re-runs; user can
            # delete via ``make clean-all``.


# ---------------------------------------------------------------------------
# CLI entrypoint -- referenced by [project.scripts] in pyproject.toml
# ---------------------------------------------------------------------------


@click.command("fraud-download")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to a YAML config (defaults to configs/base.yaml).",
)
@click.option("--force", is_flag=True, help="Re-download even if files exist.")
@click.option("--no-unzip", is_flag=True, help="Do not auto-extract .zip archives.")
def main(config_path: str | None, force: bool, no_unzip: bool) -> None:
    """Download the IEEE-CIS Fraud Detection dataset to ``data/raw``."""
    cfg = load_config(config_path)
    configure_logging(level=cfg.logging.level, json=cfg.logging.use_json)
    downloader = IEEECISDownloader(cfg)
    downloader.download(force=force, unzip=not no_unzip)


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "DownloadError",
    "IEEECISDownloader",
    "KaggleCredentials",
    "KaggleCredentialsError",
    "main",
    "resolve_credentials",
]
