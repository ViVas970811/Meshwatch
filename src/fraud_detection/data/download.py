"""Download the IEEE-CIS Fraud Detection dataset from Kaggle.

The Kaggle CLI is imported lazily so that the rest of the package remains
importable in environments where ``kaggle`` isn't installed yet (e.g. CI
for lint-only jobs).

Credentials
-----------
Either set the env vars ``KAGGLE_USERNAME``/``KAGGLE_KEY`` (preferred in
.env), or place ``~/.kaggle/kaggle.json`` with the standard format::

    {"username": "you", "key": "your_api_key"}
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import click

from fraud_detection.utils.config import AppConfig, load_config
from fraud_detection.utils.logging import configure_logging, get_logger

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
    username: str
    key: str

    def to_env(self) -> None:
        """Export credentials to the env vars the Kaggle CLI expects."""
        os.environ["KAGGLE_USERNAME"] = self.username
        os.environ["KAGGLE_KEY"] = self.key


def _load_creds_from_env() -> KaggleCredentials | None:
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
    user = data.get("username")
    key = data.get("key")
    if not (user and key):
        return None
    return KaggleCredentials(username=user, key=key)


def resolve_credentials() -> KaggleCredentials:
    """Find Kaggle credentials or raise.

    Search order: env vars -> ``~/.kaggle/kaggle.json``.
    """
    creds = _load_creds_from_env() or _load_creds_from_file()
    if creds is None:
        msg = (
            "Kaggle credentials not found. Either set KAGGLE_USERNAME and "
            "KAGGLE_KEY env vars, or place kaggle.json at ~/.kaggle/kaggle.json "
            "(see https://www.kaggle.com/settings -> 'Create New API Token')."
        )
        raise KaggleCredentialsError(msg)
    creds.to_env()
    return creds


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------


class IEEECISDownloader:
    """Thin wrapper around the Kaggle CLI for the IEEE-CIS competition.

    We use ``competition_download_files`` (not ``dataset_download_files``)
    because IEEE-CIS is a competition, not a public dataset. This requires
    the user to have accepted the competition rules on the Kaggle website.
    """

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

        resolve_credentials()
        log.info(
            "dataset_download_start",
            target_dir=str(self.target_dir),
            slug=self.slug,
        )
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
    # Internals
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
