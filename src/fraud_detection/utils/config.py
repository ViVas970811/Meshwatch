"""Layered configuration for the fraud-detection project.

Loads ``configs/base.yaml`` (or an explicit path) first, then overlays any
environment variables prefixed with ``FRAUD_`` using pydantic-settings.

Typical use
-----------

>>> from fraud_detection.utils.config import load_config
>>> cfg = load_config()
>>> cfg.paths.data_raw
PosixPath('data/raw')

Environment overrides use a double-underscore separator for nested keys:

    FRAUD_PROJECT__SEED=7
    FRAUD_DATASET__USE_SUBSET=true
    FRAUD_SPLITS__TRAIN_FRAC=0.7
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
"""Absolute path to the ``fraud-detection-gnn/`` repository root."""

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"


# ---------------------------------------------------------------------------
# Nested config sections
# ---------------------------------------------------------------------------


class ProjectConfig(BaseModel):
    """High-level project metadata."""

    model_config = ConfigDict(extra="forbid")

    name: str = "fraud-detection-gnn"
    version: str = "0.1.0"
    seed: int = 42


class PathsConfig(BaseModel):
    """Filesystem layout.

    Relative paths are resolved against :data:`PROJECT_ROOT`; absolute paths
    are used as-is so overrides from env/CI still work.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    data_raw: Path = Path("data/raw")
    data_processed: Path = Path("data/processed")
    data_splits: Path = Path("data/splits")
    data_graphs: Path = Path("data/graphs")

    @field_validator("data_raw", "data_processed", "data_splits", "data_graphs", mode="after")
    @classmethod
    def _resolve(cls, p: Path) -> Path:
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    def ensure_exist(self) -> None:
        """Create all path dirs if missing (idempotent)."""
        for field_name in self.model_fields:
            path: Path = getattr(self, field_name)
            path.mkdir(parents=True, exist_ok=True)


class MissingGroupStrategy(BaseModel):
    """Missing-value strategy for a feature group."""

    model_config = ConfigDict(extra="forbid")

    fill_value: int | float | str = -999
    add_indicator: bool = False
    drop_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Drop columns whose missing fraction exceeds this threshold.",
    )


class MissingStrategyConfig(BaseModel):
    """Per-group missing-value strategies (see plan page 4)."""

    model_config = ConfigDict(extra="forbid")

    v_features: MissingGroupStrategy = MissingGroupStrategy(
        drop_threshold=0.75, fill_value=-999, add_indicator=True
    )
    d_features: MissingGroupStrategy = MissingGroupStrategy(fill_value=-1, add_indicator=True)
    c_features: MissingGroupStrategy = MissingGroupStrategy(fill_value=0, add_indicator=False)
    m_features: MissingGroupStrategy = MissingGroupStrategy(
        fill_value="missing", add_indicator=False
    )
    id_numeric: MissingGroupStrategy = MissingGroupStrategy(fill_value=-999, add_indicator=True)
    id_categorical: MissingGroupStrategy = MissingGroupStrategy(
        fill_value="unknown", add_indicator=False
    )
    email_domains: MissingGroupStrategy = MissingGroupStrategy(
        fill_value="unknown", add_indicator=False
    )


class PreprocessingConfig(BaseModel):
    """Preprocessing hyperparameters."""

    model_config = ConfigDict(extra="forbid")

    missing_strategy: MissingStrategyConfig = Field(default_factory=MissingStrategyConfig)
    normalize: Literal["standard", "robust", "none"] = "standard"
    clip_quantile: float = Field(default=0.999, ge=0.5, le=1.0)
    log_amount: bool = True


class DatasetConfig(BaseModel):
    """Dataset source + optional subset knobs."""

    model_config = ConfigDict(extra="forbid")

    slug: str = "ieee-fraud-detection"
    transaction_file: str = "train_transaction.csv"
    identity_file: str = "train_identity.csv"
    test_transaction_file: str = "test_transaction.csv"
    test_identity_file: str = "test_identity.csv"
    join_key: str = "TransactionID"
    target: str = "isFraud"
    time_column: str = "TransactionDT"
    use_subset: bool = False
    subset_size: int = 200_000


class SplitsConfig(BaseModel):
    """Temporal train/val/test splits."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["temporal", "random", "stratified"] = "temporal"
    train_frac: float = Field(default=0.60, gt=0.0, lt=1.0)
    val_frac: float = Field(default=0.20, gt=0.0, lt=1.0)
    test_frac: float = Field(default=0.20, gt=0.0, lt=1.0)
    shuffle: bool = False
    assert_non_overlap: bool = True

    @model_validator(mode="after")
    def _fracs_sum_to_one(self) -> SplitsConfig:
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-6:
            msg = (
                f"train/val/test fractions must sum to 1.0 "
                f"(got train={self.train_frac}, val={self.val_frac}, "
                f"test={self.test_frac}, sum={total:.6f})"
            )
            raise ValueError(msg)
        return self


class LoggingConfig(BaseModel):
    """Structured logging configuration.

    ``use_json`` is also accepted as the YAML key ``json`` via alias (kept
    for backwards compatibility with the original config schema).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    use_json: bool = Field(default=True, alias="json")
    static_context: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class _YamlSettingsSource(PydanticBaseSettingsSource):
    """Custom pydantic-settings source that loads values from a YAML file.

    Used via :meth:`AppConfig.settings_customise_sources` so env vars win
    over YAML, which in turn wins over defaults.
    """

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        yaml_path: Path | None,
    ) -> None:
        super().__init__(settings_cls)
        self.yaml_path = yaml_path
        self._data: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._data is not None:
            return self._data
        if self.yaml_path is None or not self.yaml_path.exists():
            self._data = {}
            return self._data
        with self.yaml_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):  # pragma: no cover -- defensive
            msg = f"{self.yaml_path} must contain a YAML mapping at the top level"
            raise ValueError(msg)
        self._data = loaded
        return self._data

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        data = self._load()
        if field_name in data:
            return data[field_name], field_name, True
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._load()


class AppConfig(BaseSettings):
    """Top-level application configuration.

    Loads from (in order of precedence, highest first):

    1. Environment variables prefixed ``FRAUD_`` (e.g. ``FRAUD_PROJECT__SEED=7``).
    2. A ``.env`` file in the project root (same prefix and separator).
    3. A YAML file (``configs/base.yaml`` by default).
    4. Field defaults.
    """

    model_config = SettingsConfigDict(
        env_prefix="FRAUD_",
        env_nested_delimiter="__",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # tolerate unrelated env vars
    )

    # Private class-var holding the active YAML path during ``from_yaml()``.
    _yaml_path: ClassVar[Path | None] = None

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # --------------------------- customise sources ---------------------------

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_src = _YamlSettingsSource(settings_cls, cls._yaml_path)
        # First in tuple = highest priority.
        return (
            init_settings,  # explicit kwargs
            env_settings,  # FRAUD_* env vars
            dotenv_settings,
            yaml_src,  # configs/base.yaml
            file_secret_settings,
        )

    # --------------------------- construction helpers ---------------------------

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str] = DEFAULT_CONFIG_PATH) -> AppConfig:
        """Load config from a YAML file, with env vars taking precedence."""
        yaml_path = Path(path)
        if not yaml_path.exists():
            msg = f"Config file not found: {yaml_path}"
            raise FileNotFoundError(msg)

        # Set the class-level YAML path so the custom source can read it.
        previous = cls._yaml_path
        cls._yaml_path = yaml_path
        try:
            return cls()
        finally:
            cls._yaml_path = previous


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
    """Load the application config.

    Parameters
    ----------
    path
        Optional path to a YAML config. Defaults to ``configs/base.yaml``.
        If that file is missing, falls back to pure defaults + env vars.
    """
    yaml_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if yaml_path.exists():
        return AppConfig.from_yaml(yaml_path)
    return AppConfig()


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "PROJECT_ROOT",
    "AppConfig",
    "DatasetConfig",
    "LoggingConfig",
    "MissingGroupStrategy",
    "MissingStrategyConfig",
    "PathsConfig",
    "PreprocessingConfig",
    "ProjectConfig",
    "SplitsConfig",
    "load_config",
]
