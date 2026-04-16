"""Fraud detection with heterogeneous GNNs and agentic AI investigation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fraud-detection-gnn")
except PackageNotFoundError:  # pragma: no cover -- editable install without metadata
    __version__ = "0.1.0"

__all__ = ["__version__"]
