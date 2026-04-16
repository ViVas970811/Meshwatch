#!/usr/bin/env python
"""Convenience wrapper around :mod:`fraud_detection.data.download`.

Run with::

    python scripts/download_data.py [--force] [--config path/to.yaml]

Or via make::

    make download-data
"""

from __future__ import annotations

import sys
from pathlib import Path

# When invoked as ``python scripts/download_data.py`` the package may not be
# on sys.path (if the user hasn't run ``pip install -e .`` yet). Add the src
# dir defensively so the script is self-contained.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fraud_detection.data.download import main  # noqa: E402

if __name__ == "__main__":
    main()
