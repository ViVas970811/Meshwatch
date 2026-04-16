"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Deterministic RNG for all synthetic fixtures.
_RNG = np.random.default_rng(42)


@pytest.fixture
def synthetic_ieee_df() -> pd.DataFrame:
    """A small, schema-faithful synthetic IEEE-CIS DataFrame.

    Includes at least one column from every feature family the preprocessor
    touches: V, D, C, M, id_numeric, id_categorical, email, device, product,
    card, addr.
    """
    n = 500
    t0 = 1_000_000  # arbitrary TransactionDT start (seconds)
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "isFraud": _RNG.choice([0, 1], size=n, p=[0.95, 0.05]),
            "TransactionDT": t0 + np.sort(_RNG.integers(0, 60 * 60 * 24 * 30, size=n)),
            "TransactionAmt": np.round(
                np.abs(_RNG.normal(loc=75, scale=40, size=n)).clip(min=1.0), 2
            ),
            "ProductCD": _RNG.choice(list("WCHSR"), size=n),
            "card1": _RNG.integers(1000, 20_000, size=n),
            "card4": _RNG.choice(["visa", "mastercard", "amex", np.nan], size=n),
            "card6": _RNG.choice(["debit", "credit", np.nan], size=n),
            "addr1": _RNG.integers(100, 500, size=n).astype(float),
            "addr2": _RNG.integers(10, 99, size=n).astype(float),
            "P_emaildomain": _RNG.choice(
                ["gmail.com", "yahoo.com", "hotmail.com", np.nan], size=n, p=[0.5, 0.2, 0.2, 0.1]
            ),
            "R_emaildomain": _RNG.choice(
                ["gmail.com", "outlook.com", np.nan], size=n, p=[0.4, 0.2, 0.4]
            ),
            # V features -- V1 mostly present, V99 mostly missing, V250 fully missing
            "V1": _RNG.normal(size=n),
            "V99": np.where(_RNG.random(n) < 0.8, np.nan, _RNG.normal(size=n)),
            "V250": np.full(n, np.nan),
            # D features -- timedeltas w/ some missing
            "D1": np.where(
                _RNG.random(n) < 0.3, np.nan, _RNG.integers(0, 365, size=n).astype(float)
            ),
            "D2": np.where(
                _RNG.random(n) < 0.8, np.nan, _RNG.integers(0, 365, size=n).astype(float)
            ),
            # C features -- counts, mostly present
            "C1": _RNG.integers(0, 10, size=n).astype(float),
            "C2": np.where(
                _RNG.random(n) < 0.02, np.nan, _RNG.integers(0, 10, size=n).astype(float)
            ),
            # M features -- match flags, string-valued
            "M1": _RNG.choice(["T", "F", np.nan], size=n, p=[0.4, 0.4, 0.2]),
            "M2": _RNG.choice(["T", "F", np.nan], size=n, p=[0.3, 0.3, 0.4]),
            # Identity table (joined) -- numeric id_01..id_03 + categorical id_30..id_31
            "id_01": np.where(_RNG.random(n) < 0.7, np.nan, _RNG.normal(size=n)),
            "id_02": np.where(
                _RNG.random(n) < 0.7, np.nan, _RNG.integers(0, 1000, size=n).astype(float)
            ),
            "id_30": _RNG.choice(
                ["Windows 10", "iOS", "Android", np.nan], size=n, p=[0.3, 0.2, 0.2, 0.3]
            ),
            "id_31": _RNG.choice(
                ["chrome 70.0", "safari", "firefox", np.nan], size=n, p=[0.3, 0.2, 0.2, 0.3]
            ),
            "DeviceType": _RNG.choice(["desktop", "mobile", np.nan], size=n, p=[0.4, 0.4, 0.2]),
            "DeviceInfo": _RNG.choice(["Windows", "iPhone", "SM-G960U", np.nan], size=n),
        }
    )
    return df


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """Redirect the config's data paths to a temp dir for isolation."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    splits = tmp_path / "splits"
    graphs = tmp_path / "graphs"
    for p in (raw, processed, splits, graphs):
        p.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("FRAUD_PATHS__DATA_RAW", str(raw))
    monkeypatch.setenv("FRAUD_PATHS__DATA_PROCESSED", str(processed))
    monkeypatch.setenv("FRAUD_PATHS__DATA_SPLITS", str(splits))
    monkeypatch.setenv("FRAUD_PATHS__DATA_GRAPHS", str(graphs))
    return {"raw": raw, "processed": processed, "splits": splits, "graphs": graphs}
