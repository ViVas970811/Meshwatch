"""Tests for the GraphRAG case bank (Phase 5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fraud_detection.agent.case_bank import CaseBank, CaseRecord, _seed_bank


def test_seed_bank_has_six_cases() -> None:
    bank = CaseBank.with_seed()
    assert len(bank) == 6


def test_seed_bank_covers_all_canonical_patterns() -> None:
    bank = CaseBank.with_seed()
    patterns = {r.pattern for r in _seed_bank(bank.embedding_dim)}
    assert {
        "velocity_spike",
        "card_testing",
        "collusion_ring",
        "account_takeover",
        "geo_anomaly",
    } <= patterns


def test_search_returns_top_k() -> None:
    bank = CaseBank.with_seed()
    rng = np.random.default_rng(0)
    q = rng.normal(size=(64,)).astype(np.float32)
    pairs = bank.search(q, k=3)
    assert len(pairs) == 3


def test_search_self_match_is_close_to_one() -> None:
    bank = CaseBank.with_seed(use_faiss=False)  # numpy path
    # The 0th seed embedding is generated with seed=42; rebuild it.
    case0 = _seed_bank(bank.embedding_dim)[0]
    pairs = bank.search(case0.embedding, k=1)
    rec, sim = pairs[0]
    assert rec.case_id == "CASE-001"
    assert sim == pytest.approx(1.0, abs=1e-5)


def test_search_dimension_mismatch_zero_pads() -> None:
    bank = CaseBank.with_seed()
    # 32-dim query -- should not raise.
    q = np.zeros(32, dtype=np.float32)
    out = bank.search(q, k=2)
    assert len(out) == 2


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    bank = CaseBank.with_seed()
    f = tmp_path / "cases.pkl"
    bank.save(f)
    loaded = CaseBank.load(f)
    assert len(loaded) == len(bank)
    assert loaded.embedding_dim == bank.embedding_dim


def test_add_rejects_dim_mismatch_after_first() -> None:
    bank = CaseBank(embedding_dim=8)
    bank.add(
        CaseRecord(
            case_id="x",
            pattern="none",
            summary="",
            embedding=np.zeros(8, dtype=np.float32),
        )
    )
    with pytest.raises(ValueError):
        bank.add(
            CaseRecord(
                case_id="y",
                pattern="none",
                summary="",
                embedding=np.zeros(16, dtype=np.float32),
            )
        )


def test_empty_bank_returns_no_results() -> None:
    bank = CaseBank(embedding_dim=8)
    assert bank.search(np.zeros(8, dtype=np.float32), k=3) == []


def test_numpy_fallback_when_faiss_disabled() -> None:
    bank = CaseBank.with_seed(use_faiss=False)
    rng = np.random.default_rng(7)
    q = rng.normal(size=(64,)).astype(np.float32)
    pairs = bank.search(q, k=2)
    assert len(pairs) == 2
    # Order should be by descending similarity.
    assert pairs[0][1] >= pairs[1][1]
