"""Similar-case bank for the GraphRAG retrieval tool (Phase 5).

Backed by FAISS when ``faiss-cpu`` is installed *and* a populated index is
on disk; otherwise falls back to in-memory cosine similarity over a small
seed bank. Either way the call shape is the same.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Seed bank -- 6 prototypical fraud cases, one per pattern. Hard-coded so
# tests + cold-laptop runs always have something to retrieve.
# ---------------------------------------------------------------------------


@dataclass
class CaseRecord:
    case_id: str
    pattern: str
    summary: str
    embedding: np.ndarray = field(repr=False)


def _seed_bank(dim: int = 64) -> list[CaseRecord]:
    """Six deterministic synthetic cases covering each canonical pattern.

    Embeddings are generated from a per-case RNG so similarity scores are
    reproducible across runs.
    """
    seeds = [
        (
            "CASE-001",
            "velocity_spike",
            "Card swiped 24x within 1 hour at unrelated merchants; spend $4,210.",
        ),
        ("CASE-002", "card_testing", "67 micro-charges (<=$2) across 12 merchants in 30 minutes."),
        (
            "CASE-003",
            "collusion_ring",
            "5 cards sharing a device + address; coordinated tx within 6 hours.",
        ),
        (
            "CASE-004",
            "account_takeover",
            "New device + new shipping address + 4x average ticket; preceded by password reset.",
        ),
        (
            "CASE-005",
            "geo_anomaly",
            "Charge originating from country B 2 hours after a charge from country A.",
        ),
        (
            "CASE-006",
            "none",
            "Standard recurring charge from a long-trusted card; flagged by velocity heuristic.",
        ),
    ]
    cases: list[CaseRecord] = []
    for i, (cid, pat, summary) in enumerate(seeds):
        rng = np.random.default_rng(seed=42 + i)
        vec = rng.normal(loc=0.0, scale=1.0, size=(dim,)).astype(np.float32)
        # Normalise to unit length so cosine == dot product.
        n = np.linalg.norm(vec) or 1.0
        cases.append(CaseRecord(case_id=cid, pattern=pat, summary=summary, embedding=vec / n))
    return cases


# ---------------------------------------------------------------------------
# CaseBank (in-memory + optional FAISS)
# ---------------------------------------------------------------------------


class CaseBank:
    """K-NN retrieval over historical fraud cases.

    Use :meth:`add` to grow the bank, :meth:`search` to find the top-k
    most similar cases to a query embedding. ``embedding_dim`` is
    inferred from the first record added, or set explicitly for the seed.
    """

    def __init__(self, *, embedding_dim: int = 64, use_faiss: bool = True) -> None:
        self.embedding_dim = embedding_dim
        self._records: list[CaseRecord] = []
        self._faiss_index: Any | None = None
        self._use_faiss = use_faiss

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def with_seed(cls, *, embedding_dim: int = 64, use_faiss: bool = True) -> CaseBank:
        bank = cls(embedding_dim=embedding_dim, use_faiss=use_faiss)
        for rec in _seed_bank(embedding_dim):
            bank.add(rec)
        return bank

    def add(self, record: CaseRecord) -> None:
        if record.embedding.shape[0] != self.embedding_dim:
            # Re-shape the dim on the first add, otherwise reject mismatches.
            if not self._records:
                self.embedding_dim = int(record.embedding.shape[0])
            else:
                raise ValueError(
                    f"embedding shape mismatch: got {record.embedding.shape[0]}, "
                    f"expected {self.embedding_dim}"
                )
        self._records.append(record)
        # Rebuild FAISS lazily on first ``search`` call.
        self._faiss_index = None

    def __len__(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "embedding_dim": self.embedding_dim,
                    "records": [
                        {
                            "case_id": r.case_id,
                            "pattern": r.pattern,
                            "summary": r.summary,
                            "embedding": r.embedding,
                        }
                        for r in self._records
                    ],
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> CaseBank:
        import pickle

        with Path(path).open("rb") as f:
            data = pickle.load(f)
        bank = cls(embedding_dim=int(data["embedding_dim"]))
        for rec in data["records"]:
            bank.add(
                CaseRecord(
                    case_id=rec["case_id"],
                    pattern=rec["pattern"],
                    summary=rec["summary"],
                    embedding=np.asarray(rec["embedding"], dtype=np.float32),
                )
            )
        return bank

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------

    def search(self, query: np.ndarray, *, k: int = 3) -> list[tuple[CaseRecord, float]]:
        if not self._records:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.embedding_dim:
            # If the query is the wrong size (e.g. zero-vector cold path),
            # zero-pad / truncate for graceful behaviour.
            adjusted = np.zeros(self.embedding_dim, dtype=np.float32)
            n = min(q.shape[0], self.embedding_dim)
            adjusted[:n] = q[:n]
            q = adjusted
        n = np.linalg.norm(q) or 1.0
        q = q / n

        # Try FAISS path first.
        if self._use_faiss and self._faiss_index is None:
            self._faiss_index = self._maybe_build_faiss()
        if self._faiss_index is not None:
            try:
                D, idx = self._faiss_index.search(q.reshape(1, -1).astype(np.float32), k)
                pairs = [
                    (self._records[int(i)], float(D[0][rank]))
                    for rank, i in enumerate(idx[0])
                    if 0 <= int(i) < len(self._records)
                ]
                return pairs[:k]
            except Exception as exc:
                log.debug("faiss_search_failed_falling_back", error=str(exc))
                # Fall through to numpy path

        # Numpy fallback.
        mat = np.stack([r.embedding for r in self._records])
        sims = mat @ q  # cosine since both sides are unit-normalised
        order = np.argsort(sims)[::-1][:k]
        return [(self._records[int(i)], float(sims[int(i)])) for i in order]

    def _maybe_build_faiss(self) -> Any | None:
        if not self._use_faiss:
            return None
        try:
            import faiss  # type: ignore[import-untyped]
        except Exception as exc:
            log.debug("faiss_not_installed", error=str(exc))
            return None
        try:
            mat = np.stack([r.embedding for r in self._records]).astype(np.float32)
            index = faiss.IndexFlatIP(self.embedding_dim)
            index.add(mat)
            log.info("faiss_index_built", n=len(self._records), dim=self.embedding_dim)
            return index
        except Exception as exc:
            log.warning("faiss_index_build_failed", error=str(exc))
            return None


__all__ = ["CaseBank", "CaseRecord"]
