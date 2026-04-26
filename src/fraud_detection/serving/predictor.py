"""End-to-end fraud-prediction service.

This is the production hot path -- everything between an inbound HTTP
request and the JSON :class:`FraudPrediction` response. The Phase 4 plan
allocates the latency budget like this::

    [1] Feast get_online_features(card_id)              ~2 ms
    [2] Compute real-time temporal features            ~5 ms
    [3] Get cached GNN embedding from Redis            ~1 ms
    [4] XGBoost predict_proba                          ~2 ms
    [5] SHAP explanation                              ~10 ms
    -----------------------------------------------------------
    target P95                                        <50 ms

The :class:`FraudPredictor` is built around the :class:`FraudEnsemble`
artifact bundle from Phase 3. It assembles the input row from the request
+ pre-warmed graph context, slots in the cached card embedding, and falls
back gracefully when any of (Redis, SHAP, real graph) aren't available.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fraud_detection.models import FraudEnsemble
from fraud_detection.serving.redis_cache import EmbeddingCache
from fraud_detection.serving.schemas import (
    ALERT_THRESHOLD,
    FeatureContribution,
    FraudPrediction,
    TransactionRequest,
    risk_level,
)
from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass for hot-path use (no Pydantic overhead)
# ---------------------------------------------------------------------------


@dataclass
class _PredictionResult:
    proba: float
    embedding_used_cache: bool
    contributions: list[FeatureContribution]
    timings: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FraudPredictor
# ---------------------------------------------------------------------------


class FraudPredictor:
    """Stateful predictor that owns the model + caches.

    Build at server startup, then call :meth:`predict_one` per request and
    :meth:`predict_batch` for bulk endpoints.

    Parameters
    ----------
    ensemble
        A trained :class:`FraudEnsemble` (preloaded GNN + XGBoost).
    embedding_cache
        Pre-connected :class:`EmbeddingCache`. The predictor will hit
        cache for the request's ``card1`` id; on miss it computes the
        embedding via a full GNN forward and writes it back.
    feature_columns
        The exact list of tabular columns expected by the XGBoost stage.
        Anything missing on a request is filled with 0; anything extra
        is dropped. This makes the API tolerant to schema drift without
        losing safety.
    threshold
        Score above which we mark ``is_fraud_predicted=True``.
    enable_shap
        Toggle the SHAP explainer. Skipped automatically if ``shap``
        isn't importable.
    shap_max_display
        How many top features to include in the response.
    graph_data
        Optional reference :class:`HeteroData` (Phase 2). When supplied
        and a card_id is *known* we route through the GNN branch using
        the pre-warmed embedding; when not supplied (e.g. unit tests on
        cold cards) we fall back to a zero embedding.
    """

    def __init__(
        self,
        *,
        ensemble: FraudEnsemble,
        embedding_cache: EmbeddingCache,
        feature_columns: list[str],
        threshold: float = ALERT_THRESHOLD,
        enable_shap: bool = True,
        shap_max_display: int = 5,
        graph_data: Any | None = None,
        card_id_to_embedding_index: dict[int | str, int] | None = None,
        model_version: str = "v0.3.0-gnn-model",
    ) -> None:
        self.ensemble = ensemble
        self.cache = embedding_cache
        self.feature_columns = list(feature_columns)
        self.tabular_columns = [c for c in self.feature_columns if not c.startswith("gnn_emb_")]
        self.embedding_dim = len(self.feature_columns) - len(self.tabular_columns)
        self.threshold = float(threshold)
        self.shap_max_display = int(shap_max_display)
        self.graph_data = graph_data
        self.card_id_to_embedding_index = card_id_to_embedding_index or {}
        self.model_version = model_version

        # Lazy SHAP setup.
        self._shap_explainer = None
        self._enable_shap = enable_shap
        if enable_shap:
            self._shap_explainer = self._try_build_shap_explainer()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _try_build_shap_explainer(self) -> Any | None:
        try:
            import shap

            booster = self.ensemble.xgb.model
            explainer = shap.TreeExplainer(booster)
            log.info("shap_explainer_ready")
            return explainer
        except Exception as exc:
            log.warning("shap_disabled", error=str(exc))
            return None

    def _resolve_embedding(self, card_id: int | str | None) -> tuple[np.ndarray, bool]:
        """Return the 64-d embedding to feed the XGBoost stage.

        Strategy (in order):
        1. If ``card_id`` is in the cache -> use it.
        2. If ``card_id`` is known to the graph -> compute via a single
           GNN forward and warm the cache.
        3. Otherwise -> zero embedding (cold-card fallback).
        """
        if card_id is None:
            return np.zeros(self.embedding_dim, dtype=np.float32), False

        cached = self.cache.get(card_id)
        if cached is not None and cached.shape[0] == self.embedding_dim:
            return cached, True

        if (
            self.graph_data is not None
            and card_id in self.card_id_to_embedding_index
            and self.embedding_dim > 0
        ):
            idx = self.card_id_to_embedding_index[card_id]
            with torch.no_grad():
                emb = self.ensemble.gnn.get_embeddings(self.graph_data)[idx]
            vec = emb.cpu().numpy().astype(np.float32)
            self.cache.set(card_id, vec)
            return vec, False

        return np.zeros(self.embedding_dim, dtype=np.float32), False

    def _build_tabular_row(self, request: TransactionRequest) -> np.ndarray:
        """Pull request fields + extras into the columns XGBoost was trained on.

        Missing values become 0; unknown fields are silently dropped. We
        do not run the full Phase 1 preprocessor here -- the request is
        assumed to be downstream of upstream feature engineering. This
        keeps the hot path light enough for the 50ms budget.
        """
        # Start with a zero-filled vector indexed by the tabular column order.
        row = np.zeros(len(self.tabular_columns), dtype=np.float32)
        if not self.tabular_columns:
            return row

        # Map request fields (and extras) onto the trained columns.
        # ``model_dump`` (with by_alias) gives us aliased keys like
        # ``P_emaildomain`` so they line up with IEEE-CIS names.
        payload = request.model_dump(by_alias=True)
        # Add common shorthand mappings.
        payload.setdefault("TransactionAmt", request.transaction_amt)
        payload.setdefault("TransactionDT", request.transaction_dt)
        payload.setdefault("ProductCD", request.product_cd)
        payload.setdefault("isFraud", 0)

        col_to_idx = {c: i for i, c in enumerate(self.tabular_columns)}
        for k, v in payload.items():
            idx = col_to_idx.get(k)
            if idx is None or v is None:
                continue
            try:
                row[idx] = float(v)
            except (TypeError, ValueError):
                # Non-numeric field (e.g. raw email string) -- caller should
                # have integer-encoded these already; we drop them silently.
                continue
        return row

    def _shap_contributions(self, x: np.ndarray) -> list[FeatureContribution]:
        if self._shap_explainer is None:
            return []
        try:
            sv = self._shap_explainer.shap_values(x.reshape(1, -1))
            sv = np.asarray(sv).reshape(-1)
        except Exception as exc:
            log.debug("shap_failed", error=str(exc))
            return []
        order = np.argsort(np.abs(sv))[::-1][: self.shap_max_display]
        return [
            FeatureContribution(
                feature=self.feature_columns[i],
                value=float(x[i]),
                contribution=float(sv[i]),
            )
            for i in order
        ]

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def predict_one(self, request: TransactionRequest) -> FraudPrediction:
        timings: dict[str, float] = {}
        t_total_start = time.perf_counter()

        # --- 1. Embedding (cache or GNN) ----------------------------------
        t = time.perf_counter()
        emb, _used_cache = self._resolve_embedding(request.card1)
        timings["embedding_ms"] = (time.perf_counter() - t) * 1000

        # --- 2. Tabular row -----------------------------------------------
        t = time.perf_counter()
        tab = self._build_tabular_row(request)
        timings["tabular_ms"] = (time.perf_counter() - t) * 1000

        # --- 3. Stack + XGBoost predict ------------------------------------
        t = time.perf_counter()
        x = np.concatenate([emb, tab]).astype(np.float32, copy=False)
        proba = float(self.ensemble.xgb.predict_proba(x.reshape(1, -1))[0])
        timings["xgboost_ms"] = (time.perf_counter() - t) * 1000

        # --- 4. SHAP explanation (optional, guarded) -----------------------
        t = time.perf_counter()
        contribs = self._shap_contributions(x) if self._enable_shap else []
        timings["shap_ms"] = (time.perf_counter() - t) * 1000

        timings["total_ms"] = (time.perf_counter() - t_total_start) * 1000

        return FraudPrediction(
            transaction_id=request.transaction_id,
            fraud_probability=proba,
            fraud_score=proba,  # already calibrated by XGBoost
            risk_level=risk_level(proba),
            is_fraud_predicted=proba >= self.threshold,
            threshold=self.threshold,
            top_features=contribs,
            latency_ms=timings,
            model_version=self.model_version,
        )

    def predict_batch(self, requests: Iterable[TransactionRequest]) -> list[FraudPrediction]:
        # Batch the XGBoost path to amortise overhead. Embedding lookup
        # remains per-row because each card may live in a different
        # cache shard, but XGBoost benefits from a (B, F) call.
        rows: list[np.ndarray] = []
        emb_used: list[bool] = []
        emb_timings: list[float] = []
        tab_timings: list[float] = []
        ids: list[int | str] = []
        thresholds: list[float] = []
        request_list = list(requests)

        for r in request_list:
            t0 = time.perf_counter()
            emb, used = self._resolve_embedding(r.card1)
            emb_timings.append((time.perf_counter() - t0) * 1000)
            t1 = time.perf_counter()
            tab = self._build_tabular_row(r)
            tab_timings.append((time.perf_counter() - t1) * 1000)
            rows.append(np.concatenate([emb, tab]))
            emb_used.append(used)
            ids.append(r.transaction_id)
            thresholds.append(self.threshold)

        if not rows:
            return []

        X = np.vstack(rows).astype(np.float32, copy=False)
        t_x = time.perf_counter()
        probas = self.ensemble.xgb.predict_proba(X)
        xgb_ms = (time.perf_counter() - t_x) * 1000

        out: list[FraudPrediction] = []
        for i, (r, p) in enumerate(zip(request_list, probas, strict=True)):
            t_s = time.perf_counter()
            contribs = self._shap_contributions(X[i]) if self._enable_shap else []
            shap_ms = (time.perf_counter() - t_s) * 1000
            out.append(
                FraudPrediction(
                    transaction_id=r.transaction_id,
                    fraud_probability=float(p),
                    fraud_score=float(p),
                    risk_level=risk_level(float(p)),
                    is_fraud_predicted=float(p) >= self.threshold,
                    threshold=self.threshold,
                    top_features=contribs,
                    latency_ms={
                        "embedding_ms": emb_timings[i],
                        "tabular_ms": tab_timings[i],
                        "xgboost_ms": xgb_ms / max(len(rows), 1),
                        "shap_ms": shap_ms,
                    },
                    model_version=self.model_version,
                )
            )
        return out

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "n_features": len(self.feature_columns),
            "embedding_dim": self.embedding_dim,
            "shap_enabled": self._shap_explainer is not None,
            "graph_attached": self.graph_data is not None,
            "cache_stats": self.cache.stats(),
        }


# ---------------------------------------------------------------------------
# Loader: build a FraudPredictor from disk
# ---------------------------------------------------------------------------


def load_predictor(
    *,
    ensemble_dir: str | Path = "data/models/ensemble",
    redis_url: str | None = None,
    enable_shap: bool = True,
    threshold: float = ALERT_THRESHOLD,
) -> FraudPredictor:
    """Build a predictor from a trained ensemble directory.

    Used by ``serving/app.py`` at startup.
    """
    ensemble = FraudEnsemble.load(ensemble_dir)
    cache = EmbeddingCache(url=redis_url)
    cache.connect()
    return FraudPredictor(
        ensemble=ensemble,
        embedding_cache=cache,
        feature_columns=ensemble.feature_columns,
        threshold=threshold,
        enable_shap=enable_shap,
    )


__all__ = ["FraudPredictor", "load_predictor"]
