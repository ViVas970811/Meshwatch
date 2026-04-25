"""FastAPI app for the fraud-detection serving pipeline.

Endpoints (matching Phase 4 of the implementation plan, page 9):

* ``POST /api/v1/predict``        single transaction
* ``POST /api/v1/predict/batch``  up to 100 transactions
* ``GET  /api/v1/health``         liveness + dependency status
* ``GET  /api/v1/model/info``     model version + feature schema
* ``GET  /api/v1/metrics``        Prometheus exposition
* ``WS   /ws/alerts``             real-time fraud-alert stream

Lifespan responsibilities:

* Construct :class:`FraudPredictor` from the trained ensemble bundle.
* Connect Redis (or fall back to in-memory).
* Connect Kafka producer + start the alert-fan-out consumer.
* Wire a per-app :class:`AlertBroadcaster` for the WebSocket endpoint.

The whole stack is built so that *any* dependency missing degrades to a
locally-runnable mode -- great for tests, demos, and CPU-only laptops.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

from fraud_detection.serving.middleware import RequestTimingMiddleware, metrics
from fraud_detection.serving.predictor import FraudPredictor, load_predictor
from fraud_detection.serving.schemas import (
    ALERT_THRESHOLD,
    BatchPredictRequest,
    BatchPredictResponse,
    FraudAlert,
    FraudPrediction,
    HealthStatus,
    ModelInfoResponse,
    TransactionRequest,
)
from fraud_detection.streaming.kafka_consumer import FraudAlertConsumer
from fraud_detection.streaming.kafka_producer import FraudAlertProducer
from fraud_detection.utils.logging import configure_logging, get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# WebSocket fan-out
# ---------------------------------------------------------------------------


class AlertBroadcaster:
    """Maintains a set of connected WebSocket clients + pushes alerts to all."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        log.info("ws_client_connected", n_clients=len(self._clients))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)
        log.info("ws_client_disconnected", n_clients=len(self._clients))

    async def broadcast(self, alert: FraudAlert) -> None:
        # Send synchronously to each socket; drop on failure.
        if not self._clients:
            return
        payload = alert.model_dump(mode="json")
        async with self._lock:
            stale: list[WebSocket] = []
            for ws in self._clients:
                try:
                    await ws.send_json(payload)
                except Exception as exc:
                    log.warning("ws_send_failed", error=str(exc))
                    stale.append(ws)
            for ws in stale:
                self._clients.discard(ws)
        metrics.alerts_total.labels("websocket").inc()

    @property
    def n_clients(self) -> int:
        return len(self._clients)


# ---------------------------------------------------------------------------
# App state container
# ---------------------------------------------------------------------------


class AppState:
    predictor: FraudPredictor | None
    producer: FraudAlertProducer | None
    consumer: FraudAlertConsumer | None
    broadcaster: AlertBroadcaster
    started_at: float
    settings: dict[str, Any]

    def __init__(self) -> None:
        self.predictor = None
        self.producer = None
        self.consumer = None
        self.broadcaster = AlertBroadcaster()
        self.started_at = time.time()
        self.settings = {}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


def _resolve_settings() -> dict[str, Any]:
    return {
        "ensemble_dir": os.environ.get("FRAUD_ENSEMBLE_DIR", "data/models/ensemble"),
        "redis_url": os.environ.get("REDIS_URL"),
        "kafka_bootstrap": os.environ.get("KAFKA_BOOTSTRAP_SERVERS"),
        "kafka_topic": os.environ.get("KAFKA_FRAUD_TOPIC", "fraud_alerts"),
        "alert_threshold": float(os.environ.get("FRAUD_ALERT_THRESHOLD", str(ALERT_THRESHOLD))),
        "enable_shap": os.environ.get("FRAUD_ENABLE_SHAP", "true").lower() != "false",
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(level=os.environ.get("FRAUD_LOG_LEVEL", "INFO"))
    state = AppState()
    state.settings = _resolve_settings()
    log.info("serving_startup", **state.settings)

    # Predictor (best-effort -- the API stays up even if the model is missing,
    # so /health can still report the issue).
    try:
        ensemble_dir = Path(state.settings["ensemble_dir"])
        if ensemble_dir.exists() and (ensemble_dir / "artifacts.pkl").exists():
            state.predictor = load_predictor(
                ensemble_dir=ensemble_dir,
                redis_url=state.settings["redis_url"],
                enable_shap=state.settings["enable_shap"],
                threshold=state.settings["alert_threshold"],
            )
            log.info("predictor_loaded", info=state.predictor.info())
        else:
            log.warning(
                "predictor_unavailable_no_artifacts",
                ensemble_dir=str(ensemble_dir),
            )
    except Exception as exc:
        log.exception("predictor_load_failed", error=str(exc))

    # Kafka producer (degrades to in-memory).
    state.producer = FraudAlertProducer(
        bootstrap_servers=state.settings["kafka_bootstrap"],
        topic=state.settings["kafka_topic"],
    )
    state.producer.connect()

    # Consumer task: drain Kafka (or in-mem) -> fan out to WebSockets.
    state.consumer = FraudAlertConsumer(
        bootstrap_servers=state.settings["kafka_bootstrap"],
        topic=state.settings["kafka_topic"],
        group_id="meshwatch-fanout",
    )
    state.consumer.connect()

    async def _fanout(alert: FraudAlert) -> None:
        await state.broadcaster.broadcast(alert)

    consumer_task = asyncio.create_task(state.consumer.consume_async(_fanout))

    app.state.fraud_app = state
    try:
        yield
    finally:
        log.info("serving_shutdown")
        consumer_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer_task
        if state.consumer:
            state.consumer.stop()
        if state.producer:
            state.producer.close()


# ---------------------------------------------------------------------------
# App + middleware
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(
        title="Meshwatch Fraud Detection",
        version="v0.4.0-serving-pipeline",
        description="Real-time GNN+XGBoost fraud-detection inference API.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestTimingMiddleware)
    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    """Attach every endpoint as a closure over ``request.app.state``.

    Done via a function so :func:`create_app` is the single source of truth
    and tests can build an app without a model on disk.
    """

    def _state(request: Request) -> AppState:
        return request.app.state.fraud_app

    # ---- Predict (single) ---------------------------------------------------

    @app.post("/api/v1/predict", response_model=FraudPrediction, tags=["predict"])
    async def predict(
        request: Request,
        payload: TransactionRequest = Body(...),  # noqa: B008 -- FastAPI idiom
    ) -> FraudPrediction:
        s = _state(request)
        if s.predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        result = s.predictor.predict_one(payload)
        metrics.predictions_total.labels(result.risk_level).inc()
        if result.fraud_score >= s.predictor.threshold and s.producer is not None:
            alert = _to_alert(result, payload)
            s.producer.publish(alert)
            if not s.producer.is_kafka and s.consumer is not None:
                s.consumer.push_in_memory(alert)
            metrics.alerts_total.labels("kafka").inc()
        return result

    # ---- Predict (batch) ----------------------------------------------------

    @app.post("/api/v1/predict/batch", response_model=BatchPredictResponse, tags=["predict"])
    async def predict_batch(
        request: Request,
        payload: BatchPredictRequest = Body(...),  # noqa: B008 -- FastAPI idiom
    ) -> BatchPredictResponse:
        s = _state(request)
        if s.predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        t0 = time.perf_counter()
        results = s.predictor.predict_batch(payload.transactions)
        elapsed = (time.perf_counter() - t0) * 1000
        n_alerts = 0
        for r, tx in zip(results, payload.transactions, strict=True):
            metrics.predictions_total.labels(r.risk_level).inc()
            if r.fraud_score >= s.predictor.threshold and s.producer is not None:
                alert = _to_alert(r, tx)
                s.producer.publish(alert)
                if not s.producer.is_kafka and s.consumer is not None:
                    s.consumer.push_in_memory(alert)
                metrics.alerts_total.labels("kafka").inc()
                n_alerts += 1
        return BatchPredictResponse(
            predictions=results,
            n_processed=len(results),
            n_alerts=n_alerts,
            elapsed_ms=elapsed,
        )

    # ---- Health -------------------------------------------------------------

    @app.get("/api/v1/health", response_model=HealthStatus, tags=["system"])
    async def health(request: Request) -> HealthStatus:
        s = _state(request)
        return HealthStatus(
            status="ok" if s.predictor is not None else "degraded",
            model_loaded=s.predictor is not None,
            redis_connected=(s.predictor is not None and s.predictor.cache.is_redis()),
            kafka_connected=bool(s.producer and s.producer.is_kafka),
            ray_serve_active=os.environ.get("FRAUD_RAY_SERVE", "false") == "true",
            uptime_seconds=time.time() - s.started_at,
        )

    # ---- Model info ---------------------------------------------------------

    @app.get("/api/v1/model/info", response_model=ModelInfoResponse, tags=["system"])
    async def model_info(request: Request) -> ModelInfoResponse:
        s = _state(request)
        if s.predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        ensemble = s.predictor.ensemble
        gnn = ensemble.gnn
        try:
            importance = ensemble.xgb.feature_importance(kind="gain")
            top = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:25])
        except Exception:
            top = {}

        return ModelInfoResponse(
            model_version=s.predictor.model_version,
            n_parameters=gnn.n_parameters(),
            embedding_dim=gnn.embedding_dim,
            n_features=len(s.predictor.feature_columns),
            feature_columns=s.predictor.feature_columns,
            edge_types=[",".join(e) for e in gnn.edge_types],
            node_types=list(gnn.node_types),
            train_metrics={},  # Phase 7 will populate from MLflow
            feature_importance_top_k=top,
        )

    # ---- Prometheus metrics -------------------------------------------------

    @app.get("/api/v1/metrics", tags=["system"])
    async def metrics_endpoint() -> Response:
        return Response(content=metrics.render(), media_type=metrics.content_type)

    # ---- WebSocket alert stream --------------------------------------------

    @app.websocket("/ws/alerts")
    async def ws_alerts(ws: WebSocket) -> None:
        s: AppState = ws.app.state.fraud_app
        await s.broadcaster.connect(ws)
        try:
            while True:
                # Keep the connection alive; real broadcasts come from the
                # consumer task. We just await any client message (ping)
                # and discard.
                await ws.receive_text()
        except WebSocketDisconnect:
            await s.broadcaster.disconnect(ws)
        except Exception as exc:
            log.warning("ws_unexpected_error", error=str(exc))
            await s.broadcaster.disconnect(ws)

    # ---- Root convenience ---------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse({"service": "meshwatch", "version": app.version, "docs": "/docs"})


def _to_alert(result: FraudPrediction, request: TransactionRequest) -> FraudAlert:
    risk = result.risk_level
    if risk == "LOW":
        risk = "MEDIUM"
    return FraudAlert(
        transaction_id=result.transaction_id,
        fraud_score=result.fraud_score,
        risk_level=risk,  # type: ignore[arg-type]
        transaction_amt=request.transaction_amt,
        card_id=request.card1,
        top_features=result.top_features,
    )


# Module-level app for ``uvicorn fraud_detection.serving.app:app``.
app = create_app()


__all__ = ["AlertBroadcaster", "AppState", "app", "create_app", "lifespan"]
