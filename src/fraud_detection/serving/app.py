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
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from fraud_detection.monitoring import MonitoringState, monitoring_metrics, report_to_html
from fraud_detection.monitoring import get_state as get_monitoring_state
from fraud_detection.serving.middleware import RequestTimingMiddleware, metrics
from fraud_detection.serving.predictor import FraudPredictor, load_predictor
from fraud_detection.serving.schemas import (
    ALERT_THRESHOLD,
    BatchPredictRequest,
    BatchPredictResponse,
    FraudAlert,
    FraudPrediction,
    HealthStatus,
    InvestigationRequest,
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

    # Phase 5 -- LangGraph investigator. Both are ``None`` if the agent
    # extras aren't installed; the ``/api/v1/investigate`` endpoint then
    # returns 503.
    agent_deps: Any | None
    agent_compiled: Any | None

    # Phase 6 -- in-memory rolling history so the dashboard can populate
    # its initial view without forcing every demo to run a long replay.
    recent_predictions: deque[FraudPrediction]
    recent_alerts: deque[FraudAlert]

    # Phase 7 -- monitoring state holder; ``MonitoringState`` owns the
    # production performance tracker, alert evaluator, and the last drift
    # report. The same singleton is also reachable from the CLI.
    monitoring: MonitoringState

    def __init__(self) -> None:
        self.predictor = None
        self.producer = None
        self.consumer = None
        self.broadcaster = AlertBroadcaster()
        self.started_at = time.time()
        self.settings = {}
        self.agent_deps = None
        self.agent_compiled = None
        self.recent_predictions = deque(maxlen=200)
        self.recent_alerts = deque(maxlen=200)
        self.monitoring = get_monitoring_state()


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

    # Phase 5 -- compile the LangGraph investigator if extras are installed.
    if os.environ.get("FRAUD_AGENT_DISABLED", "false").lower() != "true":
        try:
            from fraud_detection.agent import AgentDeps, build_graph

            # Optional: wire a Neo4j-backed graph traversal if NEO4J_URI is set.
            neo4j_graph = None
            if os.environ.get("NEO4J_URI"):
                try:
                    from fraud_detection.agent import Neo4jGraphAdapter

                    adapter = Neo4jGraphAdapter()
                    if adapter.connect():
                        neo4j_graph = adapter
                        log.info("agent_neo4j_attached", uri=adapter.uri)
                    else:
                        log.warning("agent_neo4j_unreachable_skipping")
                except Exception as exc:
                    log.warning("agent_neo4j_init_failed", error=str(exc))

            # Optional: try a real Ollama daemon when OLLAMA_BASE_URL is set;
            # otherwise default to the deterministic stub.
            from fraud_detection.agent import get_llm

            llm = get_llm(prefer_ollama=bool(os.environ.get("OLLAMA_BASE_URL")))

            state.agent_deps = AgentDeps(llm=llm, graph=neo4j_graph)
            state.agent_compiled = build_graph(state.agent_deps)
            log.info(
                "agent_ready",
                llm=getattr(state.agent_deps.llm, "name", "stub"),
                neo4j=neo4j_graph is not None,
            )
        except Exception as exc:
            log.warning("agent_unavailable", error=str(exc))

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
        version="v0.7.0-mlops",
        description="Real-time GNN+XGBoost fraud-detection inference API.",
        lifespan=lifespan,
    )
    # Phase 6 -- allow the React dashboard (Vite dev server on :5173, plus
    # any operator-supplied origins) to call the API. We default to a
    # sensible local-dev set; override with ``FRAUD_CORS_ORIGINS=*`` for
    # an open dev box, or a comma-separated list for production.
    cors_origins = os.environ.get(
        "FRAUD_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    origins = [o.strip() for o in cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
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
        # Route through the shadow lane when a challenger is attached -- the
        # champion result is identical to a direct ``predict_one`` call,
        # but the challenger scores in the background.
        if s.monitoring.shadow is not None and s.monitoring.shadow.enabled:
            result = s.monitoring.shadow.score(payload)
        else:
            result = s.predictor.predict_one(payload)
        metrics.predictions_total.labels(result.risk_level).inc()
        s.recent_predictions.append(result)
        s.monitoring.performance.record_prediction(
            result.transaction_id, result.fraud_score, timestamp=result.served_at
        )
        if result.fraud_score >= s.predictor.threshold and s.producer is not None:
            alert = _to_alert(result, payload)
            s.producer.publish(alert)
            if not s.producer.is_kafka and s.consumer is not None:
                s.consumer.push_in_memory(alert)
            s.recent_alerts.append(alert)
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
            s.recent_predictions.append(r)
            s.monitoring.performance.record_prediction(
                r.transaction_id, r.fraud_score, timestamp=r.served_at
            )
            if r.fraud_score >= s.predictor.threshold and s.producer is not None:
                alert = _to_alert(r, tx)
                s.producer.publish(alert)
                if not s.producer.is_kafka and s.consumer is not None:
                    s.consumer.push_in_memory(alert)
                s.recent_alerts.append(alert)
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

    # ---- Recent buffer (Phase 6 dashboard first-load) ----------------------

    @app.get("/api/v1/recent", tags=["dashboard"])
    async def recent(request: Request, limit: int = 100) -> dict[str, Any]:
        s = _state(request)
        cap = max(1, min(int(limit), 200))
        preds = list(s.recent_predictions)[-cap:][::-1]
        alerts = list(s.recent_alerts)[-cap:][::-1]
        return {
            "predictions": [p.model_dump(mode="json") for p in preds],
            "alerts": [a.model_dump(mode="json") for a in alerts],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ---- Prometheus metrics -------------------------------------------------

    @app.get("/api/v1/metrics", tags=["system"])
    async def metrics_endpoint() -> Response:
        return Response(content=metrics.render(), media_type=metrics.content_type)

    # ---- Monitoring (Phase 7) ----------------------------------------------

    @app.get("/api/v1/monitoring/drift", tags=["monitoring"])
    async def drift_endpoint(request: Request) -> dict[str, Any]:
        s = _state(request)
        report = s.monitoring.last_drift_report
        if report is None:
            return {
                "status": "no_report",
                "message": "No drift report has been computed yet. Run `make drift-report`.",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        return report.to_dict()

    @app.get("/api/v1/monitoring/drift.html", response_class=HTMLResponse, tags=["monitoring"])
    async def drift_html_endpoint(request: Request) -> HTMLResponse:
        s = _state(request)
        report = s.monitoring.last_drift_report
        if report is None:
            return HTMLResponse(
                "<!doctype html><meta charset='utf-8'><title>No drift report</title>"
                "<body style='font-family:system-ui;padding:24px;'>"
                "<h1>No drift report</h1>"
                "<p>Run <code>make drift-report</code> to generate one.</p>"
                "</body>",
                status_code=200,
            )
        return HTMLResponse(report_to_html(report))

    @app.get("/api/v1/monitoring/performance", tags=["monitoring"])
    async def performance_endpoint(
        request: Request, window_minutes: int | None = None
    ) -> dict[str, Any]:
        from datetime import timedelta as _td

        s = _state(request)
        window = _td(minutes=int(window_minutes)) if window_minutes else None
        snap = s.monitoring.performance.snapshot(window=window)
        monitoring_metrics.update_performance(snap)
        return snap.to_dict()

    @app.post("/api/v1/monitoring/label", tags=["monitoring"])
    async def label_endpoint(
        request: Request,
        payload: dict[str, Any] = Body(...),  # noqa: B008
    ) -> dict[str, Any]:
        """Attach a ground-truth label to a previously-scored transaction.

        Expected body: ``{"transaction_id": "<id>", "label": 0 | 1}``.
        """
        s = _state(request)
        tx_id = payload.get("transaction_id")
        label = payload.get("label")
        if tx_id is None or label is None:
            raise HTTPException(
                status_code=400,
                detail="Request body must include 'transaction_id' and 'label'.",
            )
        try:
            label_int = int(label)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail="'label' must be coercible to int (0 or 1)."
            ) from exc
        if label_int not in (0, 1):
            raise HTTPException(status_code=400, detail="'label' must be 0 or 1.")
        found = s.monitoring.performance.record_label(tx_id, label_int)
        # Refresh the Prometheus gauges so the dashboard surfaces the change.
        snap = s.monitoring.performance.snapshot()
        monitoring_metrics.update_performance(snap)
        return {"transaction_id": tx_id, "label": label_int, "found": found}

    @app.get("/api/v1/monitoring/alerts", tags=["monitoring"])
    async def alerts_endpoint(request: Request) -> dict[str, Any]:
        s = _state(request)
        drift = s.monitoring.last_drift_report
        snap = s.monitoring.performance.snapshot()
        n_predictions = len(s.recent_predictions)
        n_alerts = sum(
            1 for p in s.recent_predictions if p.fraud_score >= s.monitoring.performance.threshold
        )
        production_fraud_rate = n_alerts / n_predictions if n_predictions > 0 else 0.0
        shadow_summary = s.monitoring.shadow.summary() if s.monitoring.shadow is not None else None
        ctx: dict[str, Any] = {
            "model_loaded": s.predictor is not None,
            "drift_overall": float(drift.overall_psi) if drift is not None else 0.0,
            "drift_severe": int(drift.n_severe) if drift is not None else 0,
            "performance": snap,
            "production_fraud_rate": production_fraud_rate,
            "reference_fraud_rate": s.monitoring.reference_fraud_rate,
            "shadow": shadow_summary,
            "latency_p95": 0.0,
            "error_rate": 0.0,
        }
        fired = s.monitoring.alerts.evaluate(ctx)
        return {
            "n_active": len(fired),
            "alerts": [a.to_dict() for a in fired],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/api/v1/monitoring/shadow", tags=["monitoring"])
    async def shadow_summary_endpoint(request: Request) -> dict[str, Any]:
        s = _state(request)
        if s.monitoring.shadow is None:
            return {
                "status": "disabled",
                "message": "Shadow deployment not configured.",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        summary = s.monitoring.shadow.summary()
        monitoring_metrics.update_shadow_summary(summary)
        return summary.to_dict()

    @app.get("/api/v1/monitoring/shadow/recent", tags=["monitoring"])
    async def shadow_recent_endpoint(request: Request, limit: int = 50) -> dict[str, Any]:
        s = _state(request)
        if s.monitoring.shadow is None:
            return {"status": "disabled", "decisions": []}
        decisions = s.monitoring.shadow.recent_decisions(limit=max(1, min(int(limit), 500)))
        return {
            "n": len(decisions),
            "decisions": [d.to_dict() for d in decisions],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ---- Investigate (Phase 5) ----------------------------------------------

    @app.post("/api/v1/investigate", tags=["investigate"])
    async def investigate_endpoint(
        request: Request,
        payload: InvestigationRequest = Body(...),  # noqa: B008 -- FastAPI idiom
    ) -> dict[str, Any]:
        s = _state(request)
        if s.agent_compiled is None or s.agent_deps is None:
            raise HTTPException(status_code=503, detail="Agent not available")

        # Resolve a FraudPrediction. If the caller didn't supply one we
        # require a transaction + a loaded predictor.
        prediction = payload.prediction
        tx = payload.transaction
        if prediction is None:
            if s.predictor is None or tx is None:
                raise HTTPException(
                    status_code=400,
                    detail="Either 'prediction' or 'transaction' (with model loaded) is required.",
                )
            prediction = s.predictor.predict_one(tx)

        from fraud_detection.agent import investigate, new_state

        request_payload = (
            tx.model_dump(by_alias=True)
            if tx is not None
            else {"transaction_id": prediction.transaction_id}
        )
        state = new_state(
            transaction_id=prediction.transaction_id,
            prediction=prediction,
            request=request_payload,
            alert_id=payload.alert_id,
        )
        t_agent = time.perf_counter()
        agent_status = "ok"
        try:
            report = investigate(state, deps=s.agent_deps, compiled=s.agent_compiled)
        except Exception:
            agent_status = "error"
            monitoring_metrics.record_agent_run(
                risk_level=str(prediction.risk_level),
                latency_seconds=time.perf_counter() - t_agent,
                status=agent_status,
            )
            raise
        tool_calls = [
            (str(c.get("name", "?")), str(c.get("status", "ok"))) for c in report.tool_calls
        ]
        monitoring_metrics.record_agent_run(
            risk_level=str(report.risk_level),
            latency_seconds=time.perf_counter() - t_agent,
            status=agent_status,
            tool_calls=tool_calls,
        )
        metrics.predictions_total.labels(report.risk_level).inc()
        return report.model_dump(mode="json")

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
