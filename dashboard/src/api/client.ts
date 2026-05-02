/**
 * Thin fetch wrapper around the FastAPI app.
 *
 * In dev we proxy /api -> http://127.0.0.1:8000 via vite.config.ts, so
 * relative URLs Just Work. Set ``VITE_API_BASE`` to override (e.g. for
 * a Docker compose deployment).
 */

import type {
  FraudPrediction,
  HealthStatus,
  InvestigationReport,
  ModelInfo,
  RecentResponse,
  TransactionRequest,
} from "./types";

const BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/+$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public body?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: {
      "content-type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!res.ok) {
    let body: unknown;
    try {
      body = await res.json();
    } catch {
      // ignore
    }
    throw new ApiError(`${res.status} ${res.statusText}`, res.status, body);
  }
  // 204 -> no body
  if (res.status === 204) return undefined as unknown as T;
  return (await res.json()) as T;
}

export const api = {
  health: () => request<HealthStatus>("/api/v1/health"),
  modelInfo: () => request<ModelInfo>("/api/v1/model/info"),
  predict: (txn: TransactionRequest) =>
    request<FraudPrediction>("/api/v1/predict", {
      method: "POST",
      body: JSON.stringify(txn),
    }),
  predictBatch: (txns: TransactionRequest[]) =>
    request("/api/v1/predict/batch", {
      method: "POST",
      body: JSON.stringify({ transactions: txns }),
    }),
  investigate: (payload: {
    transaction?: TransactionRequest;
    prediction?: FraudPrediction;
    alert_id?: string;
  }) =>
    request<InvestigationReport>("/api/v1/investigate", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  recent: () => request<RecentResponse>("/api/v1/recent"),
};

export const wsUrl = (path: string): string => {
  const base = (import.meta.env.VITE_WS_BASE ?? "").replace(/\/+$/, "");
  if (base) return `${base}${path}`;
  // No explicit base -> derive from current origin (works through the Vite proxy).
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}${path}`;
};
