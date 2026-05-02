import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, api, wsUrl } from "@/api/client";

describe("api client", () => {
  beforeEach(() => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.endsWith("/api/v1/health")) {
        return new Response(JSON.stringify({ status: "ok", model_loaded: true }), {
          status: 200,
        });
      }
      if (url.endsWith("/api/v1/recent")) {
        return new Response(JSON.stringify({ predictions: [], alerts: [], generated_at: "x" }), {
          status: 200,
        });
      }
      if (url.endsWith("/api/v1/predict")) {
        return new Response(JSON.stringify({ detail: "Model not loaded" }), { status: 503 });
      }
      return new Response("", { status: 404 });
    });
  });

  afterEach(() => vi.restoreAllMocks());

  it("health() returns parsed body", async () => {
    const r = await api.health();
    expect(r.model_loaded).toBe(true);
  });

  it("recent() returns the response shape", async () => {
    const r = await api.recent();
    expect(r.predictions).toEqual([]);
    expect(r.alerts).toEqual([]);
  });

  it("non-200 responses raise ApiError with status + body", async () => {
    await expect(api.predict({ transaction_id: 1, transaction_dt: 0, transaction_amt: 0 }))
      .rejects.toThrow(ApiError);
    try {
      await api.predict({ transaction_id: 1, transaction_dt: 0, transaction_amt: 0 });
    } catch (e) {
      expect((e as ApiError).status).toBe(503);
      expect((e as ApiError).body).toEqual({ detail: "Model not loaded" });
    }
  });
});

describe("wsUrl()", () => {
  it("derives ws:// from current origin when no base set", () => {
    Object.defineProperty(window, "location", {
      value: { protocol: "http:", host: "localhost:5173" },
      configurable: true,
    });
    expect(wsUrl("/ws/alerts")).toBe("ws://localhost:5173/ws/alerts");
  });

  it("uses wss:// for https origins", () => {
    Object.defineProperty(window, "location", {
      value: { protocol: "https:", host: "demo.example.com" },
      configurable: true,
    });
    expect(wsUrl("/ws/alerts")).toBe("wss://demo.example.com/ws/alerts");
  });
});
