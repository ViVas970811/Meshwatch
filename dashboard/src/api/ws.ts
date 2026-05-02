/**
 * WebSocket hook for /ws/alerts.
 *
 * Auto-reconnects with exponential backoff. Emits parsed FraudAlert
 * objects to the supplied callback. The hook also exposes the live
 * connection state so headers can render a status pill.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { wsUrl } from "./client";
import type { FraudAlert } from "./types";

export type WSStatus = "connecting" | "open" | "closed";

interface UseAlertSocketOptions {
  /** Called for every successfully decoded FraudAlert. */
  onAlert: (alert: FraudAlert) => void;
  /** Optional override for the path (default ``/ws/alerts``). */
  path?: string;
}

export function useAlertSocket({ onAlert, path = "/ws/alerts" }: UseAlertSocketOptions): {
  status: WSStatus;
  reconnect: () => void;
} {
  const [status, setStatus] = useState<WSStatus>("connecting");
  const onAlertRef = useRef(onAlert);
  onAlertRef.current = onAlert;

  // Keep the WebSocket instance + a rolling backoff in refs so React
  // doesn't tear them down on every render.
  const wsRef = useRef<WebSocket | null>(null);
  const backoffRef = useRef(500);
  const stoppedRef = useRef(false);
  const reconnectTimerRef = useRef<number | null>(null);

  const connect = useCallback(() => {
    if (stoppedRef.current) return;
    setStatus("connecting");
    let ws: WebSocket;
    try {
      ws = new WebSocket(wsUrl(path));
    } catch {
      // URL constructor failed (bad path). Bail without reconnecting.
      setStatus("closed");
      return;
    }
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("open");
      backoffRef.current = 500; // reset on success
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as FraudAlert;
        onAlertRef.current(data);
      } catch {
        // Drop malformed payloads silently.
      }
    };

    ws.onerror = () => {
      // ``onerror`` always precedes ``onclose``; let close handle reconnect.
    };

    ws.onclose = () => {
      setStatus("closed");
      if (stoppedRef.current) return;
      const delay = Math.min(backoffRef.current, 10_000);
      backoffRef.current = Math.min(backoffRef.current * 1.6, 10_000);
      reconnectTimerRef.current = window.setTimeout(connect, delay);
    };
  }, [path]);

  useEffect(() => {
    stoppedRef.current = false;
    connect();
    return () => {
      stoppedRef.current = true;
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
      }
      // Null out the handlers BEFORE close() so the dying socket's
      // ``onclose`` can't trigger a reconnect cascade. This matters in
      // React 18 dev StrictMode where the effect is double-invoked --
      // without this, the first WS's ``onclose`` fires after the second
      // mount has reset ``stoppedRef``, schedules a reconnect, and we
      // end up with two live sockets that each receive every alert.
      const ws = wsRef.current;
      if (ws) {
        ws.onopen = null;
        ws.onmessage = null;
        ws.onerror = null;
        ws.onclose = null;
        ws.close();
      }
      wsRef.current = null;
    };
  }, [connect]);

  const reconnect = useCallback(() => {
    backoffRef.current = 500;
    wsRef.current?.close();
  }, []);

  return { status, reconnect };
}
