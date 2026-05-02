import { useCallback, useEffect, useMemo } from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import { useAlertSocket } from "@/api/ws";
import { NavBar } from "@/components/NavBar";
import { AlertPage } from "@/pages/AlertPage";
import { CasesPage } from "@/pages/CasesPage";
import { DashboardPage } from "@/pages/DashboardPage";
import { ModelPage } from "@/pages/ModelPage";
import { NetworkPage } from "@/pages/NetworkPage";
import { useAlertStore } from "@/store/alerts";

export function App() {
  const pushAlert = useAlertStore((s) => s.pushAlert);

  // Stable callback so the WebSocket hook doesn't tear down on rerenders.
  const onAlert = useCallback(
    (alert: Parameters<typeof pushAlert>[0]) => {
      pushAlert(alert);
    },
    [pushAlert],
  );

  const { status: wsStatus } = useAlertSocket({ onAlert });

  // Document title reflects the live alert count.
  const total = useAlertStore((s) => s.alerts.length);
  useEffect(() => {
    document.title = total
      ? `(${total > 99 ? "99+" : total}) Meshwatch — Fraud Detection`
      : "Meshwatch — Fraud Detection";
  }, [total]);

  const layoutShell = useMemo(
    () => (
      <div className="grid-bg min-h-full">
        <NavBar wsStatus={wsStatus} />
        <main className="mx-auto max-w-[1400px] px-6 py-6">
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/alerts/:alertId" element={<AlertPage />} />
            <Route path="/network" element={<NetworkPage />} />
            <Route path="/model" element={<ModelPage />} />
            <Route path="/cases" element={<CasesPage />} />
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </main>
      </div>
    ),
    [wsStatus],
  );

  return layoutShell;
}
