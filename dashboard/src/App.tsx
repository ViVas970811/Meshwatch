import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, type ReactNode } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";

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
      ? `(${total > 99 ? "99+" : total}) Meshwatch — Fraud Intelligence`
      : "Meshwatch — Fraud Intelligence";
  }, [total]);

  return (
    <div className="min-h-full">
      <NavBar wsStatus={wsStatus} />
      <main className="mx-auto max-w-[1400px] px-6 py-8">
        <AnimatedRoutes />
      </main>
    </div>
  );
}

function AnimatedRoutes() {
  const location = useLocation();
  return (
    <AnimatePresence mode="wait" initial={false}>
      <Routes location={location} key={location.pathname.split("/")[1] || "root"}>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route
          path="/dashboard"
          element={
            <PageWrapper>
              <DashboardPage />
            </PageWrapper>
          }
        />
        <Route
          path="/alerts/:alertId"
          element={
            <PageWrapper>
              <AlertPage />
            </PageWrapper>
          }
        />
        <Route
          path="/network"
          element={
            <PageWrapper>
              <NetworkPage />
            </PageWrapper>
          }
        />
        <Route
          path="/model"
          element={
            <PageWrapper>
              <ModelPage />
            </PageWrapper>
          }
        />
        <Route
          path="/cases"
          element={
            <PageWrapper>
              <CasesPage />
            </PageWrapper>
          }
        />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </AnimatePresence>
  );
}

function PageWrapper({ children }: { children: ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.32, ease: [0.16, 1, 0.3, 1] }}
    >
      {children}
    </motion.div>
  );
}
