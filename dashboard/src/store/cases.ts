/**
 * Zustand store for analyst case management (Phase 6).
 *
 * Cases are created from successful agent investigations on the
 * Investigation page. Status transitions: open -> review -> resolved | escalated.
 *
 * The store is intentionally in-memory only -- Phase 7 will wire it to a
 * persistent store. Until then, the dashboard provides a useful demo loop.
 */

import { create } from "zustand";

import type { InvestigationReport } from "@/api/types";

export type CaseStatus = "open" | "review" | "resolved" | "escalated";

export interface FraudCase {
  alert_id: string;
  transaction_id: number | string;
  status: CaseStatus;
  report: InvestigationReport;
  opened_at: string;
}

interface CaseState {
  cases: Record<string, FraudCase>;
  upsertCase: (report: InvestigationReport) => FraudCase;
  setStatus: (alert_id: string, status: CaseStatus) => void;
  remove: (alert_id: string) => void;
}

export const useCaseStore = create<CaseState>((set, get) => ({
  cases: {},
  upsertCase: (report) => {
    const existing = get().cases[report.alert_id];
    const initial: CaseStatus =
      report.recommended_action === "decline"
        ? "review"
        : report.recommended_action === "escalate"
          ? "escalated"
          : "open";
    const fc: FraudCase = existing ?? {
      alert_id: report.alert_id,
      transaction_id: report.transaction_id,
      status: initial,
      report,
      opened_at: new Date().toISOString(),
    };
    // Always replace the report (latest wins).
    fc.report = report;
    set((state) => ({ cases: { ...state.cases, [report.alert_id]: fc } }));
    return fc;
  },
  setStatus: (alert_id, status) =>
    set((state) => {
      const c = state.cases[alert_id];
      if (!c) return state;
      return { cases: { ...state.cases, [alert_id]: { ...c, status } } };
    }),
  remove: (alert_id) =>
    set((state) => {
      const next = { ...state.cases };
      delete next[alert_id];
      return { cases: next };
    }),
}));
