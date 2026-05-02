/**
 * Zustand store for the live fraud-alert ring buffer.
 *
 * Capped at MAX_ALERTS so unbounded streaming doesn't leak memory.
 * Components subscribe to slices via ``useAlertStore(state => state.alerts)``.
 */

import { create } from "zustand";

import type { FraudAlert } from "@/api/types";

const MAX_ALERTS = 500;

interface AlertState {
  alerts: FraudAlert[];
  pushAlert: (a: FraudAlert) => void;
  pushAlerts: (xs: FraudAlert[]) => void;
  clear: () => void;
}

export const useAlertStore = create<AlertState>((set) => ({
  alerts: [],
  pushAlert: (a) =>
    set((state) => {
      const next = [a, ...state.alerts];
      if (next.length > MAX_ALERTS) next.length = MAX_ALERTS;
      return { alerts: next };
    }),
  pushAlerts: (xs) =>
    set((state) => {
      const next = [...xs, ...state.alerts];
      if (next.length > MAX_ALERTS) next.length = MAX_ALERTS;
      return { alerts: next };
    }),
  clear: () => set({ alerts: [] }),
}));
