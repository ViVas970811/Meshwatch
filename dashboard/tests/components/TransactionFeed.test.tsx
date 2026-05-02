import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import type { FraudAlert } from "@/api/types";
import { TransactionFeed } from "@/components/feed/TransactionFeed";

const ALERTS: FraudAlert[] = [
  {
    transaction_id: 1001,
    fraud_score: 0.95,
    risk_level: "CRITICAL",
    transaction_amt: 4210,
    card_id: 9999,
    timestamp: new Date().toISOString(),
  },
  {
    transaction_id: 1002,
    fraud_score: 0.78,
    risk_level: "HIGH",
    transaction_amt: 350,
    card_id: 8888,
    timestamp: new Date().toISOString(),
  },
];

describe("TransactionFeed", () => {
  it("renders an empty state when no alerts", () => {
    render(
      <MemoryRouter>
        <TransactionFeed alerts={[]} />
      </MemoryRouter>,
    );
    expect(screen.getByText(/Waiting for alerts/i)).toBeInTheDocument();
  });

  it("renders rows for each alert with risk badge + amount", () => {
    render(
      <MemoryRouter>
        <TransactionFeed alerts={ALERTS} />
      </MemoryRouter>,
    );
    // Risk badges
    expect(screen.getByText("CRITICAL")).toBeInTheDocument();
    expect(screen.getByText("HIGH")).toBeInTheDocument();
    // Amounts (currency-formatted)
    expect(screen.getByText("$4,210.00")).toBeInTheDocument();
    expect(screen.getByText("$350.00")).toBeInTheDocument();
  });

  it("clicking a row navigates to the alert page", async () => {
    const user = userEvent.setup();
    const mockReached = vi.fn();
    render(
      <MemoryRouter initialEntries={["/dashboard"]}>
        <Routes>
          <Route path="/dashboard" element={<TransactionFeed alerts={ALERTS} />} />
          <Route
            path="/alerts/:alertId"
            element={
              <div
                data-alert-id={
                  // @ts-expect-error -- stash so we can assert
                  window.__last_alert_id
                }
              >
                routed
              </div>
            }
          />
        </Routes>
      </MemoryRouter>,
    );

    // The first row corresponds to txn 1001.
    const row = screen.getByText(/txn 1001/).closest("li");
    expect(row).not.toBeNull();
    await user.click(row!);
    // After navigating, the route renders 'routed'.
    expect(screen.getByText("routed")).toBeInTheDocument();
    mockReached();
  });
});
