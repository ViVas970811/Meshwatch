import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import { CasesPage } from "@/pages/CasesPage";
import { NetworkPage } from "@/pages/NetworkPage";

// Stub recharts' ResponsiveContainer so charts render with a fixed size in
// jsdom (which has zero layout). This avoids the "width(0) and height(0)"
// noise without faking the rest of the chart pipeline.
vi.mock("recharts", async () => {
  const actual = await vi.importActual<typeof import("recharts")>("recharts");
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div style={{ width: 800, height: 400 }}>{children}</div>
    ),
  };
});

// react-force-graph-2d touches the canvas heavily on mount; stub it so
// the network page renders structurally.
vi.mock("react-force-graph-2d", () => ({
  default: () => <div data-testid="force-graph-stub" />,
}));

function withProviders(ui: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return (
    <QueryClientProvider client={qc}>
      <MemoryRouter>
        <Routes>
          <Route path="/" element={ui} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe("CasesPage", () => {
  it("renders empty state when there are no cases", () => {
    render(withProviders(<CasesPage />));
    expect(screen.getByText("Investigations")).toBeInTheDocument();
    expect(screen.getByText(/No cases yet/i)).toBeInTheDocument();
  });

  it("renders all status filter tabs", () => {
    render(withProviders(<CasesPage />));
    for (const label of ["All", "Open", "Review", "Resolved", "Escalated"]) {
      expect(screen.getByRole("button", { name: new RegExp(`^${label}\\b`) })).toBeInTheDocument();
    }
  });
});

describe("NetworkPage", () => {
  it("renders header + force-graph stub + community list", () => {
    render(withProviders(<NetworkPage />));
    expect(screen.getByText("Fraud network")).toBeInTheDocument();
    expect(screen.getByTestId("force-graph-stub")).toBeInTheDocument();
    expect(screen.getByText("Eastern velocity ring")).toBeInTheDocument();
  });
});
