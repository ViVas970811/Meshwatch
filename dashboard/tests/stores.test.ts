import { beforeEach, describe, expect, it } from "vitest";

import type { FraudAlert, InvestigationReport } from "@/api/types";
import { useAlertStore } from "@/store/alerts";
import { useCaseStore } from "@/store/cases";

const _alert = (id: number): FraudAlert => ({
  transaction_id: id,
  fraud_score: 0.8,
  risk_level: "HIGH",
  transaction_amt: 100,
  card_id: id + 1,
  timestamp: new Date().toISOString(),
});

describe("useAlertStore", () => {
  beforeEach(() => useAlertStore.getState().clear());

  it("pushAlert prepends new alerts (newest first)", () => {
    useAlertStore.getState().pushAlert(_alert(1));
    useAlertStore.getState().pushAlert(_alert(2));
    expect(useAlertStore.getState().alerts.map((a) => a.transaction_id)).toEqual([2, 1]);
  });

  it("caps at 500 alerts", () => {
    for (let i = 0; i < 510; i++) useAlertStore.getState().pushAlert(_alert(i));
    expect(useAlertStore.getState().alerts).toHaveLength(500);
    // Newest first -> head should be id 509.
    expect(useAlertStore.getState().alerts[0].transaction_id).toBe(509);
  });

  it("clear empties the buffer", () => {
    useAlertStore.getState().pushAlert(_alert(1));
    useAlertStore.getState().clear();
    expect(useAlertStore.getState().alerts).toHaveLength(0);
  });
});

const _report = (
  override: Partial<InvestigationReport> = {},
): InvestigationReport => ({
  alert_id: "inv-1",
  transaction_id: 1,
  risk_level: "CRITICAL",
  depth: "deep",
  fraud_score: 0.95,
  summary: "x",
  narrative: "y",
  recommended_action: "decline",
  confidence: 0.9,
  requires_human_review: true,
  entity_risks: [],
  matched_patterns: [],
  similar_cases: [],
  tools_used: ["a", "b"],
  tool_calls: [],
  elapsed_ms: 30,
  model: "stub",
  ...override,
});

describe("useCaseStore", () => {
  beforeEach(() => {
    // Manually clear since the store doesn't expose a method for it.
    Object.keys(useCaseStore.getState().cases).forEach((id) =>
      useCaseStore.getState().remove(id),
    );
  });

  it("upsertCase creates with status=review for decline action", () => {
    const c = useCaseStore.getState().upsertCase(_report());
    expect(c.status).toBe("review");
    expect(useCaseStore.getState().cases["inv-1"]).toBeDefined();
  });

  it("upsertCase keeps existing status when re-run", () => {
    const first = useCaseStore.getState().upsertCase(_report());
    useCaseStore.getState().setStatus(first.alert_id, "resolved");
    const second = useCaseStore.getState().upsertCase(_report({ summary: "v2" }));
    expect(second.status).toBe("resolved");
    expect(useCaseStore.getState().cases["inv-1"].report.summary).toBe("v2");
  });

  it("escalate action seeds escalated status", () => {
    const c = useCaseStore.getState().upsertCase(_report({ recommended_action: "escalate" }));
    expect(c.status).toBe("escalated");
  });

  it("approve action seeds open status", () => {
    const c = useCaseStore
      .getState()
      .upsertCase(_report({ recommended_action: "approve" }));
    expect(c.status).toBe("open");
  });

  it("setStatus + remove work", () => {
    useCaseStore.getState().upsertCase(_report());
    useCaseStore.getState().setStatus("inv-1", "resolved");
    expect(useCaseStore.getState().cases["inv-1"].status).toBe("resolved");
    useCaseStore.getState().remove("inv-1");
    expect(useCaseStore.getState().cases["inv-1"]).toBeUndefined();
  });
});
