import { useMemo } from "react";

import type { FraudAlert } from "@/api/types";
import { Stat } from "@/components/ui/Stat";

export function AlertCounter({ alerts }: { alerts: FraudAlert[] }) {
  const summary = useMemo(() => {
    let critical = 0;
    let high = 0;
    let medium = 0;
    for (const a of alerts) {
      if (a.risk_level === "CRITICAL") critical++;
      else if (a.risk_level === "HIGH") high++;
      else if (a.risk_level === "MEDIUM") medium++;
    }
    return { critical, high, medium, total: alerts.length };
  }, [alerts]);

  return (
    <Stat
      label="Active alerts"
      value={summary.total}
      hint={
        <span className="text-ink-300">
          <span className="text-risk-critical">{summary.critical}</span> CRIT ·{" "}
          <span className="text-risk-high">{summary.high}</span> HIGH ·{" "}
          <span className="text-risk-medium">{summary.medium}</span> MED
        </span>
      }
    />
  );
}
