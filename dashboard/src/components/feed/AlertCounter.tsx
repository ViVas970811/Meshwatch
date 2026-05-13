import { useMemo } from "react";

import type { FraudAlert } from "@/api/types";
import { AnimatedNumber, Stat } from "@/components/ui/Stat";

export function AlertCounter({ alerts, delay = 0 }: { alerts: FraudAlert[]; delay?: number }) {
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

  const accent = summary.critical > 0 ? "danger" : summary.high > 0 ? "warn" : "neutral";

  return (
    <Stat
      label="Active alerts"
      value={<AnimatedNumber value={summary.total} />}
      accent={accent}
      delay={delay}
      hint={
        <span className="flex items-center gap-2 text-ink-300">
          <span className="inline-flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-risk-critical shadow-[0_0_6px_currentColor]" />
            <span className="text-risk-critical">{summary.critical}</span>
            <span className="text-ink-500">crit</span>
          </span>
          <span className="text-ink-700">·</span>
          <span className="inline-flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-risk-high" />
            <span className="text-risk-high">{summary.high}</span>
            <span className="text-ink-500">high</span>
          </span>
          <span className="text-ink-700">·</span>
          <span className="inline-flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-risk-medium" />
            <span className="text-risk-medium">{summary.medium}</span>
            <span className="text-ink-500">med</span>
          </span>
        </span>
      }
    />
  );
}
