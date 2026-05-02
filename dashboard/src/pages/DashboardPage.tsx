import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo } from "react";

import { api } from "@/api/client";
import { FraudRateGauge } from "@/components/charts/FraudRateGauge";
import { LatencyMonitor } from "@/components/charts/LatencyMonitor";
import { ScoreDistributionChart } from "@/components/charts/ScoreDistributionChart";
import { AlertCounter } from "@/components/feed/AlertCounter";
import { TransactionFeed } from "@/components/feed/TransactionFeed";
import { Stat } from "@/components/ui/Stat";
import { fmtMs } from "@/lib/format";
import { useAlertStore } from "@/store/alerts";

export function DashboardPage() {
  const alerts = useAlertStore((s) => s.alerts);
  const pushAlerts = useAlertStore((s) => s.pushAlerts);

  const { data: recent } = useQuery({
    queryKey: ["recent"],
    queryFn: api.recent,
    refetchInterval: 5000,
  });

  // Backfill the alert store with anything from /api/v1/recent when the
  // store is empty -- this fills in the dashboard on page reload before
  // any new WebSocket alert lands.
  const recentAlerts = recent?.alerts;
  useEffect(() => {
    if (!recentAlerts) return;
    if (alerts.length > 0) return;
    if (recentAlerts.length === 0) return;
    pushAlerts(recentAlerts);
    // We only want to backfill once on first load, hence the empty deps.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recentAlerts]);

  const predictions = recent?.predictions ?? [];

  const fraudRate = useMemo(() => {
    if (predictions.length === 0) return 0;
    const flagged = predictions.filter((p) => p.is_fraud_predicted).length;
    return flagged / predictions.length;
  }, [predictions]);

  const latencyP95 = useMemo(() => {
    if (predictions.length === 0) return 0;
    const sorted = predictions
      .map((p) => Number(p.latency_ms?.total_ms ?? 0))
      .sort((a, b) => a - b);
    return sorted[Math.floor(0.95 * sorted.length)] ?? 0;
  }, [predictions]);

  return (
    <div className="grid gap-6">
      <header className="flex items-end justify-between">
        <div>
          <div className="text-xs uppercase tracking-widest text-ink-400">Real-time monitor</div>
          <h1 className="text-2xl font-semibold">Live fraud activity</h1>
          <p className="text-sm text-ink-300">
            Alerts stream over <code className="text-ink-200">/ws/alerts</code>; predictions
            populate via <code className="text-ink-200">/api/v1/recent</code> every 5 s.
          </p>
        </div>
      </header>

      {/* Top stat row */}
      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <AlertCounter alerts={alerts} />
        <Stat
          label="Recent predictions"
          value={predictions.length}
          hint={`window: ${predictions.length} most recent`}
        />
        <Stat
          label="Latency P95"
          value={fmtMs(latencyP95)}
          hint="end-to-end /predict, target <50ms"
        />
        <Stat
          label="Alerts (live buffer)"
          value={alerts.length}
          hint="capped at 500 by Zustand store"
        />
      </section>

      {/* Charts row */}
      <section className="grid gap-4 lg:grid-cols-3">
        <FraudRateGauge rate={fraudRate} total={predictions.length} />
        <div className="lg:col-span-2">
          <ScoreDistributionChart predictions={predictions} />
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <LatencyMonitor predictions={predictions} />
        </div>
        <TransactionFeed alerts={alerts} />
      </section>
    </div>
  );
}
