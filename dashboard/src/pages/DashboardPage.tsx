import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { useEffect, useMemo } from "react";

import { api } from "@/api/client";
import { FraudRateGauge } from "@/components/charts/FraudRateGauge";
import { LatencyMonitor } from "@/components/charts/LatencyMonitor";
import { ScoreDistributionChart } from "@/components/charts/ScoreDistributionChart";
import { AlertCounter } from "@/components/feed/AlertCounter";
import { TransactionFeed } from "@/components/feed/TransactionFeed";
import { AnimatedNumber, Stat } from "@/components/ui/Stat";
import { useAlertStore } from "@/store/alerts";

export function DashboardPage() {
  const alerts = useAlertStore((s) => s.alerts);
  const pushAlerts = useAlertStore((s) => s.pushAlerts);

  const { data: recent } = useQuery({
    queryKey: ["recent"],
    queryFn: api.recent,
    refetchInterval: 5000,
  });

  const recentAlerts = recent?.alerts;
  useEffect(() => {
    if (!recentAlerts) return;
    if (alerts.length > 0) return;
    if (recentAlerts.length === 0) return;
    pushAlerts(recentAlerts);
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
    <div className="space-y-8">
      {/* Hero header */}
      <motion.header
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
        className="flex flex-wrap items-end justify-between gap-4"
      >
        <div>
          <div className="kicker">Real-time monitor</div>
          <h1 className="mt-1 page-title">Live fraud activity</h1>
          <p className="mt-1.5 page-sub">
            Transactions, risk scores, and alerts updated in real time.
          </p>
        </div>
      </motion.header>

      {/* Top KPI row */}
      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <AlertCounter alerts={alerts} delay={0.0} />
        <Stat
          label="Transactions scored"
          value={<AnimatedNumber value={predictions.length} />}
          hint="Across the current monitoring window"
          delay={0.05}
        />
        <Stat
          label="Average response time"
          value={
            <AnimatedNumber
              value={latencyP95}
              decimals={1}
              suffix=" ms"
              thousands={false}
            />
          }
          accent={latencyP95 < 50 ? "good" : latencyP95 < 100 ? "warn" : "danger"}
          hint="95th percentile, target < 50 ms"
          delay={0.1}
        />
        <Stat
          label="Total alerts"
          value={<AnimatedNumber value={alerts.length} />}
          hint="Received this session"
          delay={0.15}
        />
      </section>

      {/* Visualization row 1 */}
      <section className="grid gap-4 lg:grid-cols-3">
        <FraudRateGauge rate={fraudRate} total={predictions.length} />
        <div className="lg:col-span-2">
          <ScoreDistributionChart predictions={predictions} />
        </div>
      </section>

      {/* Visualization row 2 */}
      <section className="grid gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <LatencyMonitor predictions={predictions} />
        </div>
        <TransactionFeed alerts={alerts} />
      </section>
    </div>
  );
}
