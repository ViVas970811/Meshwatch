import { useMemo } from "react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { FraudPrediction } from "@/api/types";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Empty } from "@/components/ui/Empty";
import { AREA_GRADIENT_STOPS, CHART_TOKENS } from "@/lib/chart-theme";
import { fmtMs } from "@/lib/format";

export function LatencyMonitor({ predictions }: { predictions: FraudPrediction[] }) {
  const series = useMemo(() => {
    return predictions
      .slice(0, 100)
      .map((p, i) => ({
        idx: predictions.length - i,
        total_ms: Number(p.latency_ms?.total_ms ?? 0),
      }))
      .reverse();
  }, [predictions]);

  const stats = useMemo(() => {
    if (predictions.length === 0) return { p50: 0, p95: 0, p99: 0 };
    const sorted = predictions
      .map((p) => Number(p.latency_ms?.total_ms ?? 0))
      .filter(Number.isFinite)
      .sort((a, b) => a - b);
    const at = (q: number) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
    return { p50: at(0.5), p95: at(0.95), p99: at(0.99) };
  }, [predictions]);

  return (
    <Card>
      <CardHeader
        title="Inference latency"
        subtitle="end-to-end /predict over the trailing window"
        right={
          <div className="flex gap-4 text-[11px]">
            <Quant label="P50" v={stats.p50} />
            <Quant label="P95" v={stats.p95} accent />
            <Quant label="P99" v={stats.p99} />
          </div>
        }
      />
      <CardBody>
        {predictions.length === 0 ? (
          <Empty title="No latency data yet" />
        ) : (
          <div className="h-36">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={series} margin={{ left: 4, right: 4, top: 8, bottom: 0 }}>
                <defs>
                  <linearGradient id="latencyGrad" x1="0" y1="0" x2="0" y2="1">
                    {AREA_GRADIENT_STOPS.map((s) => (
                      <stop key={s.offset} offset={s.offset} stopColor={s.color} stopOpacity={s.opacity} />
                    ))}
                  </linearGradient>
                </defs>
                <CartesianGrid {...CHART_TOKENS.grid} />
                <XAxis dataKey="idx" hide />
                <YAxis {...CHART_TOKENS.axis} width={32} tickFormatter={(v) => `${v}ms`} />
                <Tooltip
                  {...CHART_TOKENS.tooltip}
                  formatter={(v: number) => [`${fmtMs(v)}`, "latency"]}
                  labelFormatter={(v) => `request #${v}`}
                />
                <Area
                  type="monotone"
                  dataKey="total_ms"
                  stroke={CHART_TOKENS.accents.primary}
                  strokeWidth={2}
                  fill="url(#latencyGrad)"
                  isAnimationActive
                  animationDuration={500}
                  animationEasing="ease-out"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardBody>
    </Card>
  );
}

function Quant({ label, v, accent }: { label: string; v: number; accent?: boolean }) {
  return (
    <div className="flex items-baseline gap-1.5">
      <span className="text-[10px] font-medium uppercase tracking-kicker text-ink-400">{label}</span>
      <span className={accent ? "font-mono text-sm font-semibold text-ink-50" : "font-mono text-ink-100"}>
        {fmtMs(v)}
      </span>
    </div>
  );
}
