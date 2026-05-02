import { useMemo } from "react";
import {
  Area,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { FraudPrediction } from "@/api/types";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Empty } from "@/components/ui/Empty";
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
        title="Latency"
        subtitle="end-to-end /predict (ms)"
        right={
          <div className="flex gap-3 text-[11px] text-ink-300">
            <Quant label="p50" v={stats.p50} />
            <Quant label="p95" v={stats.p95} />
            <Quant label="p99" v={stats.p99} />
          </div>
        }
      />
      <CardBody>
        {predictions.length === 0 ? (
          <Empty title="No latency data yet" />
        ) : (
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={series} margin={{ left: 4, right: 4, top: 4, bottom: 0 }}>
                <defs>
                  <linearGradient id="latencyGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#5b8def" stopOpacity={0.5} />
                    <stop offset="100%" stopColor="#5b8def" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="idx" hide />
                <YAxis tick={{ fontSize: 10, fill: "#7c87ad" }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{
                    background: "#11172b",
                    border: "1px solid #2a3253",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  formatter={(v: number) => [`${fmtMs(v)}`, "total"]}
                  labelFormatter={(v) => `#${v}`}
                />
                <Area
                  type="monotone"
                  dataKey="total_ms"
                  stroke="#5b8def"
                  strokeWidth={1.5}
                  fill="url(#latencyGrad)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardBody>
    </Card>
  );
}

function Quant({ label, v }: { label: string; v: number }) {
  return (
    <div className="flex items-baseline gap-1">
      <span className="uppercase tracking-wider text-ink-400">{label}</span>
      <span className="font-mono text-ink-100">{fmtMs(v)}</span>
    </div>
  );
}
