import { Cell, Pie, PieChart, ResponsiveContainer } from "recharts";

import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { fmtPct } from "@/lib/format";

interface Props {
  rate: number;        // [0, 1]
  baseline?: number;   // [0, 1]; the historical baseline for context
  total: number;       // sample size
}

export function FraudRateGauge({ rate, baseline = 0.035, total }: Props) {
  const r = clamp01(rate);
  const data = [
    { name: "fraud", value: r },
    { name: "ok", value: Math.max(0, 1 - r) },
  ];
  const color = r >= 0.1 ? "#ef4444" : r >= 0.05 ? "#f97316" : r >= 0.02 ? "#eab308" : "#22c55e";

  return (
    <Card>
      <CardHeader title="Fraud rate" subtitle={`window: last ${total} predictions`} />
      <CardBody className="flex items-center gap-4 p-5">
        <div className="relative h-32 w-32 shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data}
                dataKey="value"
                innerRadius={45}
                outerRadius={60}
                startAngle={90}
                endAngle={-270}
                strokeWidth={0}
              >
                <Cell fill={color} />
                <Cell fill="#1a2138" />
              </Pie>
            </PieChart>
          </ResponsiveContainer>
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
            <div className="text-2xl font-semibold tabular-nums">{fmtPct(r)}</div>
            <div className="text-[10px] uppercase tracking-widest text-ink-400">
              {total === 0 ? "no data" : "current"}
            </div>
          </div>
        </div>
        <div className="grid gap-2 text-xs text-ink-300">
          <div>
            <div className="stat-label">baseline</div>
            <div className="text-ink-100">{fmtPct(baseline)}</div>
          </div>
          <div>
            <div className="stat-label">delta</div>
            <div className={r > baseline ? "text-risk-high" : "text-risk-low"}>
              {fmtPct(r - baseline, 2)}
            </div>
          </div>
        </div>
      </CardBody>
    </Card>
  );
}

const clamp01 = (n: number) => Math.max(0, Math.min(1, n));
