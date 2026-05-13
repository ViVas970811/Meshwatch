import { motion } from "framer-motion";
import { Cell, Pie, PieChart, ResponsiveContainer } from "recharts";

import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { fmtPct } from "@/lib/format";

interface Props {
  rate: number; // [0, 1]
  baseline?: number; // [0, 1]; historical baseline
  total: number; // sample size
}

export function FraudRateGauge({ rate, baseline = 0.035, total }: Props) {
  const r = clamp01(rate);
  const data = [
    { name: "fraud", value: r },
    { name: "ok", value: Math.max(0, 1 - r) },
  ];
  const color =
    r >= 0.1 ? "#ef4444" : r >= 0.05 ? "#f97316" : r >= 0.02 ? "#f59e0b" : "#10b981";
  const delta = r - baseline;
  const deltaTone = delta > 0.005 ? "text-risk-high" : delta < -0.005 ? "text-risk-low" : "text-ink-400";

  return (
    <Card>
      <CardHeader
        title="Production fraud rate"
        subtitle={`Trailing ${total || 0} predictions`}
      />
      <CardBody className="flex items-center gap-5 p-5">
        <div className="relative h-32 w-32 shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <defs>
                <linearGradient id={`gaugeFill-${color.slice(1)}`} x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stopColor={color} stopOpacity={0.9} />
                  <stop offset="100%" stopColor={color} stopOpacity={0.55} />
                </linearGradient>
              </defs>
              <Pie
                data={data}
                dataKey="value"
                innerRadius={46}
                outerRadius={60}
                startAngle={90}
                endAngle={-270}
                strokeWidth={0}
                isAnimationActive
                animationDuration={700}
                animationEasing="ease-out"
              >
                <Cell fill={`url(#gaugeFill-${color.slice(1)})`} />
                <Cell fill="rgba(45, 56, 90, 0.5)" />
              </Pie>
            </PieChart>
          </ResponsiveContainer>
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
            className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center"
          >
            <div className="text-display-sm font-semibold tabular-nums text-ink-50">
              {fmtPct(r)}
            </div>
            <div className="kicker mt-0.5">{total === 0 ? "no data" : "current"}</div>
          </motion.div>
        </div>
        <div className="grid gap-3 text-sm">
          <div>
            <div className="kicker">Baseline</div>
            <div className="mt-0.5 font-mono text-ink-100">{fmtPct(baseline)}</div>
          </div>
          <div>
            <div className="kicker">Delta</div>
            <div className={`mt-0.5 font-mono font-semibold ${deltaTone}`}>
              {delta > 0 ? "+" : ""}
              {fmtPct(delta, 2)}
            </div>
          </div>
        </div>
      </CardBody>
    </Card>
  );
}

const clamp01 = (n: number) => Math.max(0, Math.min(1, n));
