import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { FraudPrediction } from "@/api/types";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Empty } from "@/components/ui/Empty";
import { scoreColor } from "@/lib/colors";

interface Props {
  predictions: FraudPrediction[];
  bins?: number;
}

export function ScoreDistributionChart({ predictions, bins = 20 }: Props) {
  const data = useMemo(() => buildBins(predictions, bins), [predictions, bins]);
  return (
    <Card>
      <CardHeader title="Score distribution" subtitle={`${predictions.length} recent predictions`} />
      <CardBody>
        {predictions.length === 0 ? (
          <Empty title="No predictions yet" hint="Send a /api/v1/predict to populate." />
        ) : (
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data} margin={{ left: 4, right: 4, top: 4, bottom: 0 }}>
                <CartesianGrid stroke="#1a2138" vertical={false} />
                <XAxis
                  dataKey="midpoint"
                  type="number"
                  domain={[0, 1]}
                  tickFormatter={(v) => v.toFixed(1)}
                  tick={{ fontSize: 10, fill: "#7c87ad" }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  allowDecimals={false}
                  tick={{ fontSize: 10, fill: "#7c87ad" }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  cursor={{ fill: "rgba(255,255,255,0.04)" }}
                  contentStyle={{
                    background: "#11172b",
                    border: "1px solid #2a3253",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  formatter={(v) => [v, "count"]}
                  labelFormatter={(v: number) => `score ~ ${v.toFixed(2)}`}
                />
                <Bar dataKey="count" radius={[3, 3, 0, 0]} fillOpacity={0.85}>
                  {data.map((d, i) => (
                    <CellTinted key={i} score={d.midpoint} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardBody>
    </Card>
  );
}

function CellTinted({ score }: { score: number }) {
  // Recharts injects this into the Bar via cloneElement when used inside <Bar>.
  return <rect fill={scoreColor(score)} />;
}

interface Bin {
  midpoint: number;
  count: number;
}

function buildBins(predictions: FraudPrediction[], bins: number): Bin[] {
  const out: Bin[] = Array.from({ length: bins }, (_, i) => ({
    midpoint: (i + 0.5) / bins,
    count: 0,
  }));
  for (const p of predictions) {
    const idx = Math.min(bins - 1, Math.max(0, Math.floor(p.fraud_score * bins)));
    out[idx].count += 1;
  }
  return out;
}
