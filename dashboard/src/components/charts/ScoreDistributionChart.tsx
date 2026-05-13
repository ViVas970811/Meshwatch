import { useMemo } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { FraudPrediction } from "@/api/types";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Empty } from "@/components/ui/Empty";
import { CHART_TOKENS } from "@/lib/chart-theme";
import { scoreColor } from "@/lib/colors";

interface Props {
  predictions: FraudPrediction[];
  bins?: number;
}

export function ScoreDistributionChart({ predictions, bins = 20 }: Props) {
  const data = useMemo(() => buildBins(predictions, bins), [predictions, bins]);
  return (
    <Card>
      <CardHeader
        title="Risk score distribution"
        subtitle={`Across the last ${predictions.length} scored transactions`}
      />
      <CardBody>
        {predictions.length === 0 ? (
          <Empty
            title="No data yet"
            hint="Score distribution will appear here once transactions are processed."
          />
        ) : (
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data} margin={{ left: 4, right: 4, top: 8, bottom: 0 }}>
                <CartesianGrid {...CHART_TOKENS.grid} />
                <XAxis
                  {...CHART_TOKENS.axis}
                  dataKey="midpoint"
                  type="number"
                  domain={[0, 1]}
                  tickFormatter={(v) => v.toFixed(1)}
                />
                <YAxis {...CHART_TOKENS.axis} allowDecimals={false} width={28} />
                <Tooltip
                  {...CHART_TOKENS.tooltip}
                  formatter={(v) => [v, "count"]}
                  labelFormatter={(v: number) => `score ≈ ${v.toFixed(2)}`}
                />
                <Bar
                  dataKey="count"
                  radius={[4, 4, 0, 0]}
                  fillOpacity={0.9}
                  isAnimationActive
                  animationDuration={500}
                  animationEasing="ease-out"
                >
                  {data.map((d, i) => (
                    <Cell key={i} fill={scoreColor(d.midpoint)} />
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
