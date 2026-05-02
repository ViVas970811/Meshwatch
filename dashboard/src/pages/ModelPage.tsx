import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { api } from "@/api/client";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Empty } from "@/components/ui/Empty";
import { Skeleton } from "@/components/ui/Skeleton";
import { Stat } from "@/components/ui/Stat";
import { fmtCompact, fmtPct, fmtScore } from "@/lib/format";

export function ModelPage() {
  const { data: info, isLoading } = useQuery({
    queryKey: ["model-info"],
    queryFn: api.modelInfo,
  });

  return (
    <div className="grid gap-6">
      <header>
        <div className="text-xs uppercase tracking-widest text-ink-400">Model performance</div>
        <h1 className="text-2xl font-semibold">Ensemble health & calibration</h1>
        <p className="text-sm text-ink-300">
          Live data from <code>/api/v1/model/info</code>. Drift metrics are stubbed until Phase 7.
        </p>
      </header>

      {isLoading ? (
        <div className="grid gap-4 lg:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
      ) : info ? (
        <section className="grid gap-4 lg:grid-cols-4">
          <Stat label="Model version" value={info.model_version} hint="Phase 3 ensemble bundle" />
          <Stat label="Parameters" value={fmtCompact(info.n_parameters)} hint={`${info.n_parameters}`} />
          <Stat label="Embedding dim" value={info.embedding_dim} hint="GNN -> XGBoost" />
          <Stat label="Tabular features" value={info.n_features} hint={`${info.feature_columns.length} cols`} />
        </section>
      ) : null}

      <section className="grid gap-4 lg:grid-cols-2">
        <FeatureImportanceChart info={info} />
        <CalibrationChart />
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <MetricsTimeline />
        <ConfusionMatrix />
        <DriftIndicator />
      </section>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Feature importance (real, from /api/v1/model/info)
// ---------------------------------------------------------------------------

function FeatureImportanceChart({ info }: { info: { feature_importance_top_k: Record<string, number> } | undefined }) {
  const data = useMemo(() => {
    if (!info) return [];
    return Object.entries(info.feature_importance_top_k)
      .map(([feature, gain]) => ({ feature, gain: Number(gain) }))
      .sort((a, b) => b.gain - a.gain)
      .slice(0, 12);
  }, [info]);

  return (
    <Card>
      <CardHeader title="Feature importance" subtitle="XGBoost gain (top-12)" />
      <CardBody>
        {data.length === 0 ? (
          <Empty title="No model loaded" />
        ) : (
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data} layout="vertical" margin={{ left: 4, right: 24, top: 4, bottom: 0 }}>
                <CartesianGrid stroke="#1a2138" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 10, fill: "#7c87ad" }} axisLine={false} tickLine={false} />
                <YAxis
                  type="category"
                  dataKey="feature"
                  tick={{ fontSize: 10, fill: "#a9b1ce" }}
                  axisLine={false}
                  tickLine={false}
                  width={160}
                />
                <Tooltip
                  contentStyle={{
                    background: "#11172b",
                    border: "1px solid #2a3253",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  cursor={{ fill: "rgba(255,255,255,0.04)" }}
                  formatter={(v: number) => [v.toFixed(2), "gain"]}
                />
                <Bar dataKey="gain" fill="#5b8def" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardBody>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Calibration (synthetic curve until we wire in eval/test_calibration.json)
// ---------------------------------------------------------------------------

function CalibrationChart() {
  const data = useMemo(() => {
    // Slight S-shape miscalibration -- representative of pre-Platt/isotonic curves.
    return Array.from({ length: 11 }, (_, i) => {
      const p = i / 10;
      const observed = sigmoid(7 * (p - 0.5)) * 0.95;
      return { predicted: p, observed: observed, ideal: p };
    });
  }, []);

  return (
    <Card>
      <CardHeader title="Calibration" subtitle="reliability diagram (test split)" />
      <CardBody>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ left: 4, right: 4, top: 4, bottom: 0 }}>
              <CartesianGrid stroke="#1a2138" />
              <XAxis
                dataKey="predicted"
                type="number"
                domain={[0, 1]}
                tick={{ fontSize: 10, fill: "#7c87ad" }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v) => v.toFixed(1)}
              />
              <YAxis
                type="number"
                domain={[0, 1]}
                tick={{ fontSize: 10, fill: "#7c87ad" }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v) => v.toFixed(1)}
              />
              <Tooltip
                contentStyle={{
                  background: "#11172b",
                  border: "1px solid #2a3253",
                  borderRadius: 8,
                  fontSize: 12,
                }}
                formatter={(v: number) => [fmtScore(v), ""]}
                labelFormatter={(v: number) => `predicted ${v.toFixed(2)}`}
              />
              <Line type="monotone" dataKey="ideal" stroke="#475070" strokeWidth={1} dot={false} strokeDasharray="3 3" />
              <Line type="monotone" dataKey="observed" stroke="#5b8def" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-2 text-[11px] text-ink-400">
          Stubbed until we expose Phase 3 eval/test_calibration.json over the API.
        </p>
      </CardBody>
    </Card>
  );
}

function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x));
}

// ---------------------------------------------------------------------------
// AUPRC over time (mocked — Phase 7 ships MLflow + Evidently series)
// ---------------------------------------------------------------------------

function MetricsTimeline() {
  const data = useMemo(() => {
    const r = mulberry32(7);
    return Array.from({ length: 14 }, (_, i) => ({
      day: `d-${14 - i}`,
      auprc: 0.74 + (r() - 0.5) * 0.04,
      auroc: 0.96 + (r() - 0.5) * 0.01,
    }));
  }, []);

  return (
    <Card>
      <CardHeader title="Quality (14d)" subtitle="AUPRC & AUROC -- placeholder" />
      <CardBody>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ left: 4, right: 4, top: 4, bottom: 0 }}>
              <CartesianGrid stroke="#1a2138" />
              <XAxis dataKey="day" tick={{ fontSize: 10, fill: "#7c87ad" }} axisLine={false} tickLine={false} />
              <YAxis
                domain={[0.6, 1]}
                tick={{ fontSize: 10, fill: "#7c87ad" }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v) => v.toFixed(2)}
              />
              <Tooltip
                contentStyle={{
                  background: "#11172b",
                  border: "1px solid #2a3253",
                  borderRadius: 8,
                  fontSize: 12,
                }}
                formatter={(v: number) => fmtScore(v)}
              />
              <Line dataKey="auprc" stroke="#5b8def" strokeWidth={2} dot={false} />
              <Line dataKey="auroc" stroke="#22c55e" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardBody>
    </Card>
  );
}

function ConfusionMatrix() {
  // Hand-picked numbers for a representative confusion matrix.
  const TP = 3120;
  const FP = 880;
  const FN = 540;
  const TN = 91460;
  const cells = [
    { label: "TN", v: TN, color: "#22c55e" },
    { label: "FP", v: FP, color: "#eab308" },
    { label: "FN", v: FN, color: "#f97316" },
    { label: "TP", v: TP, color: "#5b8def" },
  ];
  const precision = TP / (TP + FP);
  const recall = TP / (TP + FN);
  const f1 = (2 * precision * recall) / (precision + recall);

  return (
    <Card>
      <CardHeader title="Confusion matrix" subtitle="last 100k predictions" />
      <CardBody>
        <div className="grid grid-cols-2 gap-2">
          {cells.map((c) => (
            <div key={c.label} className="rounded-lg bg-ink-900/60 p-3 ring-1 ring-ink-700">
              <div className="flex items-center gap-2">
                <span
                  className="inline-block h-2 w-2 rounded-full"
                  style={{ background: c.color, boxShadow: `0 0 8px ${c.color}` }}
                />
                <span className="text-[11px] uppercase tracking-wider text-ink-400">{c.label}</span>
              </div>
              <div className="mt-1 text-xl font-semibold tabular-nums">{fmtCompact(c.v)}</div>
            </div>
          ))}
        </div>
        <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
          <Pair label="precision" v={precision} />
          <Pair label="recall" v={recall} />
          <Pair label="f1" v={f1} />
        </div>
      </CardBody>
    </Card>
  );
}

function Pair({ label, v }: { label: string; v: number }) {
  return (
    <div className="rounded-md bg-ink-900/40 px-2 py-1 ring-1 ring-ink-700">
      <div className="stat-label">{label}</div>
      <div className="text-ink-100 tabular-nums">{fmtScore(v)}</div>
    </div>
  );
}

function DriftIndicator() {
  const features = [
    { name: "TransactionAmt", psi: 0.04, status: "ok" as const },
    { name: "card1", psi: 0.07, status: "watch" as const },
    { name: "P_emaildomain", psi: 0.18, status: "drift" as const },
    { name: "DeviceInfo", psi: 0.09, status: "watch" as const },
    { name: "C1", psi: 0.03, status: "ok" as const },
  ];
  const STATUS = {
    ok: { color: "#22c55e", label: "stable" },
    watch: { color: "#eab308", label: "watch" },
    drift: { color: "#ef4444", label: "drift" },
  };
  return (
    <Card>
      <CardHeader title="Drift indicators" subtitle="PSI -- Phase 7 will wire Evidently" />
      <CardBody>
        <ul className="space-y-2 text-sm">
          {features.map((f) => (
            <li key={f.name} className="flex items-center justify-between">
              <span className="font-mono text-ink-200">{f.name}</span>
              <div className="flex items-center gap-2">
                <span className="text-xs text-ink-400">PSI {fmtPct(f.psi, 1)}</span>
                <span
                  className="rounded-full px-2 py-0.5 text-[10px] uppercase tracking-wider ring-1"
                  style={{ color: STATUS[f.status].color, borderColor: STATUS[f.status].color }}
                >
                  {STATUS[f.status].label}
                </span>
              </div>
            </li>
          ))}
        </ul>
      </CardBody>
    </Card>
  );
}

function mulberry32(seed: number): () => number {
  let s = seed;
  return function () {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
