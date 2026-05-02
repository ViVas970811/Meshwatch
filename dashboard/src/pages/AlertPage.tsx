import { useMutation, useQuery } from "@tanstack/react-query";
import { ArrowLeft, Wand2 } from "lucide-react";
import { useMemo } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { api } from "@/api/client";
import type {
  FraudAlert,
  FraudPrediction,
  InvestigationReport,
  TransactionRequest,
} from "@/api/types";
import { NetworkGraph, type GraphLink, type GraphNode } from "@/components/network/NetworkGraph";
import { ActionBadge, RiskBadge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Empty } from "@/components/ui/Empty";
import { Skeleton } from "@/components/ui/Skeleton";
import { fmtCurrency, fmtMs, fmtScore } from "@/lib/format";
import { useCaseStore } from "@/store/cases";

export function AlertPage() {
  const { alertId = "" } = useParams<{ alertId: string }>();
  const navigate = useNavigate();
  const upsertCase = useCaseStore((s) => s.upsertCase);
  const cases = useCaseStore((s) => s.cases);

  const txnId = alertId.replace(/^inv-/, "");

  const { data: recent } = useQuery({ queryKey: ["recent"], queryFn: api.recent });

  const prediction = useMemo(
    () => recent?.predictions.find((p) => String(p.transaction_id) === txnId),
    [recent, txnId],
  );
  const alert = useMemo(
    () => recent?.alerts.find((a) => String(a.transaction_id) === txnId),
    [recent, txnId],
  );

  const existingReport: InvestigationReport | undefined = cases[alertId]?.report;

  const investigateMu = useMutation({
    mutationFn: () => {
      const txn: TransactionRequest = {
        transaction_id: txnId,
        transaction_dt: 86_400,
        transaction_amt: alert?.transaction_amt ?? prediction?.fraud_score ?? 1,
        card1: typeof alert?.card_id === "number" ? alert.card_id : 13926,
      };
      return api.investigate({
        transaction: txn,
        prediction,
        alert_id: alertId,
      });
    },
    onSuccess: (rep) => upsertCase(rep),
  });

  const report = investigateMu.data ?? existingReport;

  return (
    <div className="grid gap-6">
      <div className="flex items-center gap-3">
        <Button variant="ghost" size="sm" onClick={() => navigate(-1)}>
          <ArrowLeft className="h-4 w-4" /> Back
        </Button>
        <div>
          <div className="text-xs uppercase tracking-widest text-ink-400">Alert investigation</div>
          <h1 className="font-mono text-xl">{alertId}</h1>
        </div>
      </div>

      <section className="grid gap-4 lg:grid-cols-3">
        <AlertDetail alert={alert} prediction={prediction} />
        <RiskFactorChart prediction={prediction} report={report} />
        <CardTimeline alert={alert} prediction={prediction} />
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <GraphExplorer report={report} alert={alert} />
        <InvestigationPanel
          report={report}
          loading={investigateMu.isPending}
          error={investigateMu.error as Error | null}
          onRun={() => investigateMu.mutate()}
        />
      </section>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Subcomponents
// ---------------------------------------------------------------------------

function AlertDetail({
  alert,
  prediction,
}: {
  alert: FraudAlert | undefined;
  prediction: FraudPrediction | undefined;
}) {
  return (
    <Card>
      <CardHeader title="Transaction" subtitle="from /api/v1/recent" />
      <CardBody className="space-y-2 text-sm">
        <Row label="Score">{prediction ? fmtScore(prediction.fraud_score) : "—"}</Row>
        <Row label="Risk">
          {prediction?.risk_level ? <RiskBadge level={prediction.risk_level} /> : "—"}
        </Row>
        <Row label="Amount">{alert ? fmtCurrency(alert.transaction_amt) : "—"}</Row>
        <Row label="Card id">
          <code className="font-mono text-ink-200">{alert?.card_id ?? "—"}</code>
        </Row>
        <Row label="Threshold">{prediction ? fmtScore(prediction.threshold) : "—"}</Row>
        <Row label="Latency">
          {prediction ? fmtMs(Number(prediction.latency_ms?.total_ms ?? 0)) : "—"}
        </Row>
        <Row label="Model">
          <code className="text-xs">{prediction?.model_version ?? "—"}</code>
        </Row>
      </CardBody>
    </Card>
  );
}

function RiskFactorChart({
  prediction,
  report,
}: {
  prediction: FraudPrediction | undefined;
  report: InvestigationReport | undefined;
}) {
  const features = useMemo(() => {
    if (prediction?.top_features?.length) return prediction.top_features.slice(0, 8);
    return [];
  }, [prediction]);

  return (
    <Card>
      <CardHeader title="Risk factors" subtitle="SHAP top contributions" />
      <CardBody>
        {features.length === 0 ? (
          <Empty
            title="SHAP not populated"
            hint={
              report
                ? "Tabular SHAP wasn't logged with this prediction. Use the agent's matched_patterns below."
                : "Run the agent to populate richer evidence."
            }
          />
        ) : (
          <ul className="space-y-2">
            {features.map((f) => {
              const max = Math.max(...features.map((x) => Math.abs(x.contribution)));
              const w = max ? Math.abs(f.contribution) / max : 0;
              const positive = f.contribution >= 0;
              return (
                <li key={f.feature} className="text-xs">
                  <div className="flex justify-between">
                    <span className="text-ink-200">{f.feature}</span>
                    <span className={positive ? "text-risk-critical" : "text-risk-low"}>
                      {(f.contribution >= 0 ? "+" : "") + f.contribution.toFixed(3)}
                    </span>
                  </div>
                  <div className="mt-1 h-1.5 overflow-hidden rounded bg-ink-700/60">
                    <div
                      className={positive ? "bg-risk-critical/80" : "bg-risk-low/80"}
                      style={{ width: `${w * 100}%`, height: "100%" }}
                    />
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </CardBody>
    </Card>
  );
}

function CardTimeline({
  alert,
  prediction,
}: {
  alert: FraudAlert | undefined;
  prediction: FraudPrediction | undefined;
}) {
  return (
    <Card>
      <CardHeader title="Card timeline" subtitle="recent on same card_id" />
      <CardBody>
        {!alert && !prediction ? (
          <Empty title="No alert context" />
        ) : (
          <ol className="relative space-y-3 border-l border-ink-700/60 pl-4 text-xs">
            <li>
              <div className="absolute -left-[5px] mt-1 h-2 w-2 rounded-full bg-risk-high" />
              <div className="text-ink-200">This transaction</div>
              <div className="text-ink-400">
                score {prediction ? fmtScore(prediction.fraud_score) : "—"}, amount{" "}
                {alert ? fmtCurrency(alert.transaction_amt) : "—"}
              </div>
            </li>
            <li>
              <div className="absolute -left-[5px] mt-1 h-2 w-2 rounded-full bg-ink-500" />
              <div className="text-ink-300">Phase 7 will wire Feast online-store history here.</div>
            </li>
          </ol>
        )}
      </CardBody>
    </Card>
  );
}

function GraphExplorer({
  report,
  alert,
}: {
  report: InvestigationReport | undefined;
  alert: FraudAlert | undefined;
}) {
  const nodes: GraphNode[] = useMemo(() => {
    const cardId = String(alert?.card_id ?? "card");
    const ns: GraphNode[] = [
      {
        id: cardId,
        label: `card ${cardId}`,
        risk_level: report?.risk_level ?? "MEDIUM",
        group: "card",
        size: 9,
      },
    ];
    for (const e of report?.entity_risks ?? []) {
      ns.push({
        id: `${e.entity_type}:${e.entity_id}`,
        label: `${e.entity_type} ${e.entity_id}`,
        risk_level:
          e.risk_score >= 0.9 ? "CRITICAL" : e.risk_score >= 0.7 ? "HIGH" : e.risk_score >= 0.4 ? "MEDIUM" : "LOW",
        group: e.entity_type,
        size: 5 + Math.round(e.risk_score * 5),
      });
    }
    return ns;
  }, [alert, report]);

  const links: GraphLink[] = useMemo(() => {
    const cardId = String(alert?.card_id ?? "card");
    return (report?.entity_risks ?? []).map((e) => ({
      source: cardId,
      target: `${e.entity_type}:${e.entity_id}`,
      weight: e.risk_score,
    }));
  }, [alert, report]);

  return (
    <Card>
      <CardHeader title="Graph explorer" subtitle="card + agent-derived entities" />
      <CardBody className="p-0">
        {nodes.length <= 1 ? (
          <div className="px-5 py-10">
            <Empty title="Graph appears after investigation" hint="Run the agent to populate." />
          </div>
        ) : (
          <NetworkGraph nodes={nodes} links={links} height={360} highlightId={String(alert?.card_id)} />
        )}
      </CardBody>
    </Card>
  );
}

function InvestigationPanel({
  report,
  loading,
  error,
  onRun,
}: {
  report: InvestigationReport | undefined;
  loading: boolean;
  error: Error | null;
  onRun: () => void;
}) {
  return (
    <Card>
      <CardHeader
        title="Agent investigation"
        subtitle={report ? `model: ${report.model}` : "POST /api/v1/investigate"}
        right={
          <Button onClick={onRun} disabled={loading} variant="primary">
            <Wand2 className="h-4 w-4" />
            {loading ? "Investigating..." : report ? "Re-run" : "Run agent"}
          </Button>
        }
      />
      <CardBody>
        {loading ? (
          <div className="space-y-2">
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-4 w-2/3" />
            <Skeleton className="h-4 w-1/2" />
          </div>
        ) : error ? (
          <div className="text-sm text-risk-critical">
            Investigation failed: {error.message}
          </div>
        ) : !report ? (
          <Empty
            title="No investigation yet"
            hint="Click Run agent to invoke the LangGraph orchestrator."
          />
        ) : (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-2">
              <RiskBadge level={report.risk_level} />
              <ActionBadge action={report.recommended_action} />
              {report.requires_human_review ? (
                <span className="rounded-full bg-risk-high/15 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-risk-high ring-1 ring-risk-high/30">
                  human review
                </span>
              ) : null}
              <span className="text-xs text-ink-400">
                {report.tools_used.length}/8 tools • {fmtMs(report.elapsed_ms)} • conf{" "}
                {fmtScore(report.confidence)}
              </span>
            </div>

            <div>
              <div className="stat-label mb-1">Summary</div>
              <p className="text-sm text-ink-200">{report.summary}</p>
            </div>

            <div>
              <div className="stat-label mb-1">Narrative</div>
              <p className="whitespace-pre-wrap text-sm leading-relaxed text-ink-200">
                {report.narrative}
              </p>
            </div>

            {report.matched_patterns?.length ? (
              <div>
                <div className="stat-label mb-1">Matched patterns</div>
                <ul className="space-y-1 text-sm">
                  {report.matched_patterns.map((p, i) => (
                    <li key={`${p.name}-${i}`} className="flex justify-between gap-3">
                      <span className="text-ink-200">{p.name}</span>
                      <span className="text-ink-400">conf {fmtScore(p.confidence)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        )}
      </CardBody>
    </Card>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-ink-700/40 pb-2 last:border-0 last:pb-0">
      <span className="stat-label">{label}</span>
      <span className="text-ink-100">{children}</span>
    </div>
  );
}
