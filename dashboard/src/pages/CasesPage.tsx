import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { ActionBadge, RiskBadge } from "@/components/ui/Badge";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Empty } from "@/components/ui/Empty";
import { fmtRelativeTime, truncId } from "@/lib/format";
import { cn } from "@/lib/cn";
import { useCaseStore, type CaseStatus } from "@/store/cases";

const TABS: { value: CaseStatus | "all"; label: string }[] = [
  { value: "all", label: "All" },
  { value: "open", label: "Open" },
  { value: "review", label: "Review" },
  { value: "resolved", label: "Resolved" },
  { value: "escalated", label: "Escalated" },
];

export function CasesPage() {
  const cases = useCaseStore((s) => s.cases);
  const setStatus = useCaseStore((s) => s.setStatus);
  const navigate = useNavigate();
  const [tab, setTab] = useState<CaseStatus | "all">("all");

  const list = useMemo(() => {
    return Object.values(cases)
      .filter((c) => tab === "all" || c.status === tab)
      .sort((a, b) => (a.opened_at < b.opened_at ? 1 : -1));
  }, [cases, tab]);

  const counts = useMemo(() => {
    const out: Record<string, number> = { all: 0, open: 0, review: 0, resolved: 0, escalated: 0 };
    for (const c of Object.values(cases)) {
      out.all += 1;
      out[c.status] += 1;
    }
    return out;
  }, [cases]);

  return (
    <div className="grid gap-6">
      <header>
        <div className="text-xs uppercase tracking-widest text-ink-400">Case management</div>
        <h1 className="text-2xl font-semibold">Investigations</h1>
        <p className="text-sm text-ink-300">
          Built from successful agent investigations on the Alert page. In-memory only — Phase 7
          will persist these.
        </p>
      </header>

      <div className="flex flex-wrap gap-2">
        {TABS.map((t) => (
          <button
            key={t.value}
            onClick={() => setTab(t.value)}
            className={cn(
              "rounded-md px-3 py-1.5 text-xs ring-1 transition-colors",
              tab === t.value
                ? "bg-ink-700 text-ink-100 ring-ink-600"
                : "bg-ink-900/40 text-ink-300 ring-ink-700 hover:bg-ink-700/60",
            )}
          >
            {t.label}{" "}
            <span className="ml-1 text-ink-400">{counts[t.value] ?? 0}</span>
          </button>
        ))}
      </div>

      <Card>
        <CardHeader title="Cases" subtitle={`${list.length} matching`} />
        <CardBody className="p-0">
          {list.length === 0 ? (
            <Empty
              title="No cases yet"
              hint="Open an alert and click 'Run agent' to generate one."
            />
          ) : (
            <table className="w-full text-left text-sm">
              <thead className="border-b border-ink-700/60 text-xs uppercase tracking-wider text-ink-400">
                <tr>
                  <th className="px-5 py-3 font-medium">Alert</th>
                  <th className="px-5 py-3 font-medium">Risk</th>
                  <th className="px-5 py-3 font-medium">Action</th>
                  <th className="px-5 py-3 font-medium">Tools</th>
                  <th className="px-5 py-3 font-medium">Opened</th>
                  <th className="px-5 py-3 font-medium">Status</th>
                  <th className="px-5 py-3 font-medium">Move</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ink-700/60">
                {list.map((c) => (
                  <tr key={c.alert_id} className="hover:bg-ink-700/40">
                    <td className="px-5 py-3 font-mono text-ink-100">
                      <button
                        className="text-accent hover:underline"
                        onClick={() => navigate(`/alerts/${encodeURIComponent(c.alert_id)}`)}
                      >
                        {truncId(c.alert_id, 12, 4)}
                      </button>
                      <div className="text-[11px] text-ink-400">
                        txn {truncId(c.transaction_id, 8, 4)}
                      </div>
                    </td>
                    <td className="px-5 py-3">
                      <RiskBadge level={c.report.risk_level} size="sm" />
                    </td>
                    <td className="px-5 py-3">
                      <ActionBadge action={c.report.recommended_action} />
                    </td>
                    <td className="px-5 py-3 text-ink-300">
                      {c.report.tools_used.length}/8
                    </td>
                    <td className="px-5 py-3 text-ink-300">
                      {fmtRelativeTime(c.opened_at)}
                    </td>
                    <td className="px-5 py-3">
                      <StatusPill status={c.status} />
                    </td>
                    <td className="px-5 py-3">
                      <select
                        value={c.status}
                        onChange={(e) =>
                          setStatus(c.alert_id, e.target.value as CaseStatus)
                        }
                        className="rounded-md bg-ink-900 px-2 py-1 text-xs ring-1 ring-ink-700 focus:outline-none focus:ring-accent"
                      >
                        <option value="open">open</option>
                        <option value="review">review</option>
                        <option value="resolved">resolved</option>
                        <option value="escalated">escalated</option>
                      </select>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </CardBody>
      </Card>
    </div>
  );
}

function StatusPill({ status }: { status: CaseStatus }) {
  const COLOR: Record<CaseStatus, string> = {
    open: "bg-risk-medium/15 text-risk-medium ring-risk-medium/30",
    review: "bg-accent/15 text-accent-light ring-accent/30",
    resolved: "bg-risk-low/15 text-risk-low ring-risk-low/30",
    escalated: "bg-risk-critical/15 text-risk-critical ring-risk-critical/30",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium uppercase tracking-wider ring-1",
        COLOR[status],
      )}
    >
      {status}
    </span>
  );
}
