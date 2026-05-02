import { useNavigate } from "react-router-dom";

import type { FraudAlert } from "@/api/types";
import { RiskBadge } from "@/components/ui/Badge";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Empty } from "@/components/ui/Empty";
import { fmtCurrency, fmtRelativeTime, fmtScore, truncId } from "@/lib/format";

export function TransactionFeed({
  alerts,
  emptyHint,
}: {
  alerts: FraudAlert[];
  emptyHint?: string;
}) {
  const navigate = useNavigate();

  return (
    <Card>
      <CardHeader title="Live alerts" subtitle="WebSocket /ws/alerts" />
      <CardBody className="p-0">
        {alerts.length === 0 ? (
          <Empty title="Waiting for alerts" hint={emptyHint ?? "Run make demo-stream"} />
        ) : (
          <ul className="divide-y divide-ink-700/60">
            {alerts.slice(0, 50).map((a, i) => (
              <li
                key={`${a.transaction_id}-${i}`}
                className="flex cursor-pointer items-center gap-3 px-5 py-3 hover:bg-ink-700/40"
                onClick={() => navigate(`/alerts/inv-${encodeURIComponent(String(a.transaction_id))}`)}
              >
                <RiskBadge level={a.risk_level} size="sm" />
                <div className="min-w-0 grow">
                  <div className="flex items-center gap-2 text-sm">
                    <span className="font-mono text-ink-200">
                      txn {truncId(a.transaction_id, 8, 4)}
                    </span>
                    <span className="text-ink-500">•</span>
                    <span className="text-ink-300">{fmtCurrency(a.transaction_amt)}</span>
                  </div>
                  <div className="text-[11px] text-ink-400">
                    score {fmtScore(a.fraud_score)} · card {truncId(a.card_id ?? "?", 6, 3)} ·{" "}
                    {fmtRelativeTime(a.timestamp)}
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </CardBody>
    </Card>
  );
}
