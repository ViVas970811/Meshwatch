import { AnimatePresence, motion } from "framer-motion";
import { ArrowUpRight } from "lucide-react";
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
    <Card className="overflow-hidden">
      <CardHeader
        title="Live alert stream"
        subtitle="Streaming from /ws/alerts"
        right={
          <span className="inline-flex items-center gap-1.5 text-[11px] text-ink-400">
            <span className="relative inline-flex h-1.5 w-1.5">
              <span className="absolute inset-0 animate-ping rounded-full bg-success/60" />
              <span className="relative inline-block h-1.5 w-1.5 rounded-full bg-success" />
            </span>
            Live
          </span>
        }
      />
      <CardBody className="p-0">
        {alerts.length === 0 ? (
          <Empty
            title="Waiting for alerts"
            hint={emptyHint ?? "Run make demo to seed the feed."}
          />
        ) : (
          <ul className="max-h-[420px] divide-y divide-ink-700/40 overflow-y-auto">
            <AnimatePresence initial={false}>
              {alerts.slice(0, 50).map((a, i) => (
                <motion.li
                  key={`${a.transaction_id}-${i}`}
                  layout
                  initial={{ opacity: 0, x: 8 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
                  className="group flex cursor-pointer items-center gap-3 px-5 py-3 transition-colors hover:bg-ink-800/60"
                  onClick={() =>
                    navigate(`/alerts/inv-${encodeURIComponent(String(a.transaction_id))}`)
                  }
                >
                  <RiskBadge level={a.risk_level} size="sm" />
                  <div className="min-w-0 grow">
                    <div className="flex items-center gap-2 text-sm">
                      <span className="font-mono font-medium text-ink-100">
                        {truncId(a.transaction_id, 8, 4)}
                      </span>
                      <span className="text-ink-600">·</span>
                      <span className="font-medium text-ink-200">
                        {fmtCurrency(a.transaction_amt)}
                      </span>
                    </div>
                    <div className="mt-0.5 text-[11px] text-ink-400">
                      score {fmtScore(a.fraud_score)} · card {truncId(a.card_id ?? "?", 6, 3)} ·{" "}
                      {fmtRelativeTime(a.timestamp)}
                    </div>
                  </div>
                  <ArrowUpRight className="h-4 w-4 text-ink-500 opacity-0 transition-opacity duration-150 group-hover:opacity-100" />
                </motion.li>
              ))}
            </AnimatePresence>
          </ul>
        )}
      </CardBody>
    </Card>
  );
}
