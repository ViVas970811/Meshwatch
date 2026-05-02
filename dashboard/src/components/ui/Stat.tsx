import type { ReactNode } from "react";

import { cn } from "@/lib/cn";

export function Stat({
  label,
  value,
  hint,
  trend,
  className,
}: {
  label: ReactNode;
  value: ReactNode;
  hint?: ReactNode;
  trend?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("surface surface-pad", className)}>
      <div className="stat-label">{label}</div>
      <div className="mt-1 flex items-baseline gap-2">
        <div className="stat-value">{value}</div>
        {trend ? <div className="text-xs text-ink-400">{trend}</div> : null}
      </div>
      {hint ? <div className="mt-1 text-xs text-ink-400">{hint}</div> : null}
    </div>
  );
}
