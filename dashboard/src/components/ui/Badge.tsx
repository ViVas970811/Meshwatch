import type { ReactNode } from "react";

import type { RecommendedAction, RiskLevel } from "@/api/types";
import { cn } from "@/lib/cn";
import { ACTION_BG, RISK_BG } from "@/lib/colors";

interface BadgeProps {
  children?: ReactNode;
  className?: string;
}

export function Badge({ children, className }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ring-1",
        "bg-ink-700/40 text-ink-200 ring-ink-600",
        className,
      )}
    >
      {children}
    </span>
  );
}

export function RiskBadge({
  level,
  className,
  size = "md",
}: {
  level: RiskLevel;
  className?: string;
  size?: "sm" | "md";
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full font-medium ring-1 uppercase tracking-wider",
        RISK_BG[level],
        size === "sm" ? "px-1.5 py-0.5 text-[10px]" : "px-2 py-0.5 text-[11px]",
        className,
      )}
    >
      {level}
    </span>
  );
}

export function ActionBadge({
  action,
  className,
}: {
  action: RecommendedAction;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 uppercase tracking-wider",
        ACTION_BG[action],
        className,
      )}
    >
      {action}
    </span>
  );
}
