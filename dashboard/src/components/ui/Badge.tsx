import type { ReactNode } from "react";

import type { RecommendedAction, RiskLevel } from "@/api/types";
import { cn } from "@/lib/cn";
import { ACTION_BG, RISK_BG } from "@/lib/colors";

interface BadgeProps {
  children?: ReactNode;
  className?: string;
  pulse?: boolean;
}

export function Badge({ children, className, pulse }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[11px] font-medium ring-1 transition-colors",
        "bg-ink-700/50 text-ink-200 ring-ink-600/80",
        className,
      )}
    >
      {pulse ? (
        <span className="relative inline-flex h-1.5 w-1.5 shrink-0">
          <span className="absolute inset-0 animate-ping rounded-full bg-current opacity-60" />
          <span className="relative inline-block h-1.5 w-1.5 rounded-full bg-current" />
        </span>
      ) : null}
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
        "inline-flex items-center gap-1 rounded-full font-semibold uppercase tracking-wider ring-1 transition-colors",
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
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wider ring-1 transition-colors",
        ACTION_BG[action],
        className,
      )}
    >
      {action}
    </span>
  );
}
