import { motion, useMotionValue, useTransform, animate } from "framer-motion";
import { useEffect, type ReactNode } from "react";

import { cn } from "@/lib/cn";

interface StatProps {
  label: ReactNode;
  value: ReactNode;
  hint?: ReactNode;
  trend?: ReactNode;
  accent?: "neutral" | "good" | "warn" | "danger";
  delay?: number;
  className?: string;
}

const ACCENT_RULE: Record<NonNullable<StatProps["accent"]>, string> = {
  neutral: "from-accent to-accent-light",
  good: "from-success to-success-light",
  warn: "from-warning to-warning-light",
  danger: "from-danger to-danger-light",
};

export function Stat({
  label,
  value,
  hint,
  trend,
  accent = "neutral",
  delay = 0,
  className,
}: StatProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay, ease: [0.16, 1, 0.3, 1] }}
      className={cn("surface surface-pad relative overflow-hidden", className)}
    >
      {/* Top accent rule -- subtle gradient line that signals stat type. */}
      <div
        className={cn(
          "absolute inset-x-0 top-0 h-px bg-gradient-to-r opacity-70",
          ACCENT_RULE[accent],
        )}
      />
      <div className="stat-label">{label}</div>
      <div className="mt-1.5 flex items-baseline gap-2">
        <div className="stat-value">{value}</div>
        {trend ? (
          <div className="stat-delta text-ink-400">{trend}</div>
        ) : null}
      </div>
      {hint ? <div className="mt-1.5 text-xs text-ink-400">{hint}</div> : null}
    </motion.div>
  );
}

interface AnimatedNumberProps {
  /** Numeric value to count to. */
  value: number;
  /** Decimal places to show. */
  decimals?: number;
  /** Optional prefix (e.g. `"$"`). */
  prefix?: string;
  /** Optional suffix (e.g. `" ms"` or `"%"`). */
  suffix?: string;
  /** Format with thousand separators. */
  thousands?: boolean;
  /** Duration in seconds. */
  duration?: number;
  className?: string;
}

/** Counts up to `value` over `duration` seconds. Re-runs on value change. */
export function AnimatedNumber({
  value,
  decimals = 0,
  prefix = "",
  suffix = "",
  thousands = true,
  duration = 0.8,
  className,
}: AnimatedNumberProps) {
  const mv = useMotionValue(0);
  const rounded = useTransform(mv, (latest) => {
    const n = Number(latest);
    const opts: Intl.NumberFormatOptions = {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
      useGrouping: thousands,
    };
    return `${prefix}${n.toLocaleString(undefined, opts)}${suffix}`;
  });

  useEffect(() => {
    const controls = animate(mv, value, {
      duration,
      ease: [0.16, 1, 0.3, 1],
    });
    return () => controls.stop();
  }, [value, duration, mv]);

  return <motion.span className={cn("tabular-nums", className)}>{rounded}</motion.span>;
}
