import type { HTMLAttributes, ReactNode } from "react";

import { cn } from "@/lib/cn";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  interactive?: boolean;
  accent?: boolean;
}

export function Card({ className, interactive, accent, ...props }: CardProps) {
  return (
    <div
      className={cn(
        "surface",
        interactive && "surface-interactive cursor-pointer",
        accent && "ring-accent/40 shadow-glow",
        className,
      )}
      {...props}
    />
  );
}

export function CardHeader({
  title,
  subtitle,
  right,
  className,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  right?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex items-start justify-between gap-3 border-b border-ink-700/60 px-5 py-4",
        className,
      )}
    >
      <div>
        <div className="text-sm font-semibold tracking-tight text-ink-50">{title}</div>
        {subtitle ? (
          <div className="mt-0.5 text-xs text-ink-400">{subtitle}</div>
        ) : null}
      </div>
      {right ? <div className="shrink-0">{right}</div> : null}
    </div>
  );
}

export function CardBody({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("p-5", className)} {...props} />;
}
