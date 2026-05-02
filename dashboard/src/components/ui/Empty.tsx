import type { ReactNode } from "react";

import { cn } from "@/lib/cn";

export function Empty({
  title,
  hint,
  className,
}: {
  title: ReactNode;
  hint?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-1 py-10 text-center text-ink-400",
        className,
      )}
    >
      <div className="text-sm font-medium text-ink-300">{title}</div>
      {hint ? <div className="text-xs text-ink-400">{hint}</div> : null}
    </div>
  );
}
