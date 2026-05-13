import { Inbox } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "@/lib/cn";

export function Empty({
  title,
  hint,
  icon,
  className,
}: {
  title: ReactNode;
  hint?: ReactNode;
  icon?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-2.5 py-12 text-center",
        className,
      )}
    >
      <div className="grid h-11 w-11 place-items-center rounded-full bg-ink-800/70 ring-1 ring-ink-700/70">
        {icon ?? <Inbox className="h-5 w-5 text-ink-400" strokeWidth={1.75} />}
      </div>
      <div className="text-sm font-medium text-ink-200">{title}</div>
      {hint ? <div className="max-w-[28ch] text-xs text-ink-400">{hint}</div> : null}
    </div>
  );
}
