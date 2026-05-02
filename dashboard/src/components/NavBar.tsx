import { useQuery } from "@tanstack/react-query";
import { Activity, AlertTriangle, Briefcase, LineChart, Network } from "lucide-react";
import { NavLink } from "react-router-dom";

import { api } from "@/api/client";
import { cn } from "@/lib/cn";

const NAV = [
  { to: "/dashboard", label: "Monitor", icon: Activity },
  { to: "/network", label: "Network", icon: Network },
  { to: "/model", label: "Model", icon: LineChart },
  { to: "/cases", label: "Cases", icon: Briefcase },
];

export function NavBar({ wsStatus }: { wsStatus: "connecting" | "open" | "closed" }) {
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 5000,
  });

  return (
    <header className="sticky top-0 z-30 border-b border-ink-700/60 bg-ink-900/80 backdrop-blur">
      <div className="mx-auto flex max-w-[1400px] items-center justify-between gap-6 px-6 py-3">
        <div className="flex items-center gap-3">
          <div className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-accent to-accent-light shadow-md shadow-accent/40">
            <AlertTriangle className="h-4 w-4 text-white" />
          </div>
          <div>
            <div className="text-sm font-semibold leading-tight">Meshwatch</div>
            <div className="text-[10px] uppercase tracking-widest text-ink-400">
              Fraud Detection
            </div>
          </div>
        </div>

        <nav className="flex items-center gap-1">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-2 rounded-md px-3 py-1.5 text-sm transition-colors",
                  isActive
                    ? "bg-ink-700 text-ink-100 ring-1 ring-ink-600"
                    : "text-ink-300 hover:bg-ink-700/60",
                )
              }
            >
              <Icon className="h-4 w-4" />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="flex items-center gap-3">
          <StatusDot
            label={wsLabel(wsStatus)}
            color={wsStatus === "open" ? "#22c55e" : wsStatus === "connecting" ? "#eab308" : "#ef4444"}
          />
          <StatusDot
            label={health?.model_loaded ? "model" : "no model"}
            color={health?.model_loaded ? "#22c55e" : "#eab308"}
          />
          <StatusDot
            label={health?.kafka_connected ? "kafka" : "in-mem"}
            color={health?.kafka_connected ? "#22c55e" : "#7c87ad"}
          />
        </div>
      </div>
    </header>
  );
}

function wsLabel(s: "connecting" | "open" | "closed"): string {
  return s === "open" ? "ws live" : s === "connecting" ? "ws…" : "ws off";
}

function StatusDot({ label, color }: { label: string; color: string }) {
  return (
    <div className="flex items-center gap-2 rounded-full bg-ink-800/70 px-2 py-1 text-[11px] text-ink-200 ring-1 ring-ink-700">
      <span
        className="inline-block h-2 w-2 rounded-full"
        style={{ backgroundColor: color, boxShadow: `0 0 8px ${color}` }}
      />
      {label}
    </div>
  );
}
