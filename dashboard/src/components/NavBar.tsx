import { useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { Activity, Briefcase, LineChart, Network, ShieldCheck } from "lucide-react";
import { NavLink, useLocation } from "react-router-dom";

import { api } from "@/api/client";
import { cn } from "@/lib/cn";

const NAV = [
  { to: "/dashboard", label: "Monitor", icon: Activity },
  { to: "/network", label: "Network", icon: Network },
  { to: "/model", label: "Model", icon: LineChart },
  { to: "/cases", label: "Cases", icon: Briefcase },
];

export function NavBar({ wsStatus }: { wsStatus: "connecting" | "open" | "closed" }) {
  const location = useLocation();

  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 5000,
  });

  return (
    <header className="sticky top-0 z-30 border-b border-ink-700/40 bg-ink-950/75 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1400px] items-center justify-between gap-6 px-6 py-3">
        {/* Brand */}
        <NavLink to="/dashboard" className="group flex items-center gap-3">
          <div
            className="relative grid h-9 w-9 place-items-center overflow-hidden rounded-xl
              bg-gradient-to-br from-accent via-accent-light to-accent-glow
              shadow-glow ring-1 ring-accent-light/40 transition-transform
              duration-300 ease-out-expo group-hover:scale-[1.04]"
          >
            <ShieldCheck className="relative h-4.5 w-4.5 text-white" strokeWidth={2.4} />
          </div>
          <div className="leading-tight">
            <div className="text-[15px] font-semibold tracking-tight text-ink-50">Meshwatch</div>
            <div className="text-[10px] font-medium uppercase tracking-kicker text-ink-400">
              Fraud Intelligence
            </div>
          </div>
        </NavLink>

        {/* Nav -- shared layoutId on the active pill makes it slide between
            tabs with a single Tailwind ease. */}
        <nav className="relative flex items-center gap-1 rounded-full bg-ink-850/60 p-1 ring-1 ring-ink-700/60">
          {NAV.map(({ to, label, icon: Icon }) => {
            const active = location.pathname.startsWith(to);
            return (
              <NavLink
                key={to}
                to={to}
                className={cn(
                  "relative z-10 flex items-center gap-2 rounded-full px-3.5 py-1.5 text-sm font-medium transition-colors",
                  active ? "text-ink-50" : "text-ink-300 hover:text-ink-100",
                )}
              >
                {active ? (
                  <motion.span
                    layoutId="nav-active-pill"
                    className="absolute inset-0 -z-10 rounded-full bg-ink-700/90 shadow-card ring-1 ring-ink-600/80"
                    transition={{ type: "spring", stiffness: 380, damping: 32 }}
                  />
                ) : null}
                <Icon className="h-3.5 w-3.5" strokeWidth={2} />
                {label}
              </NavLink>
            );
          })}
        </nav>

        {/* Health pills */}
        <div className="flex items-center gap-2">
          <StatusPill label={wsLabel(wsStatus)} tone={wsTone(wsStatus)} />
          <StatusPill
            label={health?.model_loaded ? "Model" : "No model"}
            tone={health?.model_loaded ? "ok" : "warn"}
          />
          <StatusPill
            label={health?.kafka_connected ? "Kafka" : "Buffer"}
            tone={health?.kafka_connected ? "ok" : "neutral"}
          />
        </div>
      </div>
    </header>
  );
}

function wsLabel(s: "connecting" | "open" | "closed"): string {
  return s === "open" ? "Live" : s === "connecting" ? "Connecting" : "Offline";
}

function wsTone(s: "connecting" | "open" | "closed"): StatusTone {
  return s === "open" ? "ok" : s === "connecting" ? "warn" : "danger";
}

type StatusTone = "ok" | "warn" | "danger" | "neutral";

const TONE_COLOR: Record<StatusTone, string> = {
  ok: "#10b981",
  warn: "#f59e0b",
  danger: "#ef4444",
  neutral: "#7c87ad",
};

function StatusPill({ label, tone }: { label: string; tone: StatusTone }) {
  const color = TONE_COLOR[tone];
  const animate = tone === "ok" || tone === "warn";
  return (
    <AnimatePresence mode="popLayout">
      <motion.div
        key={`${label}-${tone}`}
        initial={{ opacity: 0, y: -4 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 4 }}
        transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
        className="flex items-center gap-2 rounded-full bg-ink-850/80 px-2.5 py-1 text-[11px] font-medium text-ink-200 ring-1 ring-ink-700/80"
      >
        <span className="relative inline-flex h-1.5 w-1.5">
          {animate ? (
            <motion.span
              animate={{ opacity: [0.4, 0.9, 0.4], scale: [0.9, 1.1, 0.9] }}
              transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
              className="absolute inset-0 rounded-full"
              style={{ backgroundColor: color }}
            />
          ) : null}
          <span
            className="relative inline-block h-1.5 w-1.5 rounded-full"
            style={{ backgroundColor: color, boxShadow: `0 0 8px ${color}` }}
          />
        </span>
        {label}
      </motion.div>
    </AnimatePresence>
  );
}
