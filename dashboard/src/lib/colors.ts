import type { RecommendedAction, RiskLevel } from "@/api/types";

export const RISK_COLOR: Record<RiskLevel, string> = {
  LOW: "#22c55e",
  MEDIUM: "#eab308",
  HIGH: "#f97316",
  CRITICAL: "#ef4444",
};

export const RISK_BG: Record<RiskLevel, string> = {
  LOW: "bg-risk-low/15 text-risk-low ring-risk-low/30",
  MEDIUM: "bg-risk-medium/15 text-risk-medium ring-risk-medium/30",
  HIGH: "bg-risk-high/15 text-risk-high ring-risk-high/30",
  CRITICAL: "bg-risk-critical/15 text-risk-critical ring-risk-critical/30",
};

export const ACTION_BG: Record<RecommendedAction, string> = {
  approve: "bg-risk-low/15 text-risk-low ring-risk-low/30",
  review: "bg-risk-medium/15 text-risk-medium ring-risk-medium/30",
  decline: "bg-risk-critical/15 text-risk-critical ring-risk-critical/30",
  escalate: "bg-risk-high/15 text-risk-high ring-risk-high/30",
};

/** Return a colour from the risk gradient given a [0,1] score. */
export const scoreColor = (score: number): string => {
  if (score >= 0.9) return RISK_COLOR.CRITICAL;
  if (score >= 0.7) return RISK_COLOR.HIGH;
  if (score >= 0.4) return RISK_COLOR.MEDIUM;
  return RISK_COLOR.LOW;
};
