/** Shared Recharts visual tokens so every chart in the app feels cohesive.
 *
 *  Importing components copy these into their <XAxis/>, <YAxis/>, <Tooltip/>,
 *  <CartesianGrid/> props. Avoids hard-coded "#1a2138" / "#7c87ad" scattered
 *  across each chart file.
 */

export const CHART_TOKENS = {
  axis: {
    tick: { fontSize: 10, fill: "#7c87ad", fontFamily: "var(--font-mono, monospace)" },
    axisLine: false as const,
    tickLine: false as const,
  },
  grid: {
    stroke: "rgba(45, 56, 90, 0.55)",
    strokeDasharray: "3 4",
    vertical: false as const,
  },
  tooltip: {
    contentStyle: {
      background: "rgba(15, 21, 44, 0.96)",
      border: "1px solid rgba(45, 56, 90, 0.9)",
      borderRadius: 10,
      padding: "8px 12px",
      fontSize: 12,
      boxShadow: "0 10px 30px -10px rgba(0,0,0,0.6)",
      backdropFilter: "blur(8px)",
      WebkitBackdropFilter: "blur(8px)",
    } as React.CSSProperties,
    cursor: { fill: "rgba(99, 102, 241, 0.06)" },
    itemStyle: { color: "#eceff7" } as React.CSSProperties,
    labelStyle: { color: "#a9b1ce", marginBottom: 4 } as React.CSSProperties,
  },
  accents: {
    primary: "#6366f1",
    primarySoft: "rgba(99, 102, 241, 0.5)",
    primaryHairline: "rgba(99, 102, 241, 0.15)",
    success: "#10b981",
    warning: "#f59e0b",
    danger: "#ef4444",
  },
};

/** Gradient stops for an Area chart fill -- use with a `<defs>` block. */
export const AREA_GRADIENT_STOPS = [
  { offset: "0%", color: "#6366f1", opacity: 0.45 },
  { offset: "55%", color: "#6366f1", opacity: 0.15 },
  { offset: "100%", color: "#6366f1", opacity: 0 },
];
