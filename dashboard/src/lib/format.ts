/** Pretty-printing helpers for numbers, currency, and timestamps. */

const CURRENCY = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2,
});

const COMPACT = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
});

export const fmtCurrency = (n: number | null | undefined): string =>
  n === null || n === undefined || Number.isNaN(n) ? "—" : CURRENCY.format(n);

export const fmtCompact = (n: number | null | undefined): string =>
  n === null || n === undefined || Number.isNaN(n) ? "—" : COMPACT.format(n);

export const fmtPct = (n: number | null | undefined, digits = 1): string => {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return `${(n * 100).toFixed(digits)}%`;
};

export const fmtScore = (n: number | null | undefined): string => {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toFixed(3);
};

export const fmtMs = (n: number | null | undefined): string => {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  if (n >= 1000) return `${(n / 1000).toFixed(2)}s`;
  return `${n.toFixed(1)} ms`;
};

export const fmtRelativeTime = (iso: string | undefined | null): string => {
  if (!iso) return "—";
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return "—";
  const diff = (Date.now() - t) / 1000;
  if (diff < 1) return "just now";
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86_400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86_400)}d ago`;
};

export const truncId = (id: string | number | null | undefined, head = 6, tail = 4): string => {
  if (id === null || id === undefined) return "—";
  const s = String(id);
  if (s.length <= head + tail + 1) return s;
  return `${s.slice(0, head)}…${s.slice(-tail)}`;
};
