import { describe, expect, it, vi } from "vitest";

import {
  fmtCompact,
  fmtCurrency,
  fmtMs,
  fmtPct,
  fmtRelativeTime,
  fmtScore,
  truncId,
} from "@/lib/format";

describe("format helpers", () => {
  it("currency formats USD with 2 decimal places", () => {
    expect(fmtCurrency(1234.5)).toBe("$1,234.50");
    expect(fmtCurrency(0)).toBe("$0.00");
  });

  it("currency returns dash for null/undefined", () => {
    expect(fmtCurrency(null)).toBe("—");
    expect(fmtCurrency(undefined)).toBe("—");
    expect(fmtCurrency(Number.NaN)).toBe("—");
  });

  it("percent formats with one decimal by default", () => {
    expect(fmtPct(0.123)).toBe("12.3%");
    expect(fmtPct(1)).toBe("100.0%");
  });

  it("percent honours digits", () => {
    expect(fmtPct(0.12345, 3)).toBe("12.345%");
  });

  it("score formats to 3 decimals", () => {
    expect(fmtScore(0.5)).toBe("0.500");
    expect(fmtScore(0.123456)).toBe("0.123");
  });

  it("ms switches to seconds above 1000", () => {
    expect(fmtMs(450.7)).toBe("450.7 ms");
    expect(fmtMs(3000)).toBe("3.00s");
  });

  it("compact formats large numbers", () => {
    expect(fmtCompact(1500)).toBe("1.5K");
    expect(fmtCompact(2_000_000)).toBe("2M");
  });

  it("relative time handles a recent and old timestamp", () => {
    const now = new Date("2026-05-01T12:00:00Z").getTime();
    vi.spyOn(Date, "now").mockReturnValue(now);
    expect(fmtRelativeTime("2026-05-01T11:59:30Z")).toBe("30s ago");
    expect(fmtRelativeTime("2026-05-01T11:00:00Z")).toBe("1h ago");
    expect(fmtRelativeTime("2026-04-30T12:00:00Z")).toBe("1d ago");
  });

  it("truncId leaves short ids alone and ellipsises long ones", () => {
    expect(truncId("abc")).toBe("abc");
    expect(truncId("0123456789abcdef")).toBe("012345…cdef");
    expect(truncId(null)).toBe("—");
  });
});
