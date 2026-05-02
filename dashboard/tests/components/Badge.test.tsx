import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ActionBadge, Badge, RiskBadge } from "@/components/ui/Badge";

describe("Badge primitives", () => {
  it("Badge renders children", () => {
    render(<Badge>hello</Badge>);
    expect(screen.getByText("hello")).toBeInTheDocument();
  });

  it("RiskBadge renders the level uppercased", () => {
    render(<RiskBadge level="CRITICAL" />);
    expect(screen.getByText("CRITICAL")).toBeInTheDocument();
  });

  it("RiskBadge supports the small size", () => {
    render(<RiskBadge level="HIGH" size="sm" />);
    expect(screen.getByText("HIGH")).toBeInTheDocument();
  });

  it("ActionBadge renders all four canonical actions", () => {
    const actions = ["approve", "review", "decline", "escalate"] as const;
    for (const a of actions) {
      const { unmount } = render(<ActionBadge action={a} />);
      expect(screen.getByText(a)).toBeInTheDocument();
      unmount();
    }
  });
});
