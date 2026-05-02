import "@testing-library/jest-dom/vitest";

import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

// Auto-tear-down between tests so DOM doesn't leak.
afterEach(() => cleanup());

// jsdom doesn't ship ResizeObserver -- recharts depends on it.
class ResizeObserverPolyfill {
  observe() {}
  unobserve() {}
  disconnect() {}
}
if (!(globalThis as any).ResizeObserver) {
  (globalThis as any).ResizeObserver = ResizeObserverPolyfill;
}

// Stub matchMedia (a few Radix utilities probe it).
if (!window.matchMedia) {
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }));
}

// jsdom HTMLCanvasElement.getContext returns null; recharts + react-force-graph
// don't need real canvas output for component-tree tests.
HTMLCanvasElement.prototype.getContext = vi.fn() as unknown as HTMLCanvasElement["getContext"];
