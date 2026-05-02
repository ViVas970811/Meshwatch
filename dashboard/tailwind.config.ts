import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Risk-level palette used by Badge / FraudRateGauge / TransactionFeed.
        risk: {
          low: "#22c55e",       // green-500
          medium: "#eab308",    // yellow-500
          high: "#f97316",      // orange-500
          critical: "#ef4444",  // red-500
        },
        ink: {
          900: "#0b1020",
          800: "#11172b",
          700: "#1a2138",
          600: "#2a3253",
          500: "#475070",
          400: "#7c87ad",
          300: "#a9b1ce",
          200: "#cfd4e6",
          100: "#eceff7",
        },
        accent: {
          DEFAULT: "#5b8def",
          light: "#86abff",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "Menlo", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
