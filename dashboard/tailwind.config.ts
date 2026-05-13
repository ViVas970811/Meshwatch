import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Risk-level palette used by Badge / FraudRateGauge / TransactionFeed.
        risk: {
          low: "#10b981", // emerald-500
          medium: "#f59e0b", // amber-500
          high: "#f97316", // orange-500
          critical: "#ef4444", // red-500
        },
        // Refined neutral scale: cooler at the top, warmer mids, true whites.
        // Calibrated against Linear / Stripe / Vercel for that "business-grade
        // dark UI" feel rather than the high-contrast cobalt of v1.
        ink: {
          950: "#070a16",
          900: "#0c1124",
          850: "#0f152c",
          800: "#141b35",
          750: "#1a2240",
          700: "#202a4a",
          600: "#2d385a",
          500: "#475070",
          400: "#7c87ad",
          300: "#a9b1ce",
          200: "#cfd4e6",
          100: "#eceff7",
          50: "#f7f8fc",
        },
        accent: {
          DEFAULT: "#6366f1", // indigo-500 -- more sophisticated than v1 blue
          light: "#818cf8",
          dark: "#4f46e5",
          glow: "#7c3aed",
        },
        success: {
          DEFAULT: "#10b981",
          light: "#34d399",
        },
        warning: {
          DEFAULT: "#f59e0b",
          light: "#fbbf24",
        },
        danger: {
          DEFAULT: "#ef4444",
          light: "#f87171",
        },
      },
      fontFamily: {
        sans: [
          "Inter var",
          "Inter",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: ["JetBrains Mono", "Fira Code", "ui-monospace", "Menlo", "monospace"],
      },
      fontSize: {
        // Slightly tighter line-heights for headings; better display feel.
        "display-sm": ["1.5rem", { lineHeight: "1.85rem", letterSpacing: "-0.018em" }],
        "display-md": ["1.875rem", { lineHeight: "2.25rem", letterSpacing: "-0.02em" }],
        "display-lg": ["2.25rem", { lineHeight: "2.5rem", letterSpacing: "-0.022em" }],
      },
      letterSpacing: {
        kicker: "0.18em",
      },
      borderRadius: {
        "2xl": "1rem",
        "3xl": "1.25rem",
      },
      boxShadow: {
        card: "0 1px 0 0 rgba(255,255,255,0.04) inset, 0 1px 2px 0 rgba(0,0,0,0.4)",
        elevated:
          "0 1px 0 0 rgba(255,255,255,0.06) inset, 0 8px 24px -8px rgba(0,0,0,0.5), 0 2px 4px -1px rgba(0,0,0,0.3)",
        glow: "0 0 0 1px rgba(99,102,241,0.3), 0 8px 24px -8px rgba(99,102,241,0.5)",
      },
      backgroundImage: {
        // Subtle radial used on the layout shell + featured surfaces.
        "grid-glow":
          "radial-gradient(at 30% -10%, rgba(99,102,241,0.10), transparent 55%), radial-gradient(at 80% 110%, rgba(129,140,248,0.06), transparent 50%)",
        "card-surface":
          "linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0) 50%)",
      },
      animation: {
        shimmer: "shimmer 1.8s ease-in-out infinite",
        "pulse-soft": "pulse-soft 2.4s ease-in-out infinite",
        "fade-in": "fade-in 0.3s ease-out both",
        "slide-up": "slide-up 0.4s cubic-bezier(0.16, 1, 0.3, 1) both",
      },
      keyframes: {
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        "pulse-soft": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.5" },
        },
        "fade-in": {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        "slide-up": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      transitionTimingFunction: {
        "out-expo": "cubic-bezier(0.16, 1, 0.3, 1)",
      },
    },
  },
  plugins: [],
};

export default config;
