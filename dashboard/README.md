# Meshwatch Dashboard (Phase 6)

React 18 + TypeScript + Vite frontend for the Meshwatch fraud-detection
pipeline. Five pages, all wired to the FastAPI app from Phase 4 + 5:

| Route | Page | Backend |
| --- | --- | --- |
| `/dashboard` | Real-time monitor (live alerts, score distribution, latency) | `WS /ws/alerts`, `GET /api/v1/recent`, `GET /api/v1/health` |
| `/alerts/:alertId` | Alert investigation (run agent + view report) | `GET /api/v1/recent`, `POST /api/v1/investigate` |
| `/network` | Force-directed fraud network graph | (synthetic until Phase 7 export endpoint) |
| `/model` | Model performance + feature importance + drift | `GET /api/v1/model/info` |
| `/cases` | Case management (open / review / resolved / escalated) | in-memory (Zustand store) |

## Quick start

```bash
# 1. Install deps
make dashboard-install        # alias for `cd dashboard && npm install`

# 2. Start the FastAPI backend in another terminal
make serve                    # http://localhost:8000

# 3. Start the dev server
make dashboard-dev            # http://localhost:5173 (proxies /api + /ws to :8000)

# Tests + type-check
make dashboard-test           # vitest run
make dashboard-lint           # tsc -b
make dashboard-build          # production bundle -> dashboard/dist/
```

## Tech stack (per plan)

* Vite 6 + React 18 + TypeScript
* TanStack Query (server-state caching, polling)
* Zustand (alert ring buffer + case store)
* Recharts (gauge, histogram, line/bar)
* react-force-graph-2d (network view)
* Radix UI primitives + Tailwind CSS

## Folder layout

```
dashboard/
├── src/
│   ├── api/           # types.ts, client.ts (fetch), ws.ts (WebSocket hook)
│   ├── components/
│   │   ├── ui/        # Card, Badge, Button, Stat, Skeleton, Empty
│   │   ├── charts/    # FraudRateGauge, ScoreDistributionChart, LatencyMonitor
│   │   ├── feed/      # TransactionFeed, AlertCounter
│   │   └── network/   # NetworkGraph (force-directed)
│   ├── pages/         # 5 pages, one file each
│   ├── store/         # alerts.ts, cases.ts (Zustand)
│   ├── lib/           # cn.ts, format.ts, colors.ts
│   ├── styles/        # globals.css (Tailwind)
│   ├── App.tsx
│   └── main.tsx
└── tests/             # 32 Vitest tests across 7 files
```

## Configuration

| Env var | Default | Used by |
| --- | --- | --- |
| `VITE_API_PROXY` | `http://127.0.0.1:8000` | dev server proxy target (vite.config.ts) |
| `VITE_API_BASE` | `""` (relative) | client.ts; set to a full URL when not behind a proxy |
| `VITE_WS_BASE` | derived from origin | ws.ts |
