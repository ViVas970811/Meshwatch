import path from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const API_TARGET = process.env.VITE_API_PROXY ?? "http://127.0.0.1:8000";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      // REST endpoints exposed by the FastAPI app (Phase 4 + Phase 5).
      "/api": {
        target: API_TARGET,
        changeOrigin: true,
      },
      // WebSocket fan-out from the Kafka consumer.
      "/ws": {
        target: API_TARGET.replace(/^http/, "ws"),
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
