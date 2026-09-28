import path from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// The built SPA is committed under src/web/static/app so `recon.py serve`
// needs no Node toolchain. FastAPI serves index.html for every page route
// and mounts the hashed assets under /static/app/.
const API_PREFIXES = [
  "/api", "/anchors", "/ball-anchors", "/ball-quality", "/joints-near",
  "/goal-element-suggest", "/pitch-fix-suggest", "/landmarks", "/pitch_lines",
  "/stadiums", "/camera", "/tracking", "/hmr_world", "/refined_poses", "/ball/preview",
]

export default defineConfig(({ command }) => ({
  // Dev serves from "/" so page routes (/anchor_editor, /viewer…) resolve
  // exactly as they do behind FastAPI; the build is mounted at /static/app/.
  base: command === "serve" ? "/" : "/static/app/",
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  build: {
    outDir: path.resolve(__dirname, "../src/web/static/app"),
    emptyOutDir: true,
    chunkSizeWarningLimit: 1200,
  },
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      API_PREFIXES.map((p) => [
        p,
        { target: process.env.FP_API ?? "http://localhost:8765", changeOrigin: true },
      ]),
    ),
  },
}))
