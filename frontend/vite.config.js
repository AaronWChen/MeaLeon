// frontend/vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // In dev (npm run dev), proxy API calls to Flask directly.
      // In Docker, the nginx proxy container handles this instead —
      // this config only affects `npm run dev` on your host.
      "/api": "http://localhost:5000",
      "/auth": "http://localhost:5000",
    },
  },
  build: {
    outDir: "dist",
  },
});
