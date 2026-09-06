import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    open: true,
  },
  build: {
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      output: {
        manualChunks: {
          "react-vendor":   ["react", "react-dom", "react-router-dom"],
          "charts-vendor":  ["recharts"],
          "ui-vendor":      ["lucide-react", "react-hot-toast", "react-dropzone"],
          "http-vendor":    ["axios"],
        },
      },
    },
  },
});
