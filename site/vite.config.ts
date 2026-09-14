import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/methodos/",
  build: {
    outDir: "dist",
    sourcemap: false,
    target: "es2020",
  },
});
