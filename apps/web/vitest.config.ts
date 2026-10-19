import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  define: { CESIUM_BASE_URL: JSON.stringify("/cesium/") },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./vitest.setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    css: false,
    // Pin opt-in flags to their shipped defaults so a developer's .env (the land docs ask
    // for VITE_ENABLE_LAND_EXPLORATION=true) cannot change what the suite asserts.
    env: { VITE_ENABLE_LAND_EXPLORATION: "false" },
    server: { deps: { inline: ["cesium"] } },
  },
});
