import { fileURLToPath } from "node:url";
import { defineConfig } from "@playwright/test";

const port = process.env.E2E_PORT ?? "5287";
export default defineConfig({
  testDir: ".",
  testMatch: "browser.spec.mjs",
  workers: 1,
  timeout: 60_000,
  outputDir: "../../../../test-results/globe-embed",
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    viewport: { width: 390, height: 844 },
    deviceScaleFactor: 3,
    screenshot: "on",
    launchOptions: {
      args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    },
  },
  webServer: {
    cwd: fileURLToPath(new URL("../..", import.meta.url)),
    command: `pnpm exec vite --host 127.0.0.1 --port ${port} --strictPort`,
    url: `http://127.0.0.1:${port}/globe-embed.html`,
    reuseExistingServer: false,
    env: { VITE_CESIUM_ION_ACCESS_TOKEN: "" },
  },
});
