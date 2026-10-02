import { defineConfig, devices } from "@playwright/test";

const chromiumPath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE;
/** Overridable so two checkouts can run their suites side by side. */
const port = process.env.E2E_PORT ?? "5173";

/**
 * End-to-end tests run against the Vite dev server with the catalog API
 * mocked at the network layer, so they are deterministic and need no
 * database. Cesium runs in headless Chromium with software WebGL.
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    ignoreHTTPSErrors: true,
    viewport: { width: 1440, height: 900 },
    launchOptions: {
      ...(chromiumPath ? { executablePath: chromiumPath } : {}),
      args: [
        "--use-gl=angle",
        "--use-angle=swiftshader",
        "--enable-unsafe-swiftshader",
        "--ignore-gpu-blocklist",
      ],
    },
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `pnpm exec vite --host 127.0.0.1 --port ${port} --strictPort`,
    url: `http://127.0.0.1:${port}`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
