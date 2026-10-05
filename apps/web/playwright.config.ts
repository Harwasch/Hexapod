import { defineConfig, devices } from "@playwright/test";

const chromiumPath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE;
/** Overridable so two checkouts can run their suites side by side. */
const port = process.env.E2E_PORT ?? "5173";

/** Software WebGL2: ANGLE on SwiftShader, as every test has always run. */
const SOFTWARE_GL = [
  "--use-gl=angle",
  "--use-angle=swiftshader",
  "--enable-unsafe-swiftshader",
  "--ignore-gpu-blocklist",
];

/**
 * Software WebGPU beside it, for the PlayCanvas WebGPU trial (docs/WEBGPU_TRIAL.md): Dawn on
 * SwiftShader's Vulkan, which Chromium ships (libvk_swiftshader). Measured on Chromium 141,
 * headless: without `--use-vulkan=swiftshader` the adapter request fails ("A valid external
 * Instance reference no longer exists"); with it the adapter is SwiftShader's (a fallback
 * adapter, 8192 px textures), and WebGL2 still runs on ANGLE as above.
 */
const SOFTWARE_WEBGPU = [
  "--enable-unsafe-webgpu",
  "--enable-features=Vulkan",
  "--use-vulkan=swiftshader",
  "--use-webgpu-adapter=swiftshader",
];

/**
 * The 3D scene specs: each drives a renderer through a dev harness page (splats, motion, LOD,
 * streaming, picking, the WebGPU trial), and together they are about two thirds of the
 * suite's time on software GL. They are the `scene` project (and `webgpu`, which is all
 * scene tests); every other spec is `chromium`. Locally both run as before. CI runs `scene`
 * and `webgpu` on main, on a dispatch, and on a pull request that touches what they render
 * from (.github/workflows/ci.yml, the `changes` job); a pull request that changes only the
 * app's pages runs `chromium`. A new spec lands in `chromium` -- run on every web change --
 * until it is listed here.
 */
const SCENE_SPECS =
  /(^|[\\/])(instances|instancesPublished|instancesScan|livingCompare|livingSurvey|livingSurveyPerf|livingSurveyScene|livingSurveyTiles|livingSurveyYard|motionRenderers|navigationPerf|poke|realSize|scanFarView|scanOverlayIdle|scanRenderers|sceneSelect|skin|skinsScan|splatLod|splatNavigation|splatStreaming|telemetry|undoYard|variants|viewCones|wind)\.spec\.ts$/;

/**
 * End-to-end tests run against the Vite dev server with the catalog API
 * mocked at the network layer, so they are deterministic and need no
 * database. Cesium runs in headless Chromium with software WebGL.
 *
 * Tests tagged `@webgpu` run only in the `webgpu` project, whose Chromium also has software
 * WebGPU; every other test runs only in `chromium`, as before.
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  // CI runs the suite in shards (every fourth test, as a `--test-list`: .github/workflows/
  // ci.yml), one runner each at `workers: 1`; each shard writes a blob report (`E2E_BLOB`)
  // that the e2e-report job merges into the one HTML report a single run would have written.
  reporter: process.env.CI
    ? process.env.E2E_BLOB
      ? [["list"], ["blob"]]
      : [["list"], ["html", { open: "never" }]]
    : "list",
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    ignoreHTTPSErrors: true,
    viewport: { width: 1440, height: 900 },
    launchOptions: {
      ...(chromiumPath ? { executablePath: chromiumPath } : {}),
      args: SOFTWARE_GL,
    },
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
      grepInvert: /@webgpu/,
      testIgnore: SCENE_SPECS,
    },
    {
      name: "scene",
      use: { ...devices["Desktop Chrome"] },
      grepInvert: /@webgpu/,
      testMatch: SCENE_SPECS,
    },
    {
      name: "webgpu",
      grep: /@webgpu/,
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: {
          ...(chromiumPath ? { executablePath: chromiumPath } : {}),
          args: [...SOFTWARE_GL, ...SOFTWARE_WEBGPU],
        },
      },
    },
  ],
  webServer: {
    command: `pnpm exec vite --host 127.0.0.1 --port ${port} --strictPort`,
    url: `http://127.0.0.1:${port}`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
