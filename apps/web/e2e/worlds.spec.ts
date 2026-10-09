import { expect, test } from "@playwright/test";
import type { WorldsReadiness } from "../src/worlds/core/api";

const readyCompute: WorldsReadiness = {
  provisioningEnabled: true,
  providers: [
    {
      id: "runpod",
      name: "RunPod",
      configured: true,
      canProvision: true,
      message: "Synthetic lifecycle test",
    },
  ],
  gateways: [],
  lifecycle: {
    enabled: true,
    running: true,
    sessionLeaseSeconds: 90,
    heartbeatIntervalSeconds: 20,
    workerIdleSeconds: 300,
    workerMaxLifetimeSeconds: 3600,
    workerStartupSeconds: 900,
    maxManagedWorkers: 1,
    maxWorkerHourlyCost: 1,
  },
};

/** The offline product must remain usable without model or cloud credentials. */
test.beforeEach(async ({ page }) => {
  await page.route("**/api/v1/worlds/**", (route) =>
    route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({ detail: "No GPU worker configured for this test." }),
    }),
  );
});

test("separate Worlds entry renders without Cesium and fits a phone", async ({ page }) => {
  const globeRequests: string[] = [];
  page.on("request", (request) => {
    if (/cesium/i.test(request.url())) globeRequests.push(request.url());
  });
  await page.goto("/worlds.html");
  await expect(page.getByRole("heading", { name: "Where will you go?" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Worlds home" })).toHaveAttribute(
    "href",
    "/worlds.html",
  );
  await expect(page.getByText("CONCEPT ART · NOT MODEL OUTPUT")).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole("heading", { name: "Where will you go?" })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
  expect(globeRequests).toEqual([]);
});

test("preview saves visual checkpoints and recorded video across reloads", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/worlds.html");
  await page.getByRole("button", { name: "Try interaction preview", exact: true }).click();
  await expect(page.getByText("INTERFACE PREVIEW", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Save scene", exact: true }).click();
  await expect(page.getByText(/Visual checkpoint saved locally/)).toBeVisible();
  await page.getByRole("button", { name: "Start recording", exact: true }).click();
  await expect(page.getByRole("button", { name: "Stop recording", exact: true })).toBeVisible();
  await page.waitForTimeout(1200); // a nonempty recording, not timing a network request
  await page.getByRole("button", { name: "Stop recording", exact: true }).click();
  await expect(page.getByRole("button", { name: "Start recording", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Exit world", exact: true }).click();
  await page.reload();
  await page.getByRole("button", { name: /^My worlds/ }).click();
  await expect(page.getByText("Visual checkpoint", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Replays", exact: true }).click();
  await expect(page.getByText("Preview recording", { exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

test("model and compute are separate and a missing GPU never silently generates", async ({
  page,
}) => {
  await page.goto("/worlds.html");
  await page.getByRole("button", { name: "Make it your world", exact: true }).click();
  await expect(page.getByRole("combobox", { name: "WORLD MODEL", exact: true })).toHaveValue(
    "astronex-world",
  );
  await expect(page.getByRole("combobox", { name: "COMPUTE PROVIDER", exact: true })).toHaveValue(
    "runpod",
  );
  await expect(page.getByRole("button", { name: "Start GPU world", exact: true })).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "Try interaction preview", exact: true }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "Enhance prompt", exact: false }).click();
  await expect(page.getByText("LOCAL WRITING GUIDE · NOT AI", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Keep original", exact: true }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
});

test("live startup provisions once, respects capability prompts and tears down its worker", async ({
  page,
}) => {
  let creations = 0;
  let checks = 0;
  const actions: unknown[] = [];
  const deleted: string[] = [];
  const caps = {
    input: { text: true, image: true, multiImage: false, video: false, audio: false },
    control: {
      wasd: true,
      mouseLook: false,
      camera6DoF: false,
      gamepad: false,
      discreteActions: true,
      continuousActions: false,
      semanticActions: false,
      promptDuringRollout: false,
      promptSwitching: true,
      timedEvents: false,
      characterReference: false,
    },
    output: { video: true, audio: false, depth: false, cameraPose: false },
    runtime: { resolutionOptions: ["832x480"], realtime: false },
    persistence: { nativeMemory: false, snapshotRestore: false, deterministicSeed: true },
    customization: { lora: false, fineTune: false, adapters: false },
    nativeActions: ["forward", "backward", "left", "right", "stop"],
  };
  await page.route("**/api/v1/worlds/**", async (route) => {
    const path = new URL(route.request().url()).pathname.replace("/api/v1/worlds", "");
    const method = route.request().method();
    const json = (body: unknown, status = 200) =>
      route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
    if (path === "/readiness") return json(readyCompute);
    if (path.endsWith("/heartbeat")) return json({ leaseExpiresAt: Date.now() / 1000 + 90 });
    if (path === "/providers")
      return json({
        providers: [
          {
            id: "runpod",
            name: "RunPod",
            configured: true,
            canProvision: true,
            message: "Test gateway",
          },
        ],
      });
    if (path === "/workers" && method === "POST") {
      creations++;
      return json({
        id: "test-worker",
        provider: "runpod",
        managed: true,
        status: "starting",
        estimatedHourlyCost: 0.6,
      });
    }
    if (method === "DELETE") {
      deleted.push(path);
      return json({ status: "stopped" });
    }
    if (path === "/workers/test-worker") {
      checks++;
      return json({ id: "test-worker", status: "ready", managed: true });
    }
    if (path === "/sessions" && method === "POST")
      return json({
        id: "test-session",
        workerId: "test-worker",
        modelId: "astronex-world",
        status: "generating",
        capabilities: caps,
        seed: 42,
      });
    if (path === "/sessions/test-session")
      return json({ id: "test-session", status: "generating", capabilities: caps });
    if (path.endsWith("/offer")) return json({ detail: "Frame polling only" }, 501);
    if (path.endsWith("/frame")) return route.fulfill({ status: 204 });
    if (path.endsWith("/actions")) {
      actions.push(route.request().postDataJSON());
      return json({ accepted: true });
    }
    if (path === "/sessions") return json({ sessions: [] });
    return json({ detail: "Unexpected mocked route" }, 404);
  });
  await page.goto("/worlds.html");
  await page.getByRole("button", { name: "Make it your world", exact: true }).click();
  await page.getByRole("checkbox", { name: /Send required inputs/ }).check();
  await page.getByRole("button", { name: "Start paid GPU world", exact: true }).click();
  const command = page.getByRole("textbox", { name: "World command", exact: true });
  await expect(command).toBeEnabled();
  await command.fill("Make the clouds red");
  await command.press("Enter");
  await expect.poll(() => actions.length).toBeGreaterThan(0);
  expect(actions).toContainEqual(
    expect.objectContaining({ type: "prompt", prompt: "Make the clouds red" }),
  );
  await page.getByRole("button", { name: "Exit world", exact: true }).click();
  await expect.poll(() => deleted).toContain("/workers/test-worker");
  expect(deleted).toContain("/sessions/test-session");
  expect(creations).toBe(1);
  expect(checks).toBeGreaterThan(0);
});

test("worker recovery ends interrupted sessions and explicitly destroys retained compute", async ({
  page,
}) => {
  const deleted: string[] = [];
  let stopped = false;
  let destroyed = false;
  await page.route("**/api/v1/worlds/**", (route) => {
    const path = new URL(route.request().url()).pathname.replace("/api/v1/worlds", "");
    const json = (body: unknown) =>
      route.fulfill({ contentType: "application/json", body: JSON.stringify(body) });
    if (path === "/readiness") return json(readyCompute);
    if (route.request().method() === "DELETE") {
      deleted.push(path);
      if (path.startsWith("/sessions")) stopped = true;
      if (path.startsWith("/workers")) destroyed = true;
      return json({ status: "stopped" });
    }
    if (path === "/workers")
      return json({
        workers: [
          {
            id: "retained-worker",
            provider: "runpod",
            status: destroyed ? "destroyed" : "ready",
            managed: true,
            estimatedHourlyCost: null,
          },
        ],
      });
    if (path === "/sessions")
      return json({
        sessions: [
          {
            id: "interrupted-session",
            workerId: "retained-worker",
            modelId: "astronex-world",
            status: stopped ? "stopped" : "error",
          },
        ],
      });
    return json({ providers: [] });
  });
  await page.goto("/worlds.html");
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await page.getByRole("button", { name: "Refresh workers", exact: true }).click();
  await expect(page.getByText("astronex-world · error")).toBeVisible();
  await page.getByRole("button", { name: "End session", exact: true }).click();
  await expect.poll(() => deleted).toContain("/sessions/interrupted-session");
  await page.getByRole("button", { name: "Destroy worker", exact: true }).click();
  expect(deleted).not.toContain("/workers/retained-worker");
  await page.getByRole("button", { name: "Confirm destroy", exact: true }).click();
  await expect(page.getByText("Provider confirmed worker destruction.")).toBeVisible();
  expect(deleted).toContain("/workers/retained-worker");
});

test("losing a retained-worker claim never destroys another tab's session", async ({ page }) => {
  let creations = 0;
  const deleted: string[] = [];
  await page.addInitScript(() =>
    localStorage.setItem("hexapod.worlds.settings.v1", JSON.stringify({ retainWorker: true })),
  );
  await page.route("**/api/v1/worlds/**", (route) => {
    const path = new URL(route.request().url()).pathname.replace("/api/v1/worlds", "");
    const method = route.request().method();
    const json = (body: unknown, status = 200) =>
      route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
    if (path === "/readiness") return json(readyCompute);
    if (method === "DELETE") {
      deleted.push(path);
      return json({});
    }
    if (path === "/providers")
      return json({
        providers: [
          { id: "runpod", name: "RunPod", configured: true, canProvision: true, message: "Test" },
        ],
      });
    if (path === "/workers" && method === "POST") {
      creations++;
      return json({ id: "unexpected", status: "ready" });
    }
    if (path === "/workers")
      return json({
        workers: [{ id: "shared-worker", provider: "runpod", status: "ready", managed: true }],
      });
    if (path === "/workers/shared-worker")
      return json({ id: "shared-worker", status: "ready", managed: true });
    if (path === "/sessions" && method === "GET") return json({ sessions: [] });
    if (path === "/sessions")
      return json({ detail: "Another tab already claimed this worker." }, 409);
    return json({});
  });
  await page.goto("/worlds.html");
  await page.getByRole("button", { name: "Make it your world", exact: true }).click();
  await page.getByRole("checkbox", { name: /Send required inputs/ }).check();
  await page.getByRole("button", { name: "Start paid GPU world", exact: true }).click();
  await expect(
    page.getByText(
      /Another tab already claimed this worker\. The retained worker was left unchanged/,
    ),
  ).toBeVisible();
  await page.getByRole("button", { name: "Exit world", exact: true }).click();
  expect(creations).toBe(0);
  expect(deleted).toEqual([]);
});
