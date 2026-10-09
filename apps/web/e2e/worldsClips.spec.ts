import { expect, test } from "@playwright/test";
import capabilities from "../src/worlds/core/runtimeCapabilities.json" with { type: "json" };

test("offline model resumes with the edited prompt and one planned camera action", async ({
  page,
}) => {
  const actions: Record<string, unknown>[] = [];
  let status = "paused";
  const caps = capabilities["helix-world"];
  await page.route("**/api/v1/worlds/**", async (route) => {
    const path = new URL(route.request().url()).pathname.replace("/api/v1/worlds", "");
    const method = route.request().method();
    const json = (body: unknown, code = 200) =>
      route.fulfill({ status: code, contentType: "application/json", body: JSON.stringify(body) });
    if (path === "/readiness")
      return json({
        provisioningEnabled: false,
        providers: [
          {
            id: "runpod",
            name: "RunPod",
            configured: true,
            canProvision: false,
            message: "Fixture",
          },
        ],
        modelProfiles: [
          { modelId: "helix-world", providers: ["runpod"], gatewayProviders: ["runpod"] },
        ],
        gateways: [
          {
            provider: "runpod",
            modelId: "helix-world",
            status: "ready",
            models: [{ id: "helix-world", status: "ready" }],
          },
        ],
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
      });
    if (method === "DELETE") return json({ status: "stopped" });
    if (path === "/workers" && method === "POST")
      return json({
        id: "clip-worker",
        modelId: "helix-world",
        provider: "runpod",
        managed: false,
        status: "ready",
      });
    if (path === "/workers/clip-worker")
      return json({ id: "clip-worker", modelId: "helix-world", status: "ready", managed: false });
    if (path === "/sessions" && method === "POST")
      return json({
        id: "clip-session",
        workerId: "clip-worker",
        modelId: "helix-world",
        status: "generating",
        capabilities: caps,
        seed: 42,
      });
    if (path === "/sessions/clip-session")
      return json({ id: "clip-session", status, capabilities: caps });
    if (path.endsWith("/heartbeat")) return json({ leaseExpiresAt: Date.now() / 1000 + 90 });
    if (path.endsWith("/offer")) return json({ detail: "Fixture has no video transport" }, 501);
    if (path.endsWith("/frame")) return route.fulfill({ status: 204 });
    if (path.endsWith("/actions")) {
      const action = route.request().postDataJSON() as Record<string, unknown>;
      actions.push(action);
      if (action.type === "resume") status = "generating";
      return json({ accepted: true });
    }
    if (path === "/sessions") return json({ sessions: [] });
    return json({ detail: "Fixture has no optional services" }, 503);
  });
  await page.goto("/worlds.html");
  await page.getByRole("button", { name: "Make it your world", exact: true }).click();
  await page
    .getByRole("combobox", { name: "WORLD MODEL", exact: true })
    .selectOption("helix-world");
  await page.locator(".w-composer-upload input[type=file]").setInputFiles({
    name: "fixture.png",
    mimeType: "image/png",
    buffer: Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aA3sAAAAASUVORK5CYII=",
      "base64",
    ),
  });
  await page.getByRole("checkbox", { name: /Send required inputs/ }).check();
  await page.getByRole("button", { name: "Start GPU world", exact: true }).click();
  await expect(page.getByRole("combobox", { name: "Next clip camera" })).toBeVisible();
  await page.getByRole("combobox", { name: "Next clip camera" }).selectOption("look_left");
  await page
    .getByRole("textbox", { name: "World command", exact: true })
    .fill("Continue toward the waterfall");
  await page.getByRole("button", { name: "Resume generation", exact: true }).click();
  await expect.poll(() => actions.some((action) => action.type === "resume")).toBe(true);
  const resume = actions.findIndex((action) => action.type === "resume");
  expect(actions.slice(resume - 2, resume + 1)).toEqual([
    expect.objectContaining({ type: "prompt", prompt: "Continue toward the waterfall" }),
    expect.objectContaining({ type: "native", action: "look_left", values: { planned: true } }),
    expect.objectContaining({ type: "resume" }),
  ]);
  await page.getByRole("button", { name: "Exit world", exact: true }).click();
});
