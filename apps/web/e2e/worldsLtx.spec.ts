import { expect, test } from "@playwright/test";
import capabilities from "../src/worlds/core/runtimeCapabilities.json" with { type: "json" };

test("LTX exploration queues text, movement settings, and speech until output applies", async ({
  page,
}) => {
  const actions: Record<string, unknown>[] = [];
  let queued = 0;
  let applied = 0;
  const caps = (capabilities as Record<string, unknown>)["ltx-2.5"];
  expect(caps).toBeTruthy();
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
          { modelId: "ltx-2.5", providers: ["runpod"], gatewayProviders: ["runpod"] },
        ],
        gateways: [
          {
            provider: "runpod",
            modelId: "ltx-2.5",
            status: "ready",
            models: [{ id: "ltx-2.5", status: "ready" }],
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
        id: "ltx-worker",
        modelId: "ltx-2.5",
        provider: "runpod",
        managed: false,
        status: "ready",
      });
    if (path === "/workers/ltx-worker")
      return json({ id: "ltx-worker", modelId: "ltx-2.5", status: "ready", managed: false });
    if (path === "/sessions" && method === "POST")
      return json({
        id: "ltx-session",
        workerId: "ltx-worker",
        modelId: "ltx-2.5",
        status: "generating",
        capabilities: caps,
        seed: 42,
      });
    if (path === "/sessions/ltx-session")
      return json({
        id: "ltx-session",
        status: "generating",
        capabilities: caps,
        queuedRevision: queued,
        appliedRevision: applied,
        generatingRevision: queued,
      });
    if (path.endsWith("/heartbeat")) return json({ leaseExpiresAt: Date.now() / 1000 + 90 });
    if (path.endsWith("/offer")) return json({ detail: "Fixture has no video transport" }, 501);
    if (path.endsWith("/frame")) return route.fulfill({ status: 204 });
    if (path.endsWith("/actions")) {
      const action = route.request().postDataJSON() as Record<string, unknown>;
      actions.push(action);
      return json({ accepted: true, revision: ++queued, appliesAt: "next-chunk" });
    }
    if (path === "/sessions") return json({ sessions: [] });
    return json({ detail: "Fixture has no optional services" }, 503);
  });
  await page.addInitScript(() => {
    class Speech {
      continuous = false;
      interimResults = false;
      lang = "en-US";
      onresult: ((event: unknown) => void) | null = null;
      onerror = null;
      onend: (() => void) | null = null;
      start() {
        (window as unknown as { testSpeech: Speech }).testSpeech = this;
      }
      abort() {
        this.onend?.();
      }
    }
    (window as unknown as { SpeechRecognition: typeof Speech }).SpeechRecognition = Speech;
  });
  await page.goto("/worlds.html");
  await page.getByRole("button", { name: "Make it your world", exact: true }).click();
  await page.getByRole("combobox", { name: "WORLD MODEL", exact: true }).selectOption("ltx-2.5");
  await page.getByRole("checkbox", { name: /Send required inputs/ }).check();
  await page.getByRole("button", { name: "Start GPU world", exact: true }).click();
  await expect(page.getByRole("combobox", { name: "Exploration mode" })).toBeEnabled();
  await page.getByRole("combobox", { name: "Exploration mode" }).selectOption("cruise");
  await page.getByRole("combobox", { name: "Generation cadence" }).selectOption("responsive");
  await page.getByRole("button", { name: "Apply exploration settings" }).click();
  await expect.poll(() => actions.some((action) => action.action === "exploration")).toBe(true);
  expect(actions.find((action) => action.action === "exploration")).toEqual({
    type: "native",
    action: "exploration",
    values: { mode: "cruise", cadence: "responsive", speed: 0.5 },
  });
  const command = page.getByRole("textbox", { name: "World command", exact: true });
  await command.fill("Make it rain around the castle");
  await command.press("Enter");
  await expect
    .poll(() => actions.some((action) => action.prompt === "Make it rain around the castle"))
    .toBe(true);
  await expect(page.locator(".wp-command-footer")).toContainText(
    "received · queued for the next chunk",
  );
  expect(await page.locator(".wp-command-footer").innerText()).not.toContain(
    "applied to generated output",
  );
  applied = queued;
  await expect(page.locator(".wp-command-footer")).toContainText("applied to generated output");
  await page.getByRole("button", { name: "Use voice input", exact: true }).click();
  await page.getByRole("checkbox", { name: "Keep listening between commands" }).check();
  await page.getByRole("button", { name: "Enable voice", exact: true }).click();
  await page.getByRole("button", { name: "Use voice input", exact: true }).click();
  await page.evaluate(() => {
    (
      window as unknown as { testSpeech: { onresult: (event: unknown) => void } }
    ).testSpeech.onresult({
      resultIndex: 0,
      results: [{ 0: { transcript: "Light the castle windows" }, isFinal: true }],
    });
  });
  await expect
    .poll(() => actions.some((action) => action.prompt === "Light the castle windows"))
    .toBe(true);
  await page.getByRole("button", { name: "Pause generation", exact: true }).click();
  await expect(page.getByRole("combobox", { name: "Exploration mode" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Use voice input", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "Exit world", exact: true }).click();
});
