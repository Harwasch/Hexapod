import { expect, test, type Page } from "@playwright/test";
import type * as Storage from "../src/worlds/core/storage";
async function fixture(page: Page) {
  const deleted: string[] = [];
  const operations: string[] = [];
  let frames = 0;
  let frameData: string[] = [];
  await page.route("**/api/v1/worlds/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname.replace("/api/v1/worlds", "");
    const method = route.request().method();
    operations.push(`${method} ${path}`);
    let body: unknown;
    if (method === "DELETE") {
      deleted.push(path);
      await route.fulfill({ status: 204 });
      return;
    }
    if (path === "/intelligence/status")
      body = {
        visionConfigured: true,
        imageConfigured: false,
        configured: true,
        message: "Fixture evaluator",
      };
    else if (path === "/characters/evaluate")
      body = {
        score: 0.72,
        confidence: 0.6,
        evidence: ["Matching fixture colors"],
        differences: ["Changed background"],
        source: "vision-llm",
        metric: "qualitative-appearance-consistency",
        note: "Qualitative fixture evidence, not identity verification",
      };
    else if (path.endsWith("/quote"))
      body = {
        hourlyCost: 0,
        estimatedCostUSD: 0,
        pricingSource: "operator-estimate",
        available: true,
      };
    else if (path === "/workers" && method === "POST")
      body = {
        id: "experiment-worker",
        provider: "local",
        status: "ready",
        managed: false,
        estimatedHourlyCost: 0,
      };
    else if (path === "/sessions" && method === "POST")
      body = { id: "experiment-session", status: "generating" };
    else if (path.endsWith("/frame")) {
      const base64 = frameData[frames++ % 2];
      if (!base64) {
        await route.fulfill({ status: 204 });
        return;
      }
      await route.fulfill({
        status: 200,
        contentType: "image/png",
        body: Buffer.from(base64, "base64"),
      });
      return;
    } else if (path.endsWith("/heartbeat")) body = { status: "playing" };
    else if (path.startsWith("/sessions/"))
      body = {
        id: "experiment-session",
        status: "playing",
        generatedFPS: 9,
        totalGeneratedFrames: 45,
      };
    else {
      await route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ detail: "No external services are used by this fixture." }),
      });
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(body),
    });
  });
  await page.goto("/worlds.html");
  frameData = await page.evaluate(async () => {
    const path = "/src/worlds/core/storage.ts";
    const { worldStore } = (await import(path)) as typeof Storage;
    await worldStore.put("projects", {
      id: "benchmark-project",
      name: "Local benchmark fixture",
      prompt: "A recorded fixture for browser testing",
      modelId: "astronex-world",
      providerId: "local",
      createdAt: 1,
      updatedAt: 1,
      assetIds: [],
      characterIds: [],
      settings: { performance: "balanced" },
    });
    const canvas = document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 40;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Canvas missing");
    const frames = ["#32657a", "#567b43"].map((color) => {
      context.fillStyle = color;
      context.fillRect(0, 0, 64, 40);
      return canvas.toDataURL("image/png").split(",")[1] ?? "";
    });
    const reference = await worldStore.saveAsset(
      new Blob([Uint8Array.from(atob(frames[0] ?? ""), (character) => character.charCodeAt(0))], {
        type: "image/png",
      }),
      "Fixture reference",
      "image",
    );
    await worldStore.put("characters", {
      id: "fixture-character",
      name: "Test explorer",
      description: "A creative character in a yellow coat",
      assetIds: [reference.id],
      createdAt: 1,
      updatedAt: 1,
    });
    return frames;
  });
  await page.reload();
  await page.getByRole("button", { name: "Model lab", exact: true }).click();
  await page.getByRole("button", { name: "Experiments", exact: true }).click();
  const lab = page.getByRole("region", { name: "Executable model experiments" });
  await lab.getByLabel("Source world").selectOption("benchmark-project");
  await lab.getByLabel("Capture seconds", { exact: true }).fill("5");
  await lab.getByLabel("Max seconds per run").fill("15");
  await lab.getByRole("checkbox", { name: /I authorize these inference sessions/ }).check();
  return { lab, deleted, operations };
}
test("local experiment records real browser capture and completes bounded cleanup", async ({
  page,
}) => {
  const { lab, deleted } = await fixture(page);
  await lab.getByRole("button", { name: "Run comparison" }).click();
  await expect(lab.getByText("completed · seed 42")).toBeVisible({ timeout: 25000 });
  expect(deleted).toEqual(["/sessions/experiment-session", "/workers/experiment-worker"]);
  const library = await page.evaluate(async () => {
    const path = "/src/worlds/core/storage.ts";
    const { worldStore } = (await import(path)) as typeof Storage;
    const results = await worldStore.list("benchmarks");
    const assets = await worldStore.list("assets");
    const videos = assets.filter((asset) => asset.kind === "video");
    return {
      results: results.map((result) => ({ frames: result.frameCount, fps: result.measuredFPS })),
      videos: videos.length,
      bytes: videos[0] ? (await worldStore.getBlob(videos[0].id))?.size : 0,
    };
  });
  expect(library.results[0]?.frames).toBeGreaterThan(1);
  expect(library.results[0]?.fps).toBe(9);
  expect(library.videos).toBe(1);
  expect(library.bytes).toBeGreaterThan(0);
  await expect(lab.getByRole("button", { name: "Play results together" })).toBeVisible();
});
test("leaving an active experiment cancels inference and removes its worker handle", async ({
  page,
}) => {
  const { lab, deleted } = await fixture(page);
  await lab.getByLabel("Capture seconds", { exact: true }).fill("30");
  await lab.getByLabel("Max seconds per run").fill("40");
  await lab.getByRole("button", { name: "Run comparison" }).click();
  await expect(lab.getByText("Replaying controls and measuring")).toBeVisible();
  await page.getByRole("button", { name: "Discover", exact: true }).click();
  await expect
    .poll(() => deleted)
    .toEqual(["/sessions/experiment-session", "/workers/experiment-worker"]);
});

test("character experiment evaluates captured samples only after consent and compute cleanup", async ({
  page,
}) => {
  const { lab, operations } = await fixture(page);
  await lab.getByLabel("Experiment mode", { exact: true }).selectOption("character-consistency");
  await lab.getByLabel("Experiment character").selectOption("fixture-character");
  await lab.getByRole("checkbox", { name: "side view", exact: true }).uncheck();
  await lab.getByRole("checkbox", { name: "New environment", exact: true }).uncheck();
  await lab.getByRole("checkbox", { name: /Also send up to four character references/ }).check();
  await lab.getByRole("button", { name: "Run character experiments" }).click();
  await expect(lab.getByText(/Observed appearance score 72.0%/)).toBeVisible({ timeout: 25000 });
  expect(operations.indexOf("DELETE /workers/experiment-worker")).toBeLessThan(
    operations.indexOf("POST /characters/evaluate"),
  );
  expect(operations.filter((value) => value === "POST /characters/evaluate")).toHaveLength(2);
  const persisted = await page.evaluate(async () => {
    const path = "/src/worlds/core/storage.ts";
    const { worldStore } = (await import(path)) as typeof Storage;
    return (await worldStore.list("benchmarks"))[0];
  });
  expect(persisted).toMatchObject({
    characterEvaluation: {
      characterId: "fixture-character",
      source: "vision-llm",
      metric: "qualitative-appearance-consistency",
      meanScore: 0.72,
    },
  });
});
