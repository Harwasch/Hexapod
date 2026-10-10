/* eslint-disable @typescript-eslint/require-await -- Async fixture doubles implement the network and storage contracts. */
import { webcrypto } from "node:crypto";
import { Blob as NodeBlob } from "node:buffer";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { WorldsApi } from "../core/api";
import type { worldStore } from "../core/storage";
import type { BenchmarkResult } from "../core/types";
import { frameSimilarity, measureFrame } from "./metrics";
import { runExperiment, validatePlan, type RunnerDependencies } from "./runner";
import { isExperimentResult, type ExperimentPlan, type GrayFrame } from "./types";
import { validatedAssessment } from "./characterEvaluation";
const fixturePlan = (): ExperimentPlan => ({
  project: {
    id: "world",
    name: "Fixture world",
    prompt: "A quiet test environment",
    modelId: "astronex-world",
    providerId: "local",
    createdAt: 1,
    updatedAt: 1,
    assetIds: [],
    characterIds: [],
    settings: { performance: "balanced" },
  },
  targets: [
    { modelId: "first", providerId: "local" },
    { modelId: "second", providerId: "local" },
  ],
  seed: 17,
  events: [
    { id: "move", type: "native", action: "forward", timestampMs: 500, values: { pressed: true } },
  ],
  durationSeconds: 5,
  maxWallTimeSeconds: 10,
  maxEstimatedCostUSD: 0,
  mode: "compare",
});
function rig(
  options: {
    cancelOnAllocation?: AbortController;
    failCleanup?: boolean;
    noFrames?: boolean;
    price?: number | null;
    allocationFails?: boolean;
    visionConfigured?: boolean;
    secondAssessmentFails?: boolean;
  } = {},
) {
  vi.stubGlobal("crypto", webcrypto);
  vi.stubGlobal("Blob", NodeBlob);
  let time = 0;
  let sequence = 0;
  let currentModel = "";
  let assessments = 0;
  const media = new Map<string, Blob>([
    ["character-reference", new Blob(["character-reference-fixture"], { type: "image/png" })],
  ]);
  const calls: { path: string; method?: string; body?: Record<string, unknown> }[] = [];
  const saved: BenchmarkResult[] = [];
  const request: WorldsApi["request"] = async <T>(path: string, init?: RequestInit): Promise<T> => {
    init ??= {};
    const body =
      typeof init.body === "string"
        ? (JSON.parse(init.body) as Record<string, unknown>)
        : undefined;
    calls.push({ path, method: init.method, body });
    let result: unknown = {};
    if (path === "/intelligence/status")
      result = { visionConfigured: options.visionConfigured ?? true };
    else if (path === "/characters/evaluate") {
      assessments++;
      if (options.secondAssessmentFails && assessments === 2)
        throw new Error("Vision unavailable for second sample");
      result = {
        score: 0.8,
        confidence: 0.7,
        evidence: ["Matching yellow coat"],
        differences: ["Lighting changed"],
        source: "vision-llm",
        metric: "qualitative-appearance-consistency",
        note: "Qualitative visual evidence only",
      };
    } else if (path.endsWith("/quote"))
      result = {
        hourlyCost: options.price === undefined ? 0 : options.price,
        pricingSource: "operator-estimate",
        available: true,
      };
    else if (path === "/workers" && init.method === "POST") {
      if (options.allocationFails) throw new Error("Allocation reply lost");
      currentModel = String(body?.modelId);
      options.cancelOnAllocation?.abort(new Error("Cancelled during allocation"));
      result = {
        id: `worker-${currentModel}`,
        status: "ready",
        provider: "local",
        estimatedHourlyCost: 0,
      };
    } else if (path === "/sessions" && init.method === "POST")
      result = { id: `session-${currentModel}`, status: "generating" };
    else if (init.method === "DELETE") {
      if (options.failCleanup) throw new Error("Cleanup fixture failed");
    } else if (path.endsWith("/frame"))
      result = options.noFrames ? null : new Blob([`fixture-${sequence++}`], { type: "image/png" });
    else if (path.endsWith("/actions")) result = { accepted: true };
    else if (path.endsWith("/heartbeat")) result = { status: "generating" };
    else if (path.startsWith("/sessions/"))
      result = {
        id: `session-${currentModel}`,
        status: "playing",
        generatedFPS: 12,
        totalGeneratedFrames: 60,
      };
    return result as T;
  };
  const store = {
    get: vi.fn(async (_store: string, id: string) =>
      media.has(id) ? { id, kind: "image" } : undefined,
    ),
    getBlob: vi.fn(async (id: string) => media.get(id)),
    saveAsset: vi.fn(async (blob: Blob) => {
      const id = `asset-${sequence++}`;
      media.set(id, blob);
      return {
        id,
        name: "fixture",
        kind: "image",
        createdAt: 1,
        mimeType: blob.type,
        size: blob.size,
      };
    }),
    put: vi.fn(async (store: string, record: BenchmarkResult) => {
      if (store === "benchmarks") saved.push(record);
    }),
  } as unknown as typeof worldStore;
  const dependencies: RunnerDependencies = {
    api: { request },
    store,
    now: () => time,
    sleep: async (ms, signal) => {
      signal.throwIfAborted();
      time += ms;
    },
    prepareImage: async (blob) =>
      `data:image/jpeg;base64,${Buffer.from(await blob.arrayBuffer()).toString("base64")}`,
    decode: async () => ({ width: 20, height: 20, pixels: new Float32Array(400).fill(0.3) }),
    capture: () => ({ frame: async () => undefined, finish: async () => undefined }),
  };
  return { dependencies, calls, saved };
}
afterEach(() => vi.unstubAllGlobals());
describe("Measured image diagnostics", () => {
  it("recovers a known translation rather than treating brightness as camera motion", () => {
    const width = 40,
      height = 32;
    let seed = 723;
    const source: GrayFrame = {
      width,
      height,
      pixels: Float32Array.from({ length: width * height }, () => {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        return seed / 2 ** 32;
      }),
    };
    const shifted: GrayFrame = { width, height, pixels: new Float32Array(width * height) };
    for (let y = 2; y < height; y++)
      for (let x = 3; x < width; x++)
        shifted.pixels[y * width + x] = source.pixels[(y - 2) * width + x - 3] ?? 0;
    const measurement = measureFrame(source, shifted);
    expect(measurement.motionX).toBe(3);
    expect(measurement.motionY).toBe(2);
    expect(measurement.motionConfidence).toBeGreaterThan(0.9);
    expect(frameSimilarity(source, source)).toBe(1);
  });
  it("reports no confident motion on a flat frame", () => {
    const frame = { width: 20, height: 20, pixels: new Float32Array(400).fill(0.5) };
    expect(measureFrame(frame, frame)).toMatchObject({
      motionX: 0,
      motionY: 0,
      motionConfidence: 0,
      pixelChange: 0,
    });
  });
});
describe("Sequential inference experiment lifecycle", () => {
  it("replays fixed seed/input and cleans up before allocating the next model", async () => {
    const fixture = rig();
    const results = await runExperiment(
      fixturePlan(),
      fixture.dependencies,
      new AbortController().signal,
      () => undefined,
    );
    expect(results).toHaveLength(2);
    expect(results.every((result) => result.status === "completed" && result.frameCount > 0)).toBe(
      true,
    );
    expect(results[0]?.inputFingerprint).toBe(results[1]?.inputFingerprint);
    const sessions = fixture.calls.filter((call) => call.path === "/sessions");
    expect(sessions.map((call) => call.body?.seed)).toEqual([17, 17]);
    expect(
      fixture.calls.findIndex(
        (call) => call.path === "/workers/worker-first" && call.method === "DELETE",
      ),
    ).toBeLessThan(
      fixture.calls.findIndex(
        (call) => call.path === "/workers" && call.body?.modelId === "second",
      ),
    );
    expect(results[0]?.actionObservations[0]).toMatchObject({ accepted: true, scheduledMs: 500 });
    expect(fixture.saved).toHaveLength(2);
  });
  it("cleans an allocation returned after cancellation without creating inference", async () => {
    const cancellation = new AbortController();
    const fixture = rig({ cancelOnAllocation: cancellation });
    const results = await runExperiment(
      fixturePlan(),
      fixture.dependencies,
      cancellation.signal,
      () => undefined,
    );
    expect(results[0]?.status).toBe("cancelled");
    expect(
      fixture.calls.some(
        (call) => call.path === "/workers/worker-first" && call.method === "DELETE",
      ),
    ).toBe(true);
    expect(fixture.calls.some((call) => call.path === "/sessions")).toBe(false);
  });
  it("halts the suite after uncertain allocation or failed cleanup", async () => {
    for (const options of [{ allocationFails: true }, { failCleanup: true }]) {
      const fixture = rig(options);
      const results = await runExperiment(
        fixturePlan(),
        fixture.dependencies,
        new AbortController().signal,
        () => undefined,
      );
      expect(results).toHaveLength(1);
      expect(results[0]?.cleanupErrors.length).toBeGreaterThan(0);
      expect(fixture.calls.filter((call) => call.path === "/workers")).toHaveLength(1);
    }
  });
  it("rejects remote unknown pricing before allocation", async () => {
    const fixture = rig({ price: null });
    const plan = fixturePlan();
    plan.targets = [{ modelId: "first", providerId: "runpod" }];
    const result = await runExperiment(
      plan,
      fixture.dependencies,
      new AbortController().signal,
      () => undefined,
    );
    expect(result[0]?.status).toBe("failed");
    expect(fixture.calls.some((call) => call.path === "/workers")).toBe(false);
  });
  it("ends a no-frame session at its wall-clock limit and never invents metrics", async () => {
    const fixture = rig({ noFrames: true });
    const plan = fixturePlan();
    plan.targets = plan.targets.slice(0, 1);
    const results = await runExperiment(
      plan,
      fixture.dependencies,
      new AbortController().signal,
      () => undefined,
    );
    expect(results[0]).toMatchObject({ status: "failed", frameCount: 0, deliveredFPS: 0 });
    expect(results[0]?.errors.join()).toContain("wall-clock");
    expect(
      fixture.calls.some(
        (call) => call.path === "/sessions/session-first" && call.method === "DELETE",
      ),
    ).toBe(true);
  });
  it("runs an inverse camera excursion with equally timed native controls", async () => {
    const fixture = rig();
    const plan = fixturePlan();
    plan.targets = [{ modelId: "astronex-world", providerId: "local" }];
    plan.mode = "return-to-view";
    const result = await runExperiment(
      plan,
      fixture.dependencies,
      new AbortController().signal,
      () => undefined,
    );
    const actions = fixture.calls
      .filter((call) => call.path.endsWith("/actions"))
      .map((call) => call.body);
    expect(actions.map((action) => action?.action)).toEqual([
      "forward",
      "forward",
      "backward",
      "backward",
    ]);
    expect(actions.map((action) => action?.values)).toEqual([
      { pressed: true },
      { pressed: false },
      { pressed: true },
      { pressed: false },
    ]);
    expect(result[0]?.returnAlignment).toMatchObject({ x: 0, y: 0, confidence: 0 });
    expect(result[0]?.requestedEvents).toHaveLength(4);
  });
  it("keeps image-only adapter requests free of unsupported text and quality options", async () => {
    const fixture = rig();
    const plan = fixturePlan();
    plan.targets = [{ modelId: "forge-wm", providerId: "local" }];
    plan.project.settings.performance = "quality";
    const results = await runExperiment(
      plan,
      fixture.dependencies,
      new AbortController().signal,
      () => undefined,
    );
    const session = fixture.calls.find((call) => call.path === "/sessions");
    expect(session?.body).toMatchObject({ prompt: "", quality: "balanced" });
    expect(results[0]?.effectivePrompt).toBe("");
  });
  it("evaluates only actual captured character frames after GPU cleanup and aggregates successful evidence", async () => {
    const fixture = rig({ secondAssessmentFails: true });
    const plan = fixturePlan();
    plan.targets = [{ modelId: "astronex-world", providerId: "local", characterView: "front" }];
    plan.mode = "character-consistency";
    plan.visionConsent = true;
    plan.character = {
      id: "character",
      name: "Explorer",
      description: "Yellow coat",
      assetIds: ["character-reference"],
      createdAt: 1,
      updatedAt: 1,
    };
    const results = await runExperiment(
      plan,
      fixture.dependencies,
      new AbortController().signal,
      () => undefined,
    );
    const result = results[0];
    expect(result?.characterEvaluation).toMatchObject({
      meanScore: 0.8,
      meanConfidence: 0.7,
      source: "vision-llm",
    });
    expect(result?.characterEvaluation?.samples).toHaveLength(1);
    expect(result?.characterEvaluation?.errors).toHaveLength(1);
    expect(
      fixture.calls.findIndex(
        (call) => call.path === "/workers/worker-astronex-world" && call.method === "DELETE",
      ),
    ).toBeLessThan(fixture.calls.findIndex((call) => call.path === "/characters/evaluate"));
    const evaluation = fixture.calls.find((call) => call.path === "/characters/evaluate");
    expect(evaluation?.body?.candidates).toEqual(
      expect.arrayContaining([
        expect.stringContaining(Buffer.from("fixture-").toString("base64").slice(0, 8)),
      ]),
    );
    expect(isExperimentResult(result as BenchmarkResult)).toBe(true);
    expect(isExperimentResult({ ...result, errors: undefined } as unknown as BenchmarkResult)).toBe(
      false,
    );
  });
  it("never sends character images to vision without separate consent", async () => {
    const fixture = rig();
    const plan = fixturePlan();
    plan.targets = [{ modelId: "astronex-world", providerId: "local" }];
    plan.mode = "character-consistency";
    plan.character = {
      id: "character",
      name: "Explorer",
      description: "Yellow coat",
      assetIds: ["character-reference"],
      createdAt: 1,
      updatedAt: 1,
    };
    const results = await runExperiment(
      plan,
      fixture.dependencies,
      new AbortController().signal,
      () => undefined,
    );
    expect(
      fixture.calls.some(
        (call) => call.path.startsWith("/characters/") || call.path === "/intelligence/status",
      ),
    ).toBe(false);
    expect(results[0]?.characterEvaluation?.source).toBe("not-evaluated");
    expect(results[0]?.characterEvaluation?.meanScore).toBeUndefined();
  });
  it("rejects an unavailable evaluator before GPU allocation and rejects biometric/fabricated score envelopes", async () => {
    const fixture = rig({ visionConfigured: false });
    const plan = fixturePlan();
    plan.mode = "character-consistency";
    plan.visionConsent = true;
    plan.character = {
      id: "character",
      name: "Explorer",
      description: "Yellow coat",
      assetIds: ["character-reference"],
      createdAt: 1,
      updatedAt: 1,
    };
    await expect(
      runExperiment(plan, fixture.dependencies, new AbortController().signal, () => undefined),
    ).rejects.toThrow("Configure the vision");
    expect(fixture.calls.some((call) => call.path === "/workers")).toBe(false);
    expect(() =>
      validatedAssessment({
        score: 1,
        confidence: 1,
        source: "vision-llm",
        metric: "biometric-identity",
      }),
    ).toThrow("Invalid qualitative");
  });
  it("rejects unsafe sweep size and invalid trajectory timing before requests", () => {
    const plan = fixturePlan();
    plan.targets = Array.from({ length: 9 }, () => ({
      modelId: "x",
      providerId: "local" as const,
    }));
    expect(() => validatePlan(plan)).toThrow("eight");
    plan.targets = plan.targets.slice(0, 1);
    const event = plan.events[0];
    if (event) event.timestampMs = -1;
    expect(() => validatePlan(plan)).toThrow("Trajectory");
  });
});
