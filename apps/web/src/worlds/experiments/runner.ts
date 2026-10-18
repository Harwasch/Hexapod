import { imageData } from "../intelligence/client";
import {
  evaluationSamples,
  updateCharacterAggregate,
  validatedAssessment,
} from "./characterEvaluation";
import { getModel } from "../core/catalog";
import type { WorldsApi, WorkerHandle } from "../core/api";
import { worldStore } from "../core/storage";
import { newId, type ControlEvent } from "../core/types";
import { frameSimilarity, grayscale, measureFrame } from "./metrics";
import { ExperimentCapture } from "./capture";
import type { ExperimentPlan, ExperimentProgress, ExperimentResult, GrayFrame } from "./types";
interface LiveSession {
  id: string;
  status: string;
  error?: string;
  generatedFPS?: number;
  totalGeneratedFrames?: number;
  frameIndex?: number;
  capabilities?: { nativeActions?: string[] };
}
interface Quote {
  hourlyCost: number | null;
  estimatedCostUSD: number | null;
  pricingSource: string;
  available?: boolean | null;
}
export interface RunnerDependencies {
  api: Pick<WorldsApi, "request">;
  store?: typeof worldStore;
  decode?: (blob: Blob) => Promise<GrayFrame>;
  prepareImage?: (blob: Blob) => Promise<string>;
  capture?: () => { frame(blob: Blob): Promise<void>; finish(): Promise<Blob | undefined> };
  now?: () => number;
  sleep?: (ms: number, signal: AbortSignal) => Promise<void>;
}
function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    signal.throwIfAborted();
    const done = () => {
      signal.removeEventListener("abort", abort);
      resolve();
    };
    const timer = setTimeout(done, ms);
    const abort = () => {
      clearTimeout(timer);
      signal.removeEventListener("abort", abort);
      reject(
        signal.reason instanceof Error
          ? signal.reason
          : new DOMException("Experiment cancelled.", "AbortError"),
      );
    };
    signal.addEventListener("abort", abort, { once: true });
  });
}
async function digest(blob: Blob): Promise<string> {
  const hash = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
  return Array.from(new Uint8Array(hash), (value) => value.toString(16).padStart(2, "0")).join("");
}
async function dataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () =>
      typeof reader.result === "string"
        ? resolve(reader.result)
        : reject(new Error("Invalid reference media."));
    reader.onerror = () => reject(new Error("Could not read the local reference."));
    reader.readAsDataURL(blob);
  });
}
function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
export function validatePlan(plan: ExperimentPlan): void {
  if (!plan.targets.length || plan.targets.length > 8)
    throw new Error("Select one to eight benchmark runs.");
  if (!Number.isInteger(plan.seed) || plan.seed < 0 || plan.seed > 2 ** 32 - 1)
    throw new Error("Seed must be an unsigned 32-bit integer.");
  if (
    !Number.isFinite(plan.durationSeconds) ||
    plan.durationSeconds < 5 ||
    plan.durationSeconds > 300
  )
    throw new Error("Capture duration must be 5–300 seconds.");
  if (
    !Number.isFinite(plan.maxWallTimeSeconds) ||
    plan.maxWallTimeSeconds < plan.durationSeconds ||
    plan.maxWallTimeSeconds > 1800
  )
    throw new Error("Per-run time limit must cover capture and be at most 30 minutes.");
  if (
    !Number.isFinite(plan.maxEstimatedCostUSD) ||
    plan.maxEstimatedCostUSD < 0 ||
    plan.maxEstimatedCostUSD > 100
  )
    throw new Error("Set an estimated suite cost limit between $0 and $100.");
  if (
    plan.events.length > 1000 ||
    plan.events.some(
      (event) =>
        !Number.isFinite(event.timestampMs) ||
        event.timestampMs < 0 ||
        event.timestampMs > plan.durationSeconds * 1000,
    )
  )
    throw new Error("Trajectory must contain at most 1000 events within the capture duration.");
  if (plan.mode === "character-consistency" && !plan.character)
    throw new Error("Choose a saved character with reference images.");
  if (
    plan.evaluationFrames !== undefined &&
    (!Number.isInteger(plan.evaluationFrames) ||
      plan.evaluationFrames < 1 ||
      plan.evaluationFrames > 3)
  )
    throw new Error("Evaluate one to three actual frames per run.");
  if (plan.project.previewOnly)
    throw new Error("Benchmarks require a real model project, not an interface preview.");
}
/** A bounded native camera excursion and inverse; visual agreement is not a 3D proof. */
export function returnTrajectory(nativeActions: string[], seconds: number): ControlEvent[] {
  const pair = [
    ["forward", "backward"],
    ["left", "right"],
  ].find((actions) => actions.every((action) => nativeActions.includes(action)));
  if (!pair)
    throw new Error("This adapter does not advertise a documented inverse camera action pair.");
  const outbound = pair[0] ?? "forward";
  const inbound = pair[1] ?? "backward";
  return [
    { id: newId(), timestampMs: 0, type: "native", action: outbound, values: { pressed: true } },
    {
      id: newId(),
      timestampMs: seconds * 250,
      type: "native",
      action: outbound,
      values: { pressed: false },
    },
    {
      id: newId(),
      timestampMs: seconds * 500,
      type: "native",
      action: inbound,
      values: { pressed: true },
    },
    {
      id: newId(),
      timestampMs: seconds * 750,
      type: "native",
      action: inbound,
      values: { pressed: false },
    },
  ];
}

/** Only real worker responses produce metrics. Each run has an independent seed/session. */
export async function runExperiment(
  plan: ExperimentPlan,
  dependencies: RunnerDependencies,
  signal: AbortSignal,
  progress: (value: ExperimentProgress) => void,
): Promise<ExperimentResult[]> {
  validatePlan(plan);
  signal.throwIfAborted();
  const store = dependencies.store ?? worldStore;
  const now = dependencies.now ?? (() => performance.now());
  const wait = dependencies.sleep ?? sleep;
  const decode = dependencies.decode ?? grayscale;
  const api = dependencies.api;
  const prepareImage = dependencies.prepareImage ?? imageData;
  const characterReferences: string[] = [];
  if (plan.mode === "character-consistency" && plan.character) {
    for (const id of plan.character.assetIds) {
      if (characterReferences.length === 4) break;
      const asset = await store.get("assets", id);
      if (asset?.kind !== "image") continue;
      const blob = await store.getBlob(id);
      if (!blob) throw new Error("Character reference is missing from the local library.");
      signal.throwIfAborted();
      characterReferences.push(await prepareImage(blob));
    }
    if (!characterReferences.length)
      throw new Error("Character consistency needs at least one saved reference image.");
    if (plan.visionConsent) {
      const status = await api.request<{ visionConfigured: boolean }>("/intelligence/status", {
        signal,
      });
      if (!status.visionConfigured)
        throw new Error(
          "Configure the vision evaluator before enabling automatic character assessment, or run without external vision.",
        );
    }
  }
  const experimentId = newId();
  const results: ExperimentResult[] = [];
  let suiteCost = 0;
  const images: string[] = [];
  let video: string | undefined;
  let mediaBytes = 0;
  for (const id of plan.project.assetIds) {
    const asset = await store.get("assets", id);
    if (!asset) throw new Error("A project reference is missing from the local library.");
    if (!["image", "video"].includes(asset.kind))
      throw new Error("Benchmark references currently support images and video only.");
    const blob = await store.getBlob(id);
    if (!blob) throw new Error("A project reference is missing from the local library.");
    mediaBytes += blob.size;
    if (mediaBytes > 4 * 1024 * 1024)
      throw new Error("Benchmark reference media must total less than 4 MB.");
    if (asset.kind === "video") {
      if (video) throw new Error("Choose at most one conditioning video.");
      video = await dataUrl(blob);
    } else images.push(await dataUrl(blob));
    if (images.length > 8) throw new Error("Choose at most eight reference images.");
  }
  const inputFingerprint = await digest(
    new Blob([
      JSON.stringify({
        prompt: plan.project.prompt,
        seed: plan.seed,
        images,
        video,
        characterReferences,
        characterDescription: plan.character?.description,
        quality: plan.project.settings.performance,
      }),
    ]),
  );
  for (const [index, target] of plan.targets.entries()) {
    if (signal.aborted) break;
    const model = getModel(target.modelId);
    const characterView = target.characterView ?? "front";
    const viewPrompt = {
      front: "Keep the character visible from the front.",
      side: "Show the same character from a side view while preserving their clothing, colors, and silhouette.",
      environment:
        "Show the same character in a different environment: an open sunlit courtyard, preserving their clothing and visible design.",
    }[characterView];
    const effectivePrompt = model.capabilities.input.text
      ? plan.mode === "character-consistency" && plan.character
        ? `${plan.project.prompt.slice(0, 5000)}. Character reference: ${plan.character.description.slice(0, 2000)}. ${viewPrompt}`
        : plan.project.prompt
      : "";
    const effectiveImages =
      plan.mode === "character-consistency"
        ? model.capabilities.input.image
          ? model.capabilities.input.multiImage
            ? characterReferences
            : characterReferences.slice(0, 1)
          : []
        : images;
    const effectiveVideo = plan.mode === "character-consistency" ? undefined : video;
    const qualityOptions = (model.capabilities.runtime as { qualityOptions?: string[] })
      .qualityOptions ?? ["balanced"];
    const effectiveQuality = qualityOptions.includes(plan.project.settings.performance)
      ? plan.project.settings.performance
      : "balanced";
    const start = now();
    let worker: WorkerHandle | undefined;
    let sessionId: string | undefined;
    let billedStart: number | undefined;
    let allocationAttempted = false;
    let hourly = 0;
    let firstAt: number | undefined;
    let previous: GrayFrame | undefined;
    let first: GrayFrame | undefined;
    let lastHash = "";
    let lastSample = -Infinity;
    let lastPoll = -Infinity;
    let lastHeartbeat = -Infinity;
    const local = new AbortController();
    const runSignal = AbortSignal.any([signal, local.signal]);
    const timer = setTimeout(
      () => local.abort(new Error("Per-run wall-clock limit reached.")),
      plan.maxWallTimeSeconds * 1000,
    );
    const capture = (dependencies.capture ?? (() => new ExperimentCapture()))();
    const result: ExperimentResult = {
      id: newId(),
      name: `${plan.project.name} · ${target.label ?? target.modelId}`.slice(0, 500),
      projectId: plan.project.id,
      modelId: target.modelId,
      providerId: target.providerId,
      createdAt: Date.now(),
      durationMs: 0,
      frameCount: 0,
      events: [],
      experimentVersion: 1,
      experimentId,
      status: "completed",
      mode: plan.mode,
      seed: plan.seed,
      inputFingerprint,
      targetLabel: target.label ?? target.modelId,
      effectivePrompt,
      effectiveQuality,
      characterEvaluation:
        plan.mode === "character-consistency" && plan.character
          ? {
              characterId: plan.character.id,
              characterName: plan.character.name,
              view: characterView,
              conditioningMethod: model.capabilities.input.image
                ? "Reference image conditioning plus supported text; no native identity guarantee"
                : "Text description only; reference images are used only for evaluation",
              samples: [],
              errors: [],
              source: "not-evaluated",
              metric: "qualitative-appearance-consistency",
            }
          : undefined,
      discoveryAction: target.discoveryAction,
      assetIds: [],
      observations: [],
      actionObservations: [],
      deliveredFPS: 0,
      errors: [],
      cleanupErrors: [],
      billedCostUSD: null,
    };
    const update = (phase: string) =>
      progress({
        target: result.targetLabel,
        index: index + 1,
        total: plan.targets.length,
        phase,
        elapsedSeconds: (now() - start) / 1000,
        frames: result.frameCount,
        estimatedCostUSD:
          suiteCost +
          (billedStart === undefined ? 0 : ((now() - billedStart) / 3_600_000) * hourly),
      });
    const request = <T>(path: string, init: RequestInit = {}, raw = false) =>
      api.request<T>(path, { ...init, signal: runSignal }, raw);
    try {
      if (
        plan.mode === "character-consistency" &&
        characterView !== "front" &&
        !model.capabilities.input.text
      )
        throw new Error(
          "This adapter cannot receive the requested view/environment prompt; choose a text-conditioned model for this variant.",
        );
      const trajectory: ControlEvent[] =
        plan.mode === "return-to-view"
          ? returnTrajectory(model.nativeActions, plan.durationSeconds)
          : target.discoveryAction
            ? [
                {
                  id: newId(),
                  timestampMs: 0,
                  type: "native",
                  action: target.discoveryAction,
                  values: { pressed: true },
                },
                {
                  id: newId(),
                  timestampMs: Math.min(2000, plan.durationSeconds * 1000),
                  type: "native",
                  action: target.discoveryAction,
                  values: { pressed: false },
                },
              ]
            : [...plan.events].sort((a, b) => a.timestampMs - b.timestampMs);
      result.requestedEvents = trajectory;
      update("Checking compute price");
      const quote = await request<Quote>(`/providers/${target.providerId}/quote`, {
        method: "POST",
        body: JSON.stringify({
          modelId: target.modelId,
          durationMinutes: plan.maxWallTimeSeconds / 60,
        }),
      });
      if (quote.available === false) throw new Error("Selected GPU/model profile is unavailable.");
      if (
        target.providerId !== "local" &&
        (quote.hourlyCost === null || !Number.isFinite(quote.hourlyCost))
      )
        throw new Error(
          "A server/provider price quote is required for bounded remote benchmark runs. Configure a rate or use local compute.",
        );
      hourly = quote.hourlyCost ?? 0;
      if (hourly < 0) throw new Error("Invalid provider price.");
      result.priceSource = quote.pricingSource;
      if (suiteCost + (hourly * plan.maxWallTimeSeconds) / 3600 > plan.maxEstimatedCostUSD)
        throw new Error(
          "Worst-case estimated run cost exceeds the remaining suite budget. Raise the explicit budget or shorten the wall-clock limit.",
        );
      update("Starting model worker");
      billedStart = now();
      // Do not abort an in-flight allocation request: retain its returned ID so cancellation can clean it up.
      allocationAttempted = true;
      worker = await api.request<WorkerHandle>("/workers", {
        method: "POST",
        body: JSON.stringify({ provider: target.providerId, modelId: target.modelId }),
        signal: AbortSignal.timeout(120_000),
      });
      runSignal.throwIfAborted();
      while (worker.status !== "ready") {
        if (["failed", "error", "destroyed", "stopped"].includes(worker.status))
          throw new Error(worker.error ?? "Model worker failed to start.");
        update("Waiting for worker readiness");
        await wait(1500, runSignal);
        worker = await request<WorkerHandle>(`/workers/${worker.id}`);
      }
      update("Loading model and first frame");
      // As with worker allocation, wait for the identifier before honoring cancellation.
      const session = await api.request<LiveSession>("/sessions", {
        method: "POST",
        body: JSON.stringify({
          workerId: worker.id,
          modelId: target.modelId,
          prompt: effectivePrompt,
          seed: plan.seed,
          quality: effectiveQuality,
          inputs: { images: effectiveImages, video: effectiveVideo },
        }),
        signal: AbortSignal.timeout(120_000),
      });
      sessionId = session.id;
      runSignal.throwIfAborted();
      let eventIndex = 0;
      while (firstAt === undefined || now() - firstAt < plan.durationSeconds * 1000) {
        runSignal.throwIfAborted();
        if (now() - start >= plan.maxWallTimeSeconds * 1000)
          throw new Error("Per-run wall-clock limit reached.");
        const elapsed = firstAt === undefined ? 0 : now() - firstAt;
        while (
          firstAt !== undefined &&
          eventIndex < trajectory.length &&
          (trajectory[eventIndex]?.timestampMs ?? Infinity) <= elapsed
        ) {
          const event = trajectory[eventIndex];
          eventIndex++;
          if (!event) continue;
          const dispatched = now();
          const observation = {
            eventId: event.id,
            scheduledMs: event.timestampMs,
            dispatchedMs: dispatched - firstAt,
            latencyMs: 0,
            accepted: false,
            error: undefined as string | undefined,
          };
          try {
            await request(`/sessions/${sessionId}/actions`, {
              method: "POST",
              body: JSON.stringify({
                type: event.type,
                action: event.action,
                prompt: event.prompt,
                values: event.values,
              }),
            });
            observation.accepted = true;
            result.events.push(event);
          } catch (error) {
            if (runSignal.aborted) throw error;
            observation.error = errorText(error);
          }
          observation.latencyMs = now() - dispatched;
          result.actionObservations.push(observation);
        }
        if (now() - lastHeartbeat > 10_000) {
          await request(`/sessions/${sessionId}/heartbeat`, {
            method: "POST",
            body: JSON.stringify({ active: true }),
          });
          lastHeartbeat = now();
        }
        if (now() - lastPoll > 1500) {
          const state = await request<LiveSession>(`/sessions/${sessionId}`);
          lastPoll = now();
          if (["error", "failed", "stopped"].includes(state.status))
            throw new Error(state.error ?? `Session ${state.status}.`);
          if (typeof state.generatedFPS === "number" && Number.isFinite(state.generatedFPS))
            result.generatedFPS = state.generatedFPS;
          if (typeof state.totalGeneratedFrames === "number")
            result.sourceFrameCount = state.totalGeneratedFrames;
        }
        const blob = await request<Blob | null>(
          `/sessions/${sessionId}/frame`,
          { cache: "no-store" },
          true,
        );
        if (blob?.size) {
          const hash = await digest(blob);
          if (hash !== lastHash) {
            lastHash = hash;
            const timestamp = now();
            if (firstAt === undefined) {
              firstAt = timestamp;
              result.firstFrameMs = timestamp - start;
              result.startupMs = timestamp - start;
            }
            const frame = await decode(blob);
            if (frame.sourceWidth && frame.sourceHeight)
              result.resolution = `${frame.sourceWidth}x${frame.sourceHeight}`;
            const observation = {
              timestampMs: timestamp - firstAt,
              ...measureFrame(previous, frame),
              assetId: undefined as string | undefined,
            };
            first ??= frame;
            previous = frame;
            result.frameCount++;
            if (timestamp - lastSample >= 3000 && result.assetIds.length < 24) {
              const asset = await store.saveAsset(
                blob,
                `${result.targetLabel} · frame ${result.frameCount}`,
                "image",
              );
              observation.assetId = asset.id;
              result.assetIds.push(asset.id);
              lastSample = timestamp;
            }
            result.observations.push(observation);
            await capture.frame(blob);
          }
        }
        update(
          firstAt === undefined
            ? "Waiting for real generated frames"
            : "Replaying controls and measuring",
        );
        await wait(250, runSignal);
      }
      if (!result.frameCount) throw new Error("No generated frames were delivered.");
    } catch (error) {
      result.status = signal.aborted ? "cancelled" : "failed";
      result.errors.push(errorText(error));
    } finally {
      clearTimeout(timer);
      if (allocationAttempted && !worker)
        result.cleanupErrors.push(
          "Allocation returned no worker identifier. Stop this suite and check Settings → workers; the server reaper remains responsible for any uncertain allocation.",
        );
      update("Closing session and compute");
      result.durationMs = firstAt === undefined ? 0 : Math.max(0, now() - firstAt);
      result.deliveredFPS = result.durationMs ? result.frameCount / (result.durationMs / 1000) : 0;
      result.measuredFPS = result.generatedFPS;
      if (first && previous) {
        result.returnFrameSimilarity = frameSimilarity(first, previous);
        const alignment = measureFrame(first, previous);
        result.returnAlignment = {
          x: alignment.motionX ?? 0,
          y: alignment.motionY ?? 0,
          confidence: alignment.motionConfidence ?? 0,
        };
      }
      if (result.actionObservations.length)
        result.latencyMs =
          result.actionObservations.reduce((sum, event) => sum + event.latencyMs, 0) /
          result.actionObservations.length;
      for (const path of [
        sessionId ? `/sessions/${sessionId}` : undefined,
        worker ? `/workers/${worker.id}` : undefined,
      ]) {
        if (!path) continue;
        try {
          await api.request(path, { method: "DELETE", signal: AbortSignal.timeout(30_000) });
        } catch (error) {
          result.cleanupErrors.push(`${path}: ${errorText(error)}`);
        }
      }
      result.estimatedCostUSD =
        billedStart === undefined ? 0 : ((now() - billedStart) / 3_600_000) * hourly;
      suiteCost += result.estimatedCostUSD;
      try {
        const video = await capture.finish();
        if (video?.size) {
          const asset = await store.saveAsset(video, `${result.name} · sampled capture`, "video");
          result.videoAssetId = asset.id;
          result.assetIds.push(asset.id);
          await store.put("replays", {
            id: newId(),
            name: result.name,
            projectId: result.projectId,
            modelId: result.modelId,
            assetId: asset.id,
            durationMs: result.durationMs,
            createdAt: Date.now(),
            events: result.events,
          });
        }
      } catch (error) {
        result.errors.push(`Capture: ${errorText(error)}`);
      }
      if (
        result.characterEvaluation &&
        plan.character &&
        plan.visionConsent &&
        !signal.aborted &&
        result.status === "completed" &&
        !result.cleanupErrors.length
      ) {
        update("Evaluating actual character samples; GPU session closed");
        for (const sample of evaluationSamples(result.observations, plan.evaluationFrames ?? 2)) {
          if (signal.aborted || !sample.assetId) break;
          try {
            const blob = await store.getBlob(sample.assetId);
            if (!blob) {
              result.characterEvaluation.errors.push("Generated sample is missing.");
              continue;
            }
            const candidate = await prepareImage(blob);
            signal.throwIfAborted();
            const assessment = validatedAssessment(
              await api.request<unknown>("/characters/evaluate", {
                method: "POST",
                body: JSON.stringify({
                  description: plan.character.description.slice(0, 4000),
                  references: characterReferences,
                  candidates: [candidate],
                }),
                signal: AbortSignal.any([signal, AbortSignal.timeout(60_000)]),
              }),
            );
            result.characterEvaluation.samples.push({
              assetId: sample.assetId,
              timestampMs: sample.timestampMs,
              score: assessment.score,
              confidence: assessment.confidence,
              evidence: assessment.evidence,
              differences: assessment.differences,
            });
          } catch (error) {
            result.characterEvaluation.errors.push(errorText(error));
            if (signal.aborted) break;
          }
        }
        updateCharacterAggregate(result);
      }
      await store.put("benchmarks", result);
      results.push(result);
      update(result.status);
    }
    if (result.cleanupErrors.length || signal.aborted) break;
  }
  return results;
}
