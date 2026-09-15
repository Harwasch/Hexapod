import type { Scene, Viewer } from "cesium";

import type { Emitter } from "@/lib/emitter";
import { QUALITY_SSE, type QualityPreset } from "@/state/settings";

import type { SceneEvents } from "./types";

export interface QualityInputs {
  preset: QualityPreset;
  manualScreenSpaceError: number | null;
  adaptive: boolean;
}

/** Everything the screen-space-error decision looks at, so it can be tested without a scene. */
export interface QualitySample {
  bounds: { base: number; min: number; max: number };
  current: number;
  /** Rendered frames per second, or null when the scene is idle (request-render mode). */
  fps: number | null;
  moving: boolean;
  loading: boolean;
  nearSite: boolean;
  altitude: number;
  /** Tileset memory in use divided by its cache budget. */
  memoryRatio: number;
}

export interface QualityDecision {
  screenSpaceError: number;
  reason: string;
}

const LOW_FPS = 26;
const STEADY_FPS = 50;
const MEMORY_PRESSURE_RATIO = 1.25;
/** Minimum rendered frames in the last second before the frame rate is trusted. */
const MIN_FRAMES_FOR_FPS = 6;
/** Low frame rate must persist this long while moving before quality is cut. */
const SUSTAINED_LOW_MS = 1200;
/** Motion at a steady frame rate needed before a cut is undone; doubles after every recovery. */
const INITIAL_RECOVERY_MS = 8000;
const MAX_RECOVERY_MS = 60_000;
/** A recovery (resolution back up) only happens once the camera has rested this long. */
const REST_BEFORE_RECOVERY_MS = 1500;
/** Below this height above ground a resting camera refines towards the preset minimum. */
const CLOSE_UP_ALTITUDE_M = 600;
/** Extra screen-space error per ladder step once resolution cuts are exhausted. */
const SSE_PENALTY_STEP = 3;

/**
 * Chooses the next maximum screen-space error. Pure so the policy is unit-testable.
 *
 * Smoothness first: while the camera moves the tile selection is frozen, because every change
 * pops tiles mid-gesture. At rest, only memory pressure coarsens (slow frames at rest are
 * tiles arriving, not a stall) and the value walks back to the preset base, which the
 * manager shifts upwards once its ladder has run out of resolution to cut. Refinement happens
 * at rest, close to a site, one step per tick, so the still image sharpens without ever
 * fighting a gesture.
 */
export function decideScreenSpaceError(sample: QualitySample): QualityDecision {
  const { bounds, current, fps, moving, loading } = sample;
  if (sample.memoryRatio > MEMORY_PRESSURE_RATIO) {
    return {
      screenSpaceError: Math.min(bounds.max, current + 4),
      reason: `memory pressure (${Math.round(sample.memoryRatio * 100)}% of budget)`,
    };
  }
  if (moving) return { screenSpaceError: current, reason: "moving (tiles held)" };
  const closeUp = sample.nearSite && sample.altitude < CLOSE_UP_ALTITUDE_M;
  if (!loading && closeUp && (fps === null || fps > STEADY_FPS + 2)) {
    return {
      screenSpaceError: Math.min(bounds.max, Math.max(bounds.min, current - 2)),
      reason: "close-up refinement",
    };
  }
  if (fps === null) {
    return {
      screenSpaceError: loading ? current : bounds.base,
      reason: loading ? "loading" : "idle",
    };
  }
  if (!loading && fps > STEADY_FPS) return { screenSpaceError: bounds.base, reason: "steady" };
  return { screenSpaceError: current, reason: loading ? "loading" : "steady" };
}

/**
 * Gaussian splats are re-sorted on the CPU every camera change, so their cost grows with splat
 * count far faster than a mesh. This is the finest screen-space error a splat may use per preset.
 */
export function splatMinimumScreenSpaceError(preset: QualityPreset): number {
  return { performance: 12, balanced: 8, ultra: 4 }[preset];
}

/** "full" renders the preset as configured; "reduced" means the ladder has cut something. */
export type RenderProfile = "full" | "reduced";

interface LadderStep {
  msaa: number;
  scale: number;
  /** Added to the preset's base and minimum screen-space error. */
  ssePenalty: number;
  label: string;
}

/**
 * Keeps motion smooth by rendering with settings that never change during a gesture, and
 * adapts those settings only on evidence: a frame rate that stays low while the camera
 * moves cuts anti-aliasing, then resolution, then tile detail, one step at a time. A cut is
 * undone only after sustained smooth motion, while the camera rests (a resolution switch
 * re-allocates the framebuffers, which is a visible hitch mid-gesture), and each recovery
 * has to earn twice as much smooth motion as the last so a borderline machine does not
 * oscillate. Screen-space error is resolution independent in Cesium (it divides by the pixel
 * ratio), so resolution steps never change which tiles are drawn.
 */
export class PerformanceManager {
  private readonly scene: Scene;
  private readonly frameTimestamps: number[] = [];
  private inputs: QualityInputs = {
    preset: "balanced",
    manualScreenSpaceError: null,
    adaptive: true,
  };
  private currentSse: number = QUALITY_SSE.balanced.base;
  private ladder: LadderStep[] = [];
  private level = 0;
  private resolutionScale = 1;
  private msaa = 4;
  private lowFpsSince: number | null = null;
  private goodMotionMs = 0;
  private recoveryMs = INITIAL_RECOVERY_MS;
  private restSince = 0;
  private lastEvaluateAt = 0;
  private pending = 0;
  private processing = 0;
  private moving = false;
  private movingUntil = 0;
  private altitude = Number.POSITIVE_INFINITY;
  private nearSite = false;
  private readonly timer: ReturnType<typeof setInterval>;
  private readonly unsubscribe: (() => void)[] = [];
  private applySse: (sse: number) => void = () => undefined;
  private memorySource: () => { bytes: number; budget: number } = () => ({ bytes: 0, budget: 1 });
  private readonly gpu: string | null;
  private readonly webgl2: boolean;
  /** Frame intervals recorded while the camera moved, since the last reset (site change). */
  private readonly motionFrameMs: number[] = [];
  private lastMotionFrameAt: number | null = null;

  constructor(
    private readonly viewer: Viewer,
    private readonly events: Emitter<SceneEvents>,
  ) {
    this.scene = viewer.scene;
    const info = readGpuInfo(viewer.canvas);
    this.gpu = info.renderer;
    this.webgl2 = info.webgl2;
    this.unsubscribe.push(
      this.scene.postRender.addEventListener(() => this.onFrame()),
      // Motion is tracked from real pose changes only (moveStart also fires on frustum and
      // canvas size changes). Nothing about the render settings changes here: a gesture
      // always runs with whatever the ladder settled on.
      viewer.camera.changed.addEventListener(() => {
        this.moving = true;
      }),
      viewer.camera.moveEnd.addEventListener(() => {
        if (!this.moving) return;
        this.moving = false;
        this.movingUntil = performance.now() + 400;
        this.restSince = performance.now();
      }),
    );
    this.timer = setInterval(() => this.evaluate(), 500);
    this.events.emit("performance", {
      gpu: this.gpu,
      webgl2: this.webgl2,
      devicePixelRatio: window.devicePixelRatio,
    });
  }

  /** Called by the SiteManager so decisions reach every site tileset. */
  bindScreenSpaceErrorSink(apply: (sse: number) => void): void {
    this.applySse = apply;
    apply(this.currentSse);
  }

  /** Lets the SiteManager report how much memory the active tilesets hold versus their budget. */
  bindMemorySource(source: () => { bytes: number; budget: number }): void {
    this.memorySource = source;
  }

  configure(inputs: QualityInputs): void {
    this.inputs = inputs;
    const bounds = QUALITY_SSE[inputs.preset];
    this.currentSse = inputs.manualScreenSpaceError ?? bounds.base;
    this.applySse(this.currentSse);
    this.scene.globe.maximumScreenSpaceError = inputs.preset === "performance" ? 3 : 2;
    // Still and moving frames share one resolution. Performance renders at the browser's
    // recommended (CSS pixel) resolution; the others use native device pixels until the
    // ladder proves the machine cannot keep up.
    this.viewer.useBrowserRecommendedResolution = inputs.preset === "performance";
    this.ladder = buildLadder(inputs.preset);
    this.level = 0;
    this.lowFpsSince = null;
    this.goodMotionMs = 0;
    this.recoveryMs = INITIAL_RECOVERY_MS;
    this.applyLevel();
    this.evaluate("configured");
  }

  reportLoading(pending: number, processing: number): void {
    this.pending = pending;
    this.processing = processing;
  }

  reportContext(altitude: number, nearSite: boolean): void {
    this.altitude = altitude;
    this.nearSite = nearSite;
  }

  get screenSpaceError(): number {
    return this.currentSse;
  }

  get splatMinimumScreenSpaceError(): number {
    return splatMinimumScreenSpaceError(this.inputs.preset);
  }

  get profile(): RenderProfile {
    return this.level === 0 ? "full" : "reduced";
  }

  /** The ladder step currently applied (step 0 is the preset as configured). */
  private get step(): LadderStep {
    return (
      this.ladder[this.level] ??
      this.ladder[this.ladder.length - 1] ?? {
        msaa: 4,
        scale: 1,
        ssePenalty: 0,
        label: "full",
      }
    );
  }

  private applyLevel(): void {
    const step = this.step;
    this.setResolutionScale(step.scale);
    this.setMsaa(step.msaa);
    this.scene.requestRender();
  }

  /** Rendered frames per second over the last second, or null when the scene is idle. */
  get fps(): number | null {
    const now = performance.now();
    const cutoff = now - 1000;
    while (this.frameTimestamps.length > 0 && (this.frameTimestamps[0] ?? 0) < cutoff)
      this.frameTimestamps.shift();
    const count = this.frameTimestamps.length;
    if (count < MIN_FRAMES_FOR_FPS) return null;
    const first = this.frameTimestamps[0] ?? now;
    const span = Math.max(1, now - first);
    return ((count - 1) * 1000) / span;
  }

  private onFrame(): void {
    const now = performance.now();
    this.frameTimestamps.push(now);
    if (this.frameTimestamps.length > 240) this.frameTimestamps.shift();
    if (this.moving) {
      if (this.lastMotionFrameAt !== null) {
        const dt = now - this.lastMotionFrameAt;
        if (dt > 0 && dt < 2000) {
          this.motionFrameMs.push(dt);
          if (this.motionFrameMs.length > 600) this.motionFrameMs.shift();
        }
      }
      this.lastMotionFrameAt = now;
    } else {
      this.lastMotionFrameAt = null;
    }
  }

  /** Starts a fresh benchmark window, e.g. when another site becomes active. */
  resetBenchmark(): void {
    this.motionFrameMs.length = 0;
    this.lastMotionFrameAt = null;
  }

  /** Motion-only statistics for comparing datasets: mean fps and 95th percentile frame time. */
  get benchmark(): { motionFps: number | null; p95FrameMs: number | null; samples: number } {
    const n = this.motionFrameMs.length;
    if (n < 5) return { motionFps: null, p95FrameMs: null, samples: n };
    const sorted = [...this.motionFrameMs].sort((a, b) => a - b);
    const mean = sorted.reduce((a, b) => a + b, 0) / n;
    const p95 = sorted[Math.min(n - 1, Math.floor(n * 0.95))] ?? mean;
    return { motionFps: 1000 / mean, p95FrameMs: p95, samples: n };
  }

  private setResolutionScale(scale: number): void {
    if (Math.abs(this.resolutionScale - scale) < 0.01) return;
    this.resolutionScale = scale;
    this.viewer.resolutionScale = scale;
  }

  private setMsaa(samples: number): void {
    if (this.msaa === samples) return;
    this.msaa = samples;
    this.scene.msaaSamples = samples;
    // FXAA is redundant on top of MSAA and costs a full-screen pass.
    this.scene.postProcessStages.fxaa.enabled = samples === 1;
    this.scene.requestRender();
  }

  /**
   * The ladder: one step down after a sustained low frame rate while moving, one step up
   * after enough smooth motion, applied at rest. Returns a reason when a step was taken.
   */
  private climbLadder(now: number, fps: number | null, moving: boolean): string | null {
    const elapsed = this.lastEvaluateAt ? Math.min(1000, now - this.lastEvaluateAt) : 0;
    if (moving && fps !== null) {
      if (fps < LOW_FPS) {
        this.goodMotionMs = 0;
        this.lowFpsSince ??= now;
        if (now - this.lowFpsSince >= SUSTAINED_LOW_MS && this.level < this.ladder.length - 1) {
          this.level += 1;
          this.lowFpsSince = now;
          this.applyLevel();
          return `low fps (${fps.toFixed(0)}) → ${this.step.label}`;
        }
        return null;
      }
      this.lowFpsSince = null;
      if (fps > STEADY_FPS) this.goodMotionMs += elapsed;
      return null;
    }
    this.lowFpsSince = null;
    if (
      !moving &&
      this.level > 0 &&
      this.goodMotionMs >= this.recoveryMs &&
      now - this.restSince >= REST_BEFORE_RECOVERY_MS
    ) {
      this.level -= 1;
      this.goodMotionMs = 0;
      this.recoveryMs = Math.min(MAX_RECOVERY_MS, this.recoveryMs * 2);
      this.applyLevel();
      return `smooth motion → ${this.step.label}`;
    }
    return null;
  }

  private evaluate(forcedReason?: string): void {
    const fps = this.fps;
    const now = performance.now();
    const moving = this.moving || now < this.movingUntil;
    const loading = this.pending > 0 || this.processing > 0;
    const preset = QUALITY_SSE[this.inputs.preset];
    const memory = this.memorySource();
    const memoryRatio = memory.budget > 0 ? memory.bytes / memory.budget : 0;
    let target = this.inputs.manualScreenSpaceError ?? preset.base;
    let reason: string;

    if (this.inputs.adaptive && this.inputs.manualScreenSpaceError === null) {
      const stepReason = this.climbLadder(now, fps, moving);
      const penalty = this.step.ssePenalty;
      const bounds = {
        base: Math.min(preset.max, preset.base + penalty),
        min: Math.min(preset.max, preset.min + penalty),
        max: preset.max,
      };
      const decision = decideScreenSpaceError({
        bounds,
        current: this.currentSse,
        fps,
        moving,
        loading,
        nearSite: this.nearSite,
        altitude: this.altitude,
        memoryRatio,
      });
      target = decision.screenSpaceError;
      reason = forcedReason ?? stepReason ?? decision.reason;
    } else {
      reason = forcedReason ?? (this.inputs.manualScreenSpaceError !== null ? "manual" : "fixed");
    }
    this.lastEvaluateAt = now;

    if (Math.abs(target - this.currentSse) >= 0.5) {
      this.currentSse = Math.round(target * 2) / 2;
      this.applySse(this.currentSse);
    }

    this.events.emit("performance", {
      gpu: this.gpu,
      webgl2: this.webgl2,
      profile: this.profile,
      fps: fps === null ? 0 : Math.round(fps),
      frameTimeMs: fps === null ? 0 : Math.round((1000 / fps) * 10) / 10,
      rendering: fps !== null,
      resolutionScale: this.resolutionScale,
      msaaSamples: this.msaa,
      devicePixelRatio: window.devicePixelRatio,
      pendingRequests: this.pending,
      tilesProcessing: this.processing,
      siteScreenSpaceError: this.currentSse,
      adaptiveReason: reason,
      moving,
      tilesetMemoryMb: Math.round(memory.bytes / 1048576),
      memoryBudgetMb: Math.round(memory.budget / 1048576),
      benchmark: this.benchmark,
    });
  }

  destroy(): void {
    clearInterval(this.timer);
    for (const off of this.unsubscribe) off();
  }
}

/**
 * Cheapest savings first: anti-aliasing, then resolution in three steps down to half, then
 * coarser tiles (the only step that changes what is drawn). Exported for the unit tests.
 */
export function buildLadder(preset: QualityPreset): LadderStep[] {
  const msaa = preset === "performance" ? 1 : 4;
  const steps: LadderStep[] = [{ msaa, scale: 1, ssePenalty: 0, label: "full" }];
  if (msaa > 1) steps.push({ msaa: 1, scale: 1, ssePenalty: 0, label: "MSAA off" });
  for (const scale of [0.8, 0.65, 0.5])
    steps.push({ msaa: 1, scale, ssePenalty: 0, label: `resolution ${scale}` });
  const { base, max } = QUALITY_SSE[preset];
  for (let penalty = SSE_PENALTY_STEP; base + penalty <= max; penalty += SSE_PENALTY_STEP)
    steps.push({ msaa: 1, scale: 0.5, ssePenalty: penalty, label: `tiles +${penalty} SSE` });
  return steps;
}

/** Reads the renderer through the public canvas API; Cesium already owns the context. */
function readGpuInfo(canvas: HTMLCanvasElement): { renderer: string | null; webgl2: boolean } {
  try {
    const gl2 = canvas.getContext("webgl2");
    const gl = gl2 ?? canvas.getContext("webgl");
    if (!gl) return { renderer: null, webgl2: false };
    const ext = gl.getExtension("WEBGL_debug_renderer_info") as {
      UNMASKED_RENDERER_WEBGL: number;
    } | null;
    const renderer = ext
      ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL))
      : String(gl.getParameter(gl.RENDERER));
    return { renderer, webgl2: gl2 !== null };
  } catch {
    return { renderer: null, webgl2: false };
  }
}
