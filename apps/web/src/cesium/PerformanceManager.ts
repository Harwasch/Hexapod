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
  moving: boolean;
  /** Any tileset (sites or the world) still has requests pending or tiles processing. */
  loading: boolean;
  /** Highest tileset memory in use divided by its cache budget, over all tilesets. */
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
/** Slow motion frames (below LOW_FPS) must add up to this much before quality is cut; smooth
 *  frames pay it back, so the evidence accumulates across short gestures. */
const SUSTAINED_LOW_MS = 800;
/** Motion at a steady frame rate needed before a cut is undone; doubles after every recovery. */
const INITIAL_RECOVERY_MS = 8000;
const MAX_RECOVERY_MS = 60_000;
/** A recovery (resolution back up) only happens once the camera has rested this long. */
const REST_BEFORE_RECOVERY_MS = 1500;
/** Idle refinement only proceeds while tileset memory is below this share of its budget. */
const REFINE_MEMORY_RATIO = 0.7;
/** Extra screen-space error per ladder step once resolution cuts are exhausted. */
const SSE_PENALTY_STEP = 3;

/**
 * Chooses the next maximum screen-space error. Pure so the policy is unit-testable.
 *
 * Smoothness first: while the camera moves the tile selection is frozen, because every change
 * pops tiles mid-gesture. At rest, memory pressure coarsens (the only thing that ever does,
 * apart from the manager's ladder shifting the bounds), loading holds, and otherwise the
 * scene uses the idle time the way a maps app does: one step finer per tick, at any height,
 * as long as there is memory headroom. Slow frames at rest are tiles arriving, never a
 * reason to coarsen. Nothing returns to the base on its own: finer tiles stay until memory
 * says otherwise, so the next gesture starts from what is already loaded.
 */
export function decideScreenSpaceError(sample: QualitySample): QualityDecision {
  const { bounds, current, moving, loading, memoryRatio } = sample;
  if (memoryRatio > MEMORY_PRESSURE_RATIO) {
    return {
      screenSpaceError: Math.min(bounds.max, current + 4),
      reason: `memory pressure (${Math.round(memoryRatio * 100)}% of budget)`,
    };
  }
  if (moving) return { screenSpaceError: current, reason: "moving (tiles held)" };
  if (loading) return { screenSpaceError: current, reason: "loading" };
  if (memoryRatio >= REFINE_MEMORY_RATIO)
    return {
      screenSpaceError: current,
      reason: `holding (${Math.round(memoryRatio * 100)}% of memory budget)`,
    };
  const target = Math.min(bounds.max, Math.max(bounds.min, current - 2));
  if (target === current) return { screenSpaceError: current, reason: "at finest" };
  return { screenSpaceError: target, reason: "idle refinement" };
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

/** Receives the screen-space error in CSS pixels and the pixel ratio the scene renders at. */
export type ScreenSpaceErrorSink = (sse: number, pixelRatio: number) => void;

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
  private slowMotionMs = 0;
  private goodMotionMs = 0;
  private recoveryMs = INITIAL_RECOVERY_MS;
  private restSince = 0;
  private pending = 0;
  private processing = 0;
  private moving = false;
  private movingUntil = 0;
  private altitude = Number.POSITIVE_INFINITY;
  private nearSite = false;
  private readonly timer: ReturnType<typeof setInterval>;
  private readonly unsubscribe: (() => void)[] = [];
  private readonly sinks: ScreenSpaceErrorSink[] = [];
  private readonly memorySources: (() => { bytes: number; budget: number })[] = [];
  private readonly loadingBySource = new Map<string, { pending: number; processing: number }>();
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

  /**
   * Registers a consumer of the current screen-space error. It receives the error in CSS
   * pixels together with the pixel ratio the scene renders at, so consumers can divide and
   * hand Cesium an error in device pixels: Cesium measures screen-space error in CSS pixels,
   * which on a HiDPI screen picks tiles twice as coarse as they look, and a maps app chooses
   * detail by the pixels you actually see.
   */
  addScreenSpaceErrorSink(apply: ScreenSpaceErrorSink): void {
    this.sinks.push(apply);
    apply(this.currentSse, this.pixelRatio);
  }

  /** Resolution scale times the device pixel ratio when rendering at native resolution. */
  get pixelRatio(): number {
    const dpr = this.viewer.useBrowserRecommendedResolution ? 1 : window.devicePixelRatio || 1;
    return Math.max(0.25, this.resolutionScale * dpr);
  }

  private applySse(sse: number): void {
    const ratio = this.pixelRatio;
    for (const sink of this.sinks) sink(sse, ratio);
  }

  /** Registers a tileset group's memory use against its cache budget (sites, the world). */
  addMemorySource(source: () => { bytes: number; budget: number }): void {
    this.memorySources.push(source);
  }

  /** The most loaded group's share of its budget; each tileset has its own cache. */
  private get memoryRatio(): number {
    let ratio = 0;
    for (const source of this.memorySources) {
      const { bytes, budget } = source();
      if (budget > 0) ratio = Math.max(ratio, bytes / budget);
    }
    return ratio;
  }

  private get memoryBytes(): { bytes: number; budget: number } {
    let bytes = 0;
    let budget = 0;
    for (const source of this.memorySources) {
      const m = source();
      bytes += m.bytes;
      budget += m.budget;
    }
    return { bytes, budget };
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
    this.slowMotionMs = 0;
    this.goodMotionMs = 0;
    this.recoveryMs = INITIAL_RECOVERY_MS;
    this.applyLevel();
    this.evaluate("configured");
  }

  /** Loading state per tileset group (sites, the world); the decision waits for all of them. */
  reportLoading(source: string, pending: number, processing: number): void {
    this.loadingBySource.set(source, { pending, processing });
    let p = 0;
    let q = 0;
    for (const entry of this.loadingBySource.values()) {
      p += entry.pending;
      q += entry.processing;
    }
    this.pending = p;
    this.processing = q;
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
    // A resolution step changes the pixel ratio, and with it the device-pixel error.
    this.applySse(this.currentSse);
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
          // Evidence for the ladder comes from every motion frame, so three short slow
          // drags count as much as one long one.
          if (dt > 1000 / LOW_FPS) {
            this.slowMotionMs += dt;
            this.goodMotionMs = 0;
          } else {
            this.slowMotionMs = Math.max(0, this.slowMotionMs - dt);
            if (dt < 1000 / STEADY_FPS) this.goodMotionMs += dt;
          }
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
   * The ladder: one step down once slow motion frames have added up (applied at once, even
   * mid-gesture, because a hitch beats staying slow), one step up after enough smooth
   * motion, applied at rest. Returns a reason when a step was taken.
   */
  private climbLadder(now: number, moving: boolean): string | null {
    if (this.slowMotionMs >= SUSTAINED_LOW_MS && this.level < this.ladder.length - 1) {
      this.level += 1;
      this.slowMotionMs = 0;
      this.goodMotionMs = 0;
      this.applyLevel();
      return `slow motion → ${this.step.label}`;
    }
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
    const memory = this.memoryBytes;
    const memoryRatio = this.memoryRatio;
    let target = this.inputs.manualScreenSpaceError ?? preset.base;
    let reason: string;

    if (this.inputs.adaptive && this.inputs.manualScreenSpaceError === null) {
      const stepReason = this.climbLadder(now, moving);
      const penalty = this.step.ssePenalty;
      const bounds = {
        base: Math.min(preset.max, preset.base + penalty),
        min: Math.min(preset.max, preset.min + penalty),
        max: preset.max,
      };
      const decision = decideScreenSpaceError({
        bounds,
        current: this.currentSse,
        moving,
        loading,
        memoryRatio,
      });
      target = decision.screenSpaceError;
      reason = forcedReason ?? stepReason ?? decision.reason;
    } else {
      reason = forcedReason ?? (this.inputs.manualScreenSpaceError !== null ? "manual" : "fixed");
    }
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
