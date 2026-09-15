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

const LOW_FPS = 28;
const STEADY_FPS = 50;
const MEMORY_PRESSURE_RATIO = 1.25;
/** Minimum rendered frames in the last second before the frame rate is trusted. */
const MIN_FRAMES_FOR_FPS = 6;
const SUSTAINED_LOW_MS = 2500;
/** Below this height above ground a resting camera refines towards the preset minimum. */
const CLOSE_UP_ALTITUDE_M = 600;

/**
 * Chooses the next maximum screen-space error. Pure so the policy is unit-testable.
 *
 * Priorities, highest first: memory pressure, motion, low frame rate, then refinement only
 * when there is measured headroom. An idle scene (no frames rendered) never changes quality,
 * because there is no evidence either way.
 */
export function decideScreenSpaceError(sample: QualitySample): QualityDecision {
  const { bounds, current, fps, moving, loading } = sample;
  if (sample.memoryRatio > MEMORY_PRESSURE_RATIO) {
    return {
      screenSpaceError: Math.min(bounds.max, current + 4),
      reason: `memory pressure (${Math.round(sample.memoryRatio * 100)}% of budget)`,
    };
  }
  if (moving) {
    return {
      screenSpaceError: Math.min(bounds.max, bounds.base + Math.max(4, bounds.base * 0.5)),
      reason: "moving",
    };
  }
  if (fps !== null && fps < LOW_FPS) {
    return {
      screenSpaceError: Math.min(bounds.max, current + 3),
      reason: `low fps (${fps.toFixed(0)})`,
    };
  }
  // At rest the scene only renders while tiles arrive, so an idle view is headroom by
  // definition: walk towards the fine end one step at a time. The next drag coarsens again
  // ("moving"), and memory pressure above caps the walk.
  const closeUp = sample.nearSite && sample.altitude < CLOSE_UP_ALTITUDE_M;
  if (!loading && closeUp && (fps === null || fps > STEADY_FPS + 2)) {
    return { screenSpaceError: Math.max(bounds.min, current - 2), reason: "close-up refinement" };
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
 * Measures rendered frames and adapts quality to what the machine can actually do: 3D Tiles
 * screen-space error, canvas resolution scale, MSAA and globe detail. It only ever nudges
 * within the bounds of the chosen preset, and treats an idle scene as "no evidence" rather
 * than as a stall.
 */
/**
 * Gaussian splats are re-sorted on the CPU every camera change, so their cost grows with splat
 * count far faster than a mesh. This is the finest screen-space error a splat may use per preset.
 */
export function splatMinimumScreenSpaceError(preset: QualityPreset): number {
  return { performance: 12, balanced: 8, ultra: 4 }[preset];
}

/** Render profile: full quality for still frames, adaptive savings while the camera moves. */
export type RenderProfile = "rest" | "motion";

export class PerformanceManager {
  private readonly scene: Scene;
  private readonly frameTimestamps: number[] = [];
  private inputs: QualityInputs = {
    preset: "balanced",
    manualScreenSpaceError: null,
    adaptive: true,
  };
  private currentSse: number = QUALITY_SSE.balanced.base;
  /** Resolution scale and MSAA used while the camera moves; still frames always use 1 and 4. */
  private motionScale = 1;
  private motionMsaa = 4;
  private profile: RenderProfile = "rest";
  private resolutionScale = 1;
  private msaa = 4;
  private lowFpsSince: number | null = null;
  private goodFpsSince: number | null = null;
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
      // Switch profiles on real pose changes only: switching resolution resizes the drawing
      // buffer, which Cesium's moveStart counts as a camera change, and that would ping-pong
      // between the two profiles forever.
      viewer.camera.changed.addEventListener(() => {
        if (this.moving) return;
        this.moving = true;
        this.applyProfile("motion");
      }),
      viewer.camera.moveEnd.addEventListener(() => {
        if (!this.moving) return;
        this.moving = false;
        this.movingUntil = performance.now() + 400;
        this.applyProfile("rest");
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
    this.motionScale = 1;
    this.motionMsaa = inputs.preset === "performance" ? 1 : 4;
    this.scene.globe.maximumScreenSpaceError = inputs.preset === "performance" ? 3 : 2;
    this.applyProfile(this.moving ? "motion" : "rest", true);
    this.lowFpsSince = null;
    this.goodFpsSince = null;
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

  /**
   * A still frame is rendered once, so it can afford native device pixels, full resolution
   * scale and MSAA whatever the machine; the adaptive savings only apply while moving.
   */
  private applyProfile(profile: RenderProfile, force = false): void {
    if (!force && this.profile === profile) return;
    this.profile = profile;
    if (profile === "rest") {
      this.viewer.useBrowserRecommendedResolution = false;
      this.setResolutionScale(1);
      this.setMsaa(this.inputs.preset === "performance" ? 1 : 4);
    } else {
      this.viewer.useBrowserRecommendedResolution = this.inputs.preset !== "ultra";
      this.setResolutionScale(this.motionScale);
      this.setMsaa(this.motionMsaa);
    }
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

  private evaluate(forcedReason?: string): void {
    const fps = this.fps;
    const now = performance.now();
    const moving = this.moving || now < this.movingUntil;
    const loading = this.pending > 0 || this.processing > 0;
    const bounds = QUALITY_SSE[this.inputs.preset];
    const base = this.inputs.manualScreenSpaceError ?? bounds.base;
    const memory = this.memorySource();
    const memoryRatio = memory.budget > 0 ? memory.bytes / memory.budget : 0;
    let target = base;
    let reason: string;

    if (this.inputs.adaptive && this.inputs.manualScreenSpaceError === null) {
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
      reason = forcedReason ?? decision.reason;

      // Frame rate is only evidence about motion cost while the camera moves; frames rendered
      // at rest are tiles arriving, and those are allowed to be slow.
      if (moving && fps !== null && fps < LOW_FPS - 4) {
        this.goodFpsSince = null;
        this.lowFpsSince ??= now;
        if (now - this.lowFpsSince > SUSTAINED_LOW_MS) {
          // Cheapest wins first: drop anti-aliasing, then render fewer pixels.
          if (this.motionMsaa > 1) {
            this.motionMsaa = 1;
            reason = "low fps → MSAA off while moving";
          } else {
            this.motionScale = Math.max(0.5, this.motionScale - 0.1);
            reason = "low fps → lower resolution while moving";
          }
          this.applyProfile("motion", true);
          this.lowFpsSince = now;
        }
      } else if (moving && fps !== null && fps > STEADY_FPS) {
        this.lowFpsSince = null;
        this.goodFpsSince ??= now;
        if (now - this.goodFpsSince > SUSTAINED_LOW_MS) {
          if (this.motionScale < 1) this.motionScale = Math.min(1, this.motionScale + 0.1);
          else if (this.motionMsaa === 1 && this.inputs.preset !== "performance")
            this.motionMsaa = 4;
          this.applyProfile("motion", true);
          this.goodFpsSince = now;
        }
      } else if (!moving) {
        this.lowFpsSince = null;
        this.goodFpsSince = null;
      }
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
