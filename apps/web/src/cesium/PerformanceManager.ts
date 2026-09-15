import type { Scene, Viewer } from "cesium";

import type { Emitter } from "@/lib/emitter";
import { QUALITY_SSE, type QualityPreset } from "@/state/settings";

import type { SceneEvents } from "./types";

export interface QualityInputs {
  preset: QualityPreset;
  manualScreenSpaceError: number | null;
  adaptive: boolean;
}

export interface AdaptiveDecision {
  screenSpaceError: number;
  resolutionScale: number;
  reason: string;
}

/**
 * Measures frame timing and adapts 3D Tiles quality to the observed conditions:
 * frame rate, device pixel ratio, loading state, camera distance and motion.
 * It only ever nudges within the bounds of the chosen preset.
 */
export class PerformanceManager {
  private readonly scene: Scene;
  private readonly frameTimes: number[] = [];
  private lastFrameAt = performance.now();
  private inputs: QualityInputs = {
    preset: "balanced",
    manualScreenSpaceError: null,
    adaptive: true,
  };
  private currentSse: number = QUALITY_SSE.balanced.base;
  private resolutionScale = 1;
  private lowFpsSince: number | null = null;
  private pending = 0;
  private processing = 0;
  private moving = false;
  private movingUntil = 0;
  private altitude = Number.POSITIVE_INFINITY;
  private nearSite = false;
  private readonly timer: ReturnType<typeof setInterval>;
  private readonly unsubscribe: (() => void)[] = [];
  private applySse: (sse: number) => void = () => undefined;
  private readonly gpu: string | null;
  private readonly webgl2: boolean;

  constructor(
    private readonly viewer: Viewer,
    private readonly events: Emitter<SceneEvents>,
  ) {
    this.scene = viewer.scene;
    const info = readGpuInfo(viewer.scene);
    this.gpu = info.renderer;
    this.webgl2 = info.webgl2;
    this.unsubscribe.push(
      this.scene.postRender.addEventListener(() => this.onFrame()),
      viewer.camera.moveStart.addEventListener(() => {
        this.moving = true;
      }),
      viewer.camera.moveEnd.addEventListener(() => {
        this.moving = false;
        this.movingUntil = performance.now() + 400;
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

  configure(inputs: QualityInputs): void {
    this.inputs = inputs;
    const bounds = QUALITY_SSE[inputs.preset];
    this.currentSse = inputs.manualScreenSpaceError ?? bounds.base;
    this.applySse(this.currentSse);
    this.viewer.useBrowserRecommendedResolution = inputs.preset !== "ultra";
    this.setResolutionScale(1);
    this.scene.postProcessStages.fxaa.enabled = inputs.preset !== "performance";
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

  get fps(): number {
    if (this.frameTimes.length === 0) return 0;
    const avg = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
    return avg > 0 ? 1000 / avg : 0;
  }

  private onFrame(): void {
    const now = performance.now();
    const dt = now - this.lastFrameAt;
    this.lastFrameAt = now;
    if (dt > 0 && dt < 2000) {
      this.frameTimes.push(dt);
      if (this.frameTimes.length > 45) this.frameTimes.shift();
    }
  }

  private setResolutionScale(scale: number): void {
    if (Math.abs(this.resolutionScale - scale) < 0.01) return;
    this.resolutionScale = scale;
    this.viewer.resolutionScale = scale;
  }

  private evaluate(forcedReason?: string): void {
    const fps = this.fps;
    const now = performance.now();
    const moving = this.moving || now < this.movingUntil;
    const loading = this.pending > 0 || this.processing > 0;
    const bounds = QUALITY_SSE[this.inputs.preset];
    const base = this.inputs.manualScreenSpaceError ?? bounds.base;
    let target = base;
    let reason = forcedReason ?? "steady";

    if (this.inputs.adaptive && this.inputs.manualScreenSpaceError === null) {
      if (moving) {
        // Fast motion: coarser tiles keep the frame rate smooth while the view settles.
        target = Math.min(bounds.max, base + Math.max(4, base * 0.5));
        reason = "moving";
      } else if (fps > 0 && fps < 28) {
        target = Math.min(bounds.max, this.currentSse + 3);
        reason = `low fps (${fps.toFixed(0)})`;
      } else if (!loading && fps > 52 && this.nearSite && this.altitude < 250) {
        // Stationary close-up inspection with headroom: refine towards the finest level.
        target = Math.max(bounds.min, this.currentSse - 2);
        reason = "close-up refinement";
      } else if (!loading && fps > 50) {
        target = base;
        reason = "steady";
      } else {
        target = this.currentSse;
        reason = loading ? "loading" : "steady";
      }
      if (fps > 0 && fps < 22) {
        this.lowFpsSince ??= now;
        if (now - this.lowFpsSince > 2000) {
          this.setResolutionScale(Math.max(0.66, this.resolutionScale - 0.1));
          reason = "low fps → lower resolution";
        }
      } else {
        this.lowFpsSince = null;
        if (fps > 50 && this.resolutionScale < 1)
          this.setResolutionScale(Math.min(1, this.resolutionScale + 0.1));
      }
    }

    if (Math.abs(target - this.currentSse) >= 0.5) {
      this.currentSse = Math.round(target * 2) / 2;
      this.applySse(this.currentSse);
    }

    this.events.emit("performance", {
      fps: Math.round(fps),
      frameTimeMs: fps > 0 ? Math.round((1000 / fps) * 10) / 10 : 0,
      resolutionScale: this.resolutionScale,
      devicePixelRatio: window.devicePixelRatio,
      pendingRequests: this.pending,
      tilesProcessing: this.processing,
      siteScreenSpaceError: this.currentSse,
      adaptiveReason: reason,
      moving,
    });
  }

  destroy(): void {
    clearInterval(this.timer);
    for (const off of this.unsubscribe) off();
  }
}

function readGpuInfo(scene: Scene): { renderer: string | null; webgl2: boolean } {
  try {
    // Scene.context is not in the public typings but is the context Cesium renders with.
    const context = (
      scene as unknown as { context?: { webgl2?: boolean; _gl?: WebGLRenderingContext } }
    ).context;
    const gl = context?._gl;
    if (!gl) return { renderer: null, webgl2: Boolean(context?.webgl2) };
    const ext = gl.getExtension("WEBGL_debug_renderer_info") as {
      UNMASKED_RENDERER_WEBGL: number;
    } | null;
    const renderer = ext
      ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL))
      : String(gl.getParameter(gl.RENDERER));
    return { renderer, webgl2: Boolean(context?.webgl2) };
  } catch {
    return { renderer: null, webgl2: false };
  }
}
