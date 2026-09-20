import type { PostProcessStage, Scene, Viewer } from "cesium";

import type { Emitter } from "@/lib/emitter";
import { QUALITY_SSE, type QualityPreset } from "@/state/settings";

import { createColorGradeStage } from "./colorGrade";

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
/** Once the camera has rested this long, the still frame is re-rendered at full quality. */
const REST_SHARPEN_MS = 500;
/** A ladder step is judged on this many motion frames before and after it. */
const STEP_JUDGE_FRAMES = 40;
/** A step that did not raise the motion frame rate by this factor is reverted. */
const STEP_MIN_GAIN = 1.15;
/** After an ineffective step, that level is not tried again for this long. */
const STEP_BLOCK_MS = 90_000;
/** Per-frame main-thread budgets for texture, program and buffer uploads, moving and at rest. */
const MOVING_UPLOAD_BUDGETS_MS: [number, number, number] = [3, 4, 6];
// Rest frames load faster than moving ones but stay short enough that a click lands quickly.
const REST_UPLOAD_BUDGETS_MS: [number, number, number] = [8, 8, 16];
/** The still frame is sharpened once loading settles, or after this long regardless. */
const REST_SHARPEN_MAX_WAIT_MS = 3000;
/** Balanced renders at most this many device pixels per CSS pixel; ultra uses them all. */
const BALANCED_MAX_PIXEL_RATIO = 1.5;

/** Anti-aliasing for the still frame; motion always renders without MSAA. */
export function restMsaaFor(preset: QualityPreset): number {
  return preset === "performance" ? 1 : preset === "balanced" ? 2 : 4;
}

/**
 * Base resolution scale for a preset on a screen with `devicePixelRatio`: a 2× display
 * renders four times the pixels of a 1× one, which is where most of a frame goes on an
 * integrated GPU. Balanced caps the effective ratio; ultra keeps every device pixel;
 * performance renders at CSS pixels through `useBrowserRecommendedResolution`.
 */
export function baseResolutionScale(preset: QualityPreset, devicePixelRatio: number): number {
  if (preset !== "balanced") return 1;
  const ratio = Math.max(1, devicePixelRatio || 1);
  return Math.min(1, BALANCED_MAX_PIXEL_RATIO / ratio);
}
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
 * scene uses the idle time the way a maps app does: straight to the finest level, at any
 * height, as long as there is memory headroom. Slow frames at rest are tiles arriving, never a
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
  // A ladder step that raised the floor coarsens at once, moving or not: the step exists
  // because frames are slow, and the expensive tileset must feel it, not only the cheap one.
  if (current < bounds.min) return { screenSpaceError: bounds.min, reason: "ladder floor" };
  if (moving) return { screenSpaceError: current, reason: "moving (tiles held)" };
  if (loading) return { screenSpaceError: current, reason: "loading" };
  if (memoryRatio >= REFINE_MEMORY_RATIO)
    return {
      screenSpaceError: current,
      reason: `holding (${Math.round(memoryRatio * 100)}% of memory budget)`,
    };
  // Straight to the finest level, the way a maps app streams once the camera stops: the
  // intermediate levels would each be requested, decoded and thrown away on the way down.
  const target = Math.min(bounds.max, bounds.min);
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

/**
 * What a rendered frame is evidence of. `motion` is a frame the camera asked for, `animation`
 * one a continuous animation (the living-survey deformer) asked for with the camera still, and
 * `none` a frame nothing is driving — a tile arriving into an idle scene.
 */
export type FrameEvidence = "none" | "motion" | "animation";

/**
 * A moving camera wins: a gesture during an animation is still a gesture, and its frames are
 * the ones responsiveness is judged on.
 */
export function frameEvidence(sample: { moving: boolean; animating: boolean }): FrameEvidence {
  if (sample.moving) return "motion";
  return sample.animating ? "animation" : "none";
}

/**
 * How much of the ladder's evidence a frame contributes. The judgement here — animated frames
 * count against quality but never for it — is deliberate and asymmetric:
 *
 * The ladder exists to keep the scene at a rate the eye accepts, and until now only a gesture
 * could produce frames, so "moving" and "frames are being produced continuously" were the same
 * thing. An animation breaks that: the deformer's per-tick cost (displace, then upload a
 * texture region on the main thread) lands on every frame with a still camera, and a ladder
 * that only looked at motion would be blind to it and never step down however slow the scene
 * got. So animated frames do supply slow evidence, and a smooth animated frame pays that debt
 * back the way a smooth motion frame does — otherwise one hitch a minute would ratchet quality
 * down for good.
 *
 * They do not supply `recovery` (the credit that undoes a cut) and they are not `judged`
 * (the before/after sample that decides whether a step helped). Both of those compare frame
 * rates across time, and an animated frame and a motion frame measure different work: the
 * animation can be smooth on a machine that cannot pan, and a step judged on motion frames
 * before and animation frames after would report a gain or loss that is only the workload
 * changing. Recovery also raises cost, and raising cost on the strength of frames that are
 * cheap precisely because nothing is being asked of the machine is how an oscillation starts.
 * The asymmetry is conservative in the right direction: an animating scene can coarsen, and
 * earns its quality back on the next gesture.
 */
export interface FrameWeight {
  /** Feeds the slow/good accounting that steps the ladder down. */
  ladder: boolean;
  /** Feeds `goodMotionMs`, the credit that steps the ladder back up. */
  recovery: boolean;
  /** Feeds the benchmark, the frame budget and the step's before/after sample. */
  judged: boolean;
}

export function frameWeight(evidence: FrameEvidence): FrameWeight {
  switch (evidence) {
    case "motion":
      return { ladder: true, recovery: true, judged: true };
    case "animation":
      return { ladder: true, recovery: false, judged: false };
    case "none":
      return { ladder: false, recovery: false, judged: false };
  }
}

/**
 * Whether the still frame may be re-rendered at full resolution and the preset's MSAA. An
 * animating scene is not at rest in the sense this means — "nothing renders until something
 * changes" is what made the sharpened frame free, and an animation renders every tick — so
 * sharpening while animating would run every animated frame in the most expensive
 * configuration the machine offers, forever.
 */
export function shouldSharpenAtRest(sample: {
  moving: boolean;
  animating: boolean;
  restedMs: number;
  loading: boolean;
}): boolean {
  if (sample.moving || sample.animating) return false;
  if (sample.restedMs < REST_SHARPEN_MS) return false;
  // While tiles arrive every one of them re-renders the frame; sharpening then would make
  // each of those renders a full-quality one and the settle feel sluggish.
  return !sample.loading || sample.restedMs >= REST_SHARPEN_MAX_WAIT_MS;
}

/** "full" renders the preset as configured; "reduced" means the ladder has cut something. */
export type RenderProfile = "full" | "reduced";

/** Median main-thread cost of a moving frame, steady frames apart from tile-loading frames. */
export interface FrameBudget {
  updateMs: number | null;
  renderMs: number | null;
  commands: number | null;
  /** Update phase while tiles were being processed: decode and upload work, not steady cost. */
  loadingUpdateMs: number | null;
  loadingFrames: number;
  steadyFrames: number;
}

/** Receives the screen-space error in CSS pixels and the pixel ratio the scene renders at. */
export type ScreenSpaceErrorSink = (sse: number, pixelRatio: number) => void;

/**
 * Tileset groups are governed separately: each has its own error walk, loading state and
 * memory budget, so the Google world filling its cache never stops a survey mesh from
 * refining, and both are held to the same rules so collected data is never allowed less
 * detail than its surroundings.
 */
export type TilesetGroup = "sites" | "world";
const GROUPS: TilesetGroup[] = ["sites", "world"];

/** No tileset is ever asked for finer than this many device pixels of error. */
export const MIN_DEVICE_PX = 2;

/** Cesium measures screen-space error in CSS pixels; hand it device pixels, quantised. */
export function devicePixelError(sse: number, pixelRatio: number): number {
  return Math.max(MIN_DEVICE_PX, Math.round((sse / pixelRatio) * 4) / 4);
}

interface GroupState {
  sse: number;
  pending: number;
  processing: number;
  reason: string;
  readonly sinks: ScreenSpaceErrorSink[];
  readonly memorySources: (() => { bytes: number; budget: number })[];
}

function newGroup(sse: number): GroupState {
  return { sse, pending: 0, processing: 0, reason: "idle", sinks: [], memorySources: [] };
}

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
  private readonly groups: Record<TilesetGroup, GroupState> = {
    sites: newGroup(QUALITY_SSE.balanced.base),
    world: newGroup(QUALITY_SSE.balanced.base),
  };
  private ladder: LadderStep[] = [];
  private level = 0;
  private resolutionScale = 1;
  private baseScale = 1;
  private msaa = 4;
  private slowMotionMs = 0;
  private goodMotionMs = 0;
  private recoveryMs = INITIAL_RECOVERY_MS;
  private restSince = 0;
  /** True while the still frame is rendered at full quality rather than the ladder's step. */
  private sharpened = false;
  /** Main-thread time of the last frame's two phases: scene update (tileset and globe
   *  traversal, JavaScript) and render (command execution and GL submission). */
  private phaseStart = 0;
  private updateMs = 0;
  private renderMs = 0;
  private readonly motionUpdateMs: number[] = [];
  private readonly motionRenderMs: number[] = [];
  private readonly motionCommands: number[] = [];
  private readonly loadingUpdateMs: number[] = [];
  private readonly loadingRenderMs: number[] = [];
  private readonly loadingCommands: number[] = [];
  /** Motion frame intervals since the last ladder step, to judge whether the step helped. */
  private readonly judgeFrameMs: number[] = [];
  private judgeBaselineMs: number | null = null;
  private blockedAboveLevel: number | null = null;
  private blockedUntil = 0;
  private moving = false;
  private movingUntil = 0;
  /** True while something other than the camera renders every tick (the splat deformer). */
  private animating = false;
  private altitude = Number.POSITIVE_INFINITY;
  private nearSite = false;
  private readonly timer: ReturnType<typeof setInterval>;
  private readonly colorGrade: PostProcessStage;
  private readonly unsubscribe: (() => void)[] = [];
  private readonly gpu: string | null;
  private readonly webgl2: boolean;
  /** Frame intervals recorded while the camera moved, since the last reset (site change). */
  private readonly motionFrameMs: number[] = [];
  /** Timestamp of the last frame that carried evidence (motion or animation), for its interval. */
  private lastEvidenceFrameAt: number | null = null;

  constructor(
    private readonly viewer: Viewer,
    private readonly events: Emitter<SceneEvents>,
  ) {
    this.scene = viewer.scene;
    const info = readGpuInfo(viewer.canvas);
    this.gpu = info.renderer;
    this.webgl2 = info.webgl2;
    this.unsubscribe.push(
      this.scene.preUpdate.addEventListener(() => {
        this.phaseStart = performance.now();
      }),
      this.scene.postUpdate.addEventListener(() => {
        this.updateMs = performance.now() - this.phaseStart;
      }),
      this.scene.preRender.addEventListener(() => {
        this.phaseStart = performance.now();
      }),
      this.scene.postRender.addEventListener(() => {
        this.renderMs = performance.now() - this.phaseStart;
        this.onFrame();
      }),
      // Motion is tracked from real pose changes only (moveStart also fires on frustum and
      // canvas size changes). Nothing about the render settings changes here: a gesture
      // always runs with whatever the ladder settled on.
      viewer.camera.changed.addEventListener(() => {
        if (!this.moving) this.setUploadBudgets(true);
        this.moving = true;
        // The still frame may have been sharpened; the gesture runs at the ladder's step.
        if (this.sharpened) this.applyLevel();
      }),
      viewer.camera.moveEnd.addEventListener(() => {
        if (!this.moving) return;
        this.moving = false;
        this.movingUntil = performance.now() + 400;
        this.restSince = performance.now();
        this.setUploadBudgets(false);
      }),
    );
    this.colorGrade = createColorGradeStage();
    this.scene.postProcessStages.add(this.colorGrade);
    this.timer = setInterval(() => this.evaluate(), 500);
    this.events.emit("performance", {
      gpu: this.gpu,
      webgl2: this.webgl2,
      devicePixelRatio: window.devicePixelRatio,
    });
  }

  /**
   * Cesium uploads arriving textures, shader programs and buffers on the main thread inside
   * the frame, up to a per-frame budget per kind (10, 10 and 30 ms by default). During a
   * gesture that is most of a frame; a maps app keeps loading but never lets it stretch a
   * frame, so the budgets shrink while the camera moves and grow back at rest, where only
   * arriving tiles cause frames anyway. The scheduler is not public API; the fields are
   * stable at runtime and the change is ignored when they are not there.
   */
  private setUploadBudgets(moving: boolean): void {
    const scheduler = (
      this.scene as unknown as { _jobScheduler?: { _budgets?: { _total: number }[] } }
    )._jobScheduler;
    const budgets = scheduler?._budgets;
    if (!budgets || budgets.length < 3) return;
    const [texture, program, buffer] = moving ? MOVING_UPLOAD_BUDGETS_MS : REST_UPLOAD_BUDGETS_MS;
    if (budgets[0]) budgets[0]._total = texture;
    if (budgets[1]) budgets[1]._total = program;
    if (budgets[2]) budgets[2]._total = buffer;
  }

  /**
   * Registers a consumer of a group's current screen-space error. It receives the error in
   * CSS pixels together with the pixel ratio the scene renders at, so consumers can hand
   * Cesium an error in device pixels (`devicePixelError`): Cesium measures screen-space error
   * in CSS pixels, which on a HiDPI screen picks tiles twice as coarse as they look.
   */
  addScreenSpaceErrorSink(group: TilesetGroup, apply: ScreenSpaceErrorSink): void {
    this.groups[group].sinks.push(apply);
    apply(this.groups[group].sse, this.pixelRatio);
  }

  /**
   * The device pixel ratio the preset renders at. Deliberately not multiplied by the
   * ladder's resolution scale: a resolution cut saves fill and must not change which tiles
   * are drawn, or every gesture start on a slow machine would pop tiles.
   */
  get pixelRatio(): number {
    return this.viewer.useBrowserRecommendedResolution ? 1 : window.devicePixelRatio || 1;
  }

  private applySse(group: TilesetGroup): void {
    const ratio = this.pixelRatio;
    const state = this.groups[group];
    for (const sink of state.sinks) sink(state.sse, ratio);
  }

  /** Registers a group's memory use against its cache budget. */
  addMemorySource(group: TilesetGroup, source: () => { bytes: number; budget: number }): void {
    this.groups[group].memorySources.push(source);
  }

  /** A group's most loaded tileset's share of its budget; each tileset has its own cache. */
  private memoryRatio(group: TilesetGroup): number {
    let ratio = 0;
    for (const source of this.groups[group].memorySources) {
      const { bytes, budget } = source();
      if (budget > 0) ratio = Math.max(ratio, bytes / budget);
    }
    return ratio;
  }

  private get memoryBytes(): { bytes: number; budget: number } {
    let bytes = 0;
    let budget = 0;
    for (const group of GROUPS)
      for (const source of this.groups[group].memorySources) {
        const m = source();
        bytes += m.bytes;
        budget += m.budget;
      }
    return { bytes, budget };
  }

  configure(inputs: QualityInputs): void {
    this.inputs = inputs;
    const bounds = QUALITY_SSE[inputs.preset];
    // The colour grade is one cheap full-screen pass; performance skips it with the rest.
    this.colorGrade.enabled = inputs.preset !== "performance";
    for (const group of GROUPS) {
      this.groups[group].sse = inputs.manualScreenSpaceError ?? bounds.base;
      this.applySse(group);
    }
    this.scene.globe.maximumScreenSpaceError = inputs.preset === "performance" ? 3 : 2;
    // Still and moving frames share one resolution. Performance renders at the browser's
    // recommended (CSS pixel) resolution; the others use native device pixels until the
    // ladder proves the machine cannot keep up.
    this.viewer.useBrowserRecommendedResolution = inputs.preset === "performance";
    this.baseScale = baseResolutionScale(inputs.preset, window.devicePixelRatio || 1);
    this.ladder = buildLadder(inputs.preset);
    this.level = 0;
    this.slowMotionMs = 0;
    this.goodMotionMs = 0;
    this.recoveryMs = INITIAL_RECOVERY_MS;
    this.applyLevel();
    this.evaluate("configured");
  }

  /** Loading state per tileset group; a group's walk waits only for its own tiles. */
  reportLoading(group: TilesetGroup, pending: number, processing: number): void {
    this.groups[group].pending = pending;
    this.groups[group].processing = processing;
  }

  /**
   * Tells the manager that something other than the camera is rendering every tick, so that a
   * still camera no longer means a still scene. Call it with `true` before the first animated
   * frame and `false` once the animation stops; `false` is the default and leaves every
   * decision exactly as it was before animation existed. Turning it on drops a sharpened still
   * frame back to the ladder's step at once, the way the first frame of a gesture does.
   */
  setAnimating(animating: boolean): void {
    if (this.animating === animating) return;
    this.animating = animating;
    if (animating && this.sharpened) this.applyLevel();
  }

  reportContext(altitude: number, nearSite: boolean): void {
    this.altitude = altitude;
    this.nearSite = nearSite;
  }

  /** The sites' current screen-space error in CSS pixels. */
  get screenSpaceError(): number {
    return this.groups.sites.sse;
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
    this.sharpened = false;
    this.setResolutionScale(step.scale * this.baseScale);
    // Motion never pays for MSAA: the still frame gets it back once it settles.
    this.setMsaa(1);
    this.scene.requestRender();
  }

  /** Tiles still arriving: each one re-renders the still frame, so it stays at motion cost. */
  private get loading(): boolean {
    return GROUPS.some((g) => this.groups[g].processing > 0 || this.groups[g].pending > 0);
  }

  /**
   * At rest nothing renders until something changes, so the still frame can afford every
   * device pixel and the preset's anti-aliasing whatever the ladder or the preset's motion
   * scale says; the switch back happens on the first frame of the next gesture, one
   * framebuffer re-allocation. `shouldSharpenAtRest` decides when that is true.
   */
  private sharpenAtRest(): void {
    if (this.sharpened) return;
    const full = this.ladder[0];
    if (!full) return;
    this.sharpened = true;
    this.setResolutionScale(full.scale);
    this.setMsaa(full.msaa);
    this.scene.requestRender();
  }

  /**
   * Rendered frames per second over the last second, or null when the scene is idle. While
   * `animating` this is the rate the animation is running at, not how responsive the scene is
   * to a gesture — which is why the snapshot carries `animating` beside it.
   */
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
    const weight = frameWeight(frameEvidence({ moving: this.moving, animating: this.animating }));
    if (!weight.ladder) {
      this.lastEvidenceFrameAt = null;
      return;
    }
    if (weight.judged) {
      // Frames during which tiles were being processed carry decode and upload work in the
      // update phase; keep them apart so the steady per-frame cost of the scene is visible.
      const loading = GROUPS.some((g) => this.groups[g].processing > 0);
      const [updates, renders, commands] = loading
        ? [this.loadingUpdateMs, this.loadingRenderMs, this.loadingCommands]
        : [this.motionUpdateMs, this.motionRenderMs, this.motionCommands];
      updates.push(this.updateMs);
      renders.push(this.renderMs);
      commands.push(this.commandCount());
      for (const list of [updates, renders, commands]) if (list.length > 600) list.shift();
    }
    const last = this.lastEvidenceFrameAt;
    this.lastEvidenceFrameAt = now;
    if (last === null) return;
    const dt = now - last;
    if (!(dt > 0 && dt < 2000)) return;
    if (weight.judged) {
      this.motionFrameMs.push(dt);
      if (this.motionFrameMs.length > 600) this.motionFrameMs.shift();
    }
    // Evidence for the ladder comes from every frame that carries any, so three short
    // slow drags count as much as one long one.
    if (dt > 1000 / LOW_FPS) {
      this.slowMotionMs += dt;
      this.goodMotionMs = 0;
    } else {
      this.slowMotionMs = Math.max(0, this.slowMotionMs - dt);
      if (weight.recovery && dt < 1000 / STEADY_FPS) this.goodMotionMs += dt;
    }
    if (weight.judged) {
      this.judgeFrameMs.push(dt);
      if (this.judgeFrameMs.length > STEP_JUDGE_FRAMES) this.judgeFrameMs.shift();
    }
  }

  /** Starts a fresh benchmark window, e.g. when another site becomes active. */
  resetBenchmark(): void {
    this.motionFrameMs.length = 0;
    for (const list of [
      this.motionUpdateMs,
      this.motionRenderMs,
      this.motionCommands,
      this.loadingUpdateMs,
      this.loadingRenderMs,
      this.loadingCommands,
    ])
      list.length = 0;
    this.lastEvidenceFrameAt = null;
  }

  /** Draw commands issued in the last frame (not in the public typings, stable at runtime). */
  private commandCount(): number {
    const frameState = (this.scene as unknown as { frameState?: { commandList?: unknown[] } })
      .frameState;
    return frameState?.commandList?.length ?? 0;
  }

  /**
   * Where a moving frame's main-thread time goes, medians over the benchmark window: the
   * update phase is JavaScript (tileset traversal, globe quadtree), the render phase is
   * command execution and GL submission (one call per tile, uniforms set from JavaScript).
   * Together with the frame interval this says whether a machine is CPU-bound in Cesium's
   * own work or waiting on the GPU.
   */
  get frameBudget(): FrameBudget {
    const median = (list: number[]): number | null => {
      if (list.length < 5) return null;
      const sorted = [...list].sort((a, b) => a - b);
      return sorted[sorted.length >> 1] ?? null;
    };
    return {
      updateMs: median(this.motionUpdateMs),
      renderMs: median(this.motionRenderMs),
      commands: median(this.motionCommands),
      loadingUpdateMs: median(this.loadingUpdateMs),
      loadingFrames: this.loadingUpdateMs.length,
      steadyFrames: this.motionUpdateMs.length,
    };
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
    // Judge the last step once enough motion frames have been seen since it: a cut that
    // did not make motion faster costs quality for nothing (a tile cut on a fill-bound
    // machine, a resolution cut on a CPU-bound one) and is taken back.
    if (this.judgeBaselineMs !== null && this.judgeFrameMs.length >= STEP_JUDGE_FRAMES) {
      const after = this.judgeFrameMs.reduce((a, b) => a + b, 0) / this.judgeFrameMs.length;
      const gain = this.judgeBaselineMs / after;
      this.judgeBaselineMs = null;
      if (gain < STEP_MIN_GAIN && this.level > 0) {
        this.blockedAboveLevel = this.level - 1;
        this.blockedUntil = now + STEP_BLOCK_MS;
        this.level -= 1;
        this.slowMotionMs = 0;
        this.applyLevel();
        return `${this.ladder[this.level + 1]?.label ?? "step"} did not help → back to ${this.step.label}`;
      }
    }
    const blocked =
      this.blockedAboveLevel !== null &&
      now < this.blockedUntil &&
      this.level >= this.blockedAboveLevel;
    if (this.slowMotionMs >= SUSTAINED_LOW_MS && this.level < this.ladder.length - 1 && !blocked) {
      const frames = this.judgeFrameMs;
      this.judgeBaselineMs =
        frames.length >= 10 ? frames.reduce((a, b) => a + b, 0) / frames.length : null;
      this.judgeFrameMs.length = 0;
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
      this.judgeBaselineMs = null;
      this.recoveryMs = Math.min(MAX_RECOVERY_MS, this.recoveryMs * 2);
      this.applyLevel();
      return `smooth motion → ${this.step.label}`;
    }
    if (
      shouldSharpenAtRest({
        moving,
        animating: this.animating,
        restedMs: now - this.restSince,
        loading: this.loading,
      })
    )
      this.sharpenAtRest();
    return null;
  }

  private evaluate(forcedReason?: string): void {
    const fps = this.fps;
    const now = performance.now();
    const moving = this.moving || now < this.movingUntil;
    const preset = QUALITY_SSE[this.inputs.preset];
    const memory = this.memoryBytes;
    const adaptive = this.inputs.adaptive && this.inputs.manualScreenSpaceError === null;
    const stepReason = adaptive ? this.climbLadder(now, moving) : null;
    // The ladder's tile penalty is a motion measure: nothing renders at rest, so the still
    // frame refines to the preset minimum whatever the ladder says, and a gesture starts by
    // coarsening to the floor (a step that does not help motion is reverted above).
    const penalty = moving ? this.step.ssePenalty : 0;
    const bounds = {
      base: Math.min(preset.max, preset.base + penalty),
      min: Math.min(preset.max, preset.min + penalty),
      max: preset.max,
    };

    let pending = 0;
    let processing = 0;
    for (const group of GROUPS) {
      const state = this.groups[group];
      pending += state.pending;
      processing += state.processing;
      let target = this.inputs.manualScreenSpaceError ?? preset.base;
      if (adaptive) {
        const decision = decideScreenSpaceError({
          bounds,
          current: state.sse,
          moving,
          loading: state.pending > 0 || state.processing > 0,
          memoryRatio: this.memoryRatio(group),
        });
        target = decision.screenSpaceError;
        state.reason = forcedReason ?? stepReason ?? decision.reason;
      } else {
        state.reason =
          forcedReason ?? (this.inputs.manualScreenSpaceError !== null ? "manual" : "fixed");
      }
      if (Math.abs(target - state.sse) >= 0.5) {
        state.sse = Math.round(target * 2) / 2;
        this.applySse(group);
      }
    }

    const reason =
      this.groups.sites.reason === this.groups.world.reason
        ? this.groups.sites.reason
        : `sites: ${this.groups.sites.reason} · world: ${this.groups.world.reason}`;
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
      pendingRequests: pending,
      tilesProcessing: processing,
      siteScreenSpaceError: this.groups.sites.sse,
      worldScreenSpaceError: this.groups.world.sse,
      adaptiveReason: reason,
      moving,
      animating: this.animating,
      tilesetMemoryMb: Math.round(memory.bytes / 1048576),
      memoryBudgetMb: Math.round(memory.budget / 1048576),
      benchmark: this.benchmark,
      frameBudget: this.frameBudget,
    });
  }

  destroy(): void {
    clearInterval(this.timer);
    if (!this.scene.isDestroyed()) this.scene.postProcessStages.remove(this.colorGrade);
    for (const off of this.unsubscribe) off();
  }
}

/**
 * Cheapest savings first: anti-aliasing, then resolution in three steps down to half, then
 * coarser tiles (the only step that changes what is drawn). Exported for the unit tests.
 */
export function buildLadder(preset: QualityPreset): LadderStep[] {
  // `msaa` is the still frame's anti-aliasing; motion frames never use MSAA at any step.
  const steps: LadderStep[] = [
    { msaa: restMsaaFor(preset), scale: 1, ssePenalty: 0, label: "full" },
  ];
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
