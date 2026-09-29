/**
 * The Living Survey: measured trees that move under a wind setting, without their measurement
 * ever changing.
 *
 * This is the piece that turns `SplatDeformer` into a feature. It owns the wind, finds the
 * loaded splat tilesets that have a motion rig, attaches a deformer to each, and drives them
 * from the scene clock — and it owns the two properties that decide whether any of that is
 * honest:
 *
 * **Time is scene time.** `t` comes from `viewer.clock.currentTime` against a fixed epoch, never
 * from `performance.now()`. The frame a person sees is therefore a pure function of the scene's
 * own clock: freeze the clock, step it to an exact value, and the same splats land in the same
 * places every time. That is what makes this testable at all, and it is why the epoch is a
 * constant rather than "when the manager was constructed".
 *
 * **Idle stays idle.** A scene in `requestRenderMode` renders only when someone asks. This asks
 * while any deformer holds displaced positions, plus the single frame where the last one lets go
 * — the restore. Nothing is asked for at rest, because the deformer writes nothing at rest:
 * `markMovingNodes` compares against the identity transform *exactly*, and `@twin/world`
 * guarantees bit-exact identity at zero wind. A tolerance anywhere upstream would quietly turn a
 * still survey into a scene that renders forever.
 *
 * What it deliberately does not do: wrap `tileset.update` (`GaussianSplatPrimitive` already owns
 * that slot), touch canonical positions (see `SplatDeformer`), or claim the wind is a speed
 * (see `WindStrength` in `@twin/world`).
 */

import { JulianDate, type Viewer } from "cesium";

import {
  createLivingMotion,
  deform,
  isForestRig,
  flutterField,
  livingFrame,
  livingMaxDisplacement,
  livingWindFromSettings,
  maxDisplacement,
  parseMotionSidecar,
  parsePlantBinding,
  parseRig,
  prepareLivingMotion,
  sortStaleness,
  WIND_CALM,
  type FlutterField,
  type LivingMotion,
  type MotionRig,
  type NodeTransform,
  type PlantBinding,
  type WindSettings,
} from "@twin/world";

import type { Emitter } from "@/lib/emitter";
import { createLogger, describeError } from "@/lib/log";
import {
  REFERENCE_GAUSSIAN_SCALE_M,
  type LivingCpuReason,
  type LivingSiteStatus,
  type LivingSurveyStatus,
} from "@/state/living";

import { rigUrlFor } from "./livingRigs";
import type { PerformanceManager } from "./PerformanceManager";
import type { SiteManager } from "./SiteManager";
import type { MotionTextureFactory } from "./splatGpuMotion";
import { cesiumMotionTextures } from "./splatGpuTextures";
import { splatTilesetOf } from "./splatInternals";
import {
  SplatDeformer,
  type DeformerMotion,
  type DeformerReason,
  type DeformerStatus,
} from "./SplatDeformer";
import type { SceneEvents } from "./types";

const log = createLogger("living-survey");

/**
 * The instant `t = 0` refers to. A fixed calendar date, not construction time, so the animation
 * is reproducible across sessions and across machines; recent, so `t` stays small enough that
 * the noise lattice keeps far more resolution than a frame interval needs.
 */
export const LIVING_EPOCH_ISO = "2026-01-01T00:00:00Z";

const EPOCH = JulianDate.fromIso8601(LIVING_EPOCH_ISO);

/** Seconds of scene time since {@link LIVING_EPOCH_ISO}. */
export function sceneSeconds(currentTime: JulianDate): number {
  const seconds = JulianDate.secondsDifference(currentTime, EPOCH);
  return Number.isFinite(seconds) ? seconds : 0;
}

/**
 * Consecutive ticks at an unchanged scene time that mean the clock has stopped, not that the
 * wind is calm.
 *
 * Two seconds at 60 Hz. The guard exists because a frozen `viewer.clock.currentTime` and a
 * motion model with no motion in it look *identical* on screen — a tree that bends once and then
 * sits there — and the two were confused once already. The model was the culprit; this makes
 * the other candidate say so out loud rather than leaving it to be guessed at again.
 */
const STALLED_CLOCK_TICKS = 120;

/**
 * Why a refusal is permanent, in words a person can act on. Only `refused` phases appear here:
 * the `waiting` reasons (`no-primitive`, `no-snapshot`, `no-texture`, `no-bake-transform`,
 * `no-capture`) are the ordinary first few frames after a tile loads and must never toast.
 */
const REFUSAL_BODY: Readonly<Record<DeformerReason, string | null>> = {
  layout:
    "The splat texture's addressing parameters are not self-consistent on this device, so the tree would be written into the wrong texels.",
  frame:
    "This capture's frame is not east-north-up, so the motion model cannot tell which way is up.",
  upright: "These points do not read as a standing tree, so the rig does not describe them.",
  checksum: "These are not the splats the motion rig was built for.",
  // Also a waiting reason: a level-of-detail snapshot mid-rebuild. Never a toast.
  tiles: null,
  bake: "The capture's placement could not be undone exactly, so the measured pose could not be guaranteed.",
  binding:
    "This capture's plants came without the binding that keeps everything else still, so nothing moves.",
  internal: "The deformer hit an unexpected error and stopped.",
  // Waiting reasons. Normal for the first frames after load; never a toast.
  "no-primitive": null,
  "no-snapshot": null,
  "no-texture": null,
  "no-bake-transform": null,
  "no-capture": null,
};

export interface LivingSurveyOptions {
  /**
   * Whether this build lets motion run in the splat vertex shader (the engine patch's hook,
   * `splatGpuMotion.ts`) at all. Default true; `VITE_SPLAT_GPU_MOTION=0` turns it off for every
   * viewer. Where it is allowed, the viewer's "Motion on GPU" setting decides
   * ({@link LivingSurveyManager.setGpuMotion}), and the CPU path stays the fallback for an
   * unpatched engine or a snapshot the shader cannot take.
   */
  readonly gpuMotion?: boolean;
  /**
   * The GPU path's texture factory. Defaults to CesiumJS's own (`cesiumMotionTextures`);
   * `null` means "unavailable". A seam for tests.
   */
  readonly motionTextures?: MotionTextureFactory | null;
}

/** What decides whether a site may use the GPU path, before the deformer has its own say. */
export interface GpuMotionGate {
  /** The build allows it (`VITE_SPLAT_GPU_MOTION`). */
  readonly allowed: boolean;
  /** The viewer wants it (the "Motion on GPU" setting). */
  readonly wanted: boolean;
  /** This CesiumJS build exports what the texture factory needs. */
  readonly available: boolean;
}

/**
 * Which path a site is on and, when it is the CPU, the first reason why — the build, then the
 * viewer, then the engine, then the snapshot. Pure, so the fallback order is testable.
 */
export function livingMotionPath(
  status: Pick<DeformerStatus, "motion" | "cpuReason">,
  gate: GpuMotionGate,
): { motionPath: DeformerMotion; cpuReason: LivingCpuReason | null } {
  if (status.motion === "gpu") return { motionPath: "gpu", cpuReason: null };
  const cpuReason: LivingCpuReason = !gate.allowed
    ? "build"
    : !gate.wanted
      ? "switched-off"
      : !gate.available || status.cpuReason !== "mixed-bake"
        ? "engine"
        : "mixed-bake";
  return { motionPath: "cpu", cpuReason };
}

/** Animated frames the motion cost is averaged over: about a second at 60 Hz. */
export const MOTION_COST_WINDOW = 60;

/**
 * How often the cost readout is republished, milliseconds. The cost changes every frame, and
 * the `living` event must not: it drives React, and a per-frame store update would cost more
 * than the GPU path it is measuring.
 */
const MOTION_COST_PUBLISH_MS = 500;

/** A mean over the last `size` samples. */
export class RollingMean {
  readonly #values: Float64Array;
  #count = 0;
  #next = 0;
  #sum = 0;

  constructor(size: number) {
    this.#values = new Float64Array(Math.max(1, size));
  }

  push(value: number): void {
    const size = this.#values.length;
    if (this.#count === size) this.#sum -= this.#values[this.#next] ?? 0;
    else this.#count += 1;
    this.#values[this.#next] = value;
    this.#sum += value;
    this.#next = (this.#next + 1) % size;
  }

  /** The mean, or null before the first sample. */
  get mean(): number | null {
    return this.#count === 0 ? null : this.#sum / this.#count;
  }

  clear(): void {
    this.#count = 0;
    this.#next = 0;
    this.#sum = 0;
  }
}

/** Rounded for display and for change detection: hundredths of a millisecond. */
function roundMs(value: number | null): number | null {
  return value === null ? null : Math.round(value * 100) / 100;
}

/** Everything held for one deformed asset. Dropped whole when its site unloads. */
interface LivingEntry {
  readonly siteId: string;
  readonly siteSlug: string;
  readonly assetId: string;
  readonly rig: MotionRig;
  /**
   * Living Mode's modal model, when the rig points at a motion sidecar (ADR 0008). Absent, the
   * rig moves under the legacy nine-sine model in `deform`.
   */
  readonly motion: LivingMotion | undefined;
  readonly deformer: SplatDeformer;
  status: DeformerStatus;
  /** Model plus write, per animated frame. */
  readonly motionCost: RollingMean;
  /** The write alone. */
  readonly applyCost: RollingMean;
  /** What was last published of the two, so the event fires only when the readout changes. */
  motionMs: number | null;
  applyMs: number | null;
}

/**
 * Which motion model drives rigs that carry a sidecar. `auto` uses the sidecar; `legacy` forces
 * the old nine-sine model everywhere — only for side-by-side comparison (`e2e/livingCompare`).
 */
export type LivingMotionModel = "auto" | "legacy";

export class LivingSurveyManager {
  readonly #viewer: Viewer;
  readonly #events: Emitter<SceneEvents>;
  readonly #sites: SiteManager;
  readonly #performance: PerformanceManager;
  readonly #entries = new Map<string, LivingEntry>();
  /** Assets whose rig is being fetched, so a burst of asset events starts one fetch. */
  readonly #pending = new Set<string>();
  /** Assets that refused, or whose rig could not be loaded. Never retried for this session. */
  readonly #declined = new Set<string>();
  readonly #unsubscribe: (() => void)[] = [];
  #wind: WindSettings = WIND_CALM;
  #model: LivingMotionModel = "auto";
  /** Counted holds that pin every tree at its measured pose (see {@link holdMeasuredPose}). */
  #holds = 0;
  #removeTick: (() => void) | null = null;
  /** Whether the previous tick left anything displaced — the restore frame needs one render. */
  #wasDisplaced = false;
  /** Scene time at the previous tick, for {@link STALLED_CLOCK_TICKS}. */
  #lastSceneSeconds: number | null = null;
  /** Consecutive ticks whose scene time was identical to the one before. */
  #stalledTicks = 0;
  /** The stall has been reported once; it is a standing condition, not a per-frame event. */
  #reportedStall = false;
  #published: LivingSurveyStatus | null = null;
  #destroyed = false;
  /** The GPU motion path's texture factory, when this CesiumJS build can make one. */
  readonly #gpu: MotionTextureFactory | undefined;
  /** The build allows the GPU path (`VITE_SPLAT_GPU_MOTION`). */
  readonly #gpuAllowed: boolean;
  /** The viewer wants it (the persisted "Motion on GPU" setting). On unless switched off. */
  #gpuWanted = true;
  /** `performance.now()` after which the cost readout may next be republished. */
  #nextCostPublish = 0;

  constructor(
    viewer: Viewer,
    events: Emitter<SceneEvents>,
    sites: SiteManager,
    performance: PerformanceManager,
    options: LivingSurveyOptions = {},
  ) {
    this.#viewer = viewer;
    this.#events = events;
    this.#sites = sites;
    this.#performance = performance;
    // The GPU path by default: its per-frame main-thread cost is per rig node, where the CPU
    // path's is per splat — about 70 ms a frame for a million-splat tree. The CPU path stays
    // the fallback for an unpatched engine, a snapshot the shader cannot take, a build that
    // forces it, and a viewer who switches the GPU path off to compare.
    this.#gpuAllowed = options.gpuMotion !== false;
    this.#gpu =
      options.motionTextures === undefined
        ? cesiumMotionTextures()
        : (options.motionTextures ?? undefined);
    if (this.#gpuAllowed && this.#gpu === undefined) {
      log.warn("GPU splat motion unavailable in this CesiumJS build; using CPU");
    }
    this.#unsubscribe.push(
      // Attach and detach timing, from SiteManager's own per-asset events: `attachTileset`
      // reports "ready", `disposeHandle` reports "idle" as a distant site is unloaded by
      // `checkProximity`. Progress updates carry no `loadState` and are ignored.
      events.on("asset", ({ patch }) => {
        if (patch.loadState !== undefined) this.reconcile();
      }),
    );
    this.reconcile();
  }

  /** The wind the scene is running. Already forced to calm by whoever handles reduced motion. */
  get wind(): WindSettings {
    return this.#wind;
  }

  /** What every attached site is doing. The same value the `living` event carries. */
  get status(): LivingSurveyStatus {
    return this.#snapshot();
  }

  /**
   * Sets the wind.
   *
   * `performance.setAnimating` is called from here rather than from the tick: whether the scene
   * animates is a property of the wind and of how many rigged sites are loaded, both of which
   * change a handful of times in a session, and a per-frame call would make the performance
   * ladder's notion of "animating" a per-frame guess.
   */
  setWind(wind: WindSettings): void {
    if (this.#destroyed) return;
    if (wind.strength === this.#wind.strength && wind.bearingDeg === this.#wind.bearingDeg) return;
    this.#wind = wind;
    this.#refreshAnimating();
    this.#publish();
    // Dropping to calm must still reach the GPU: the deformer owes one restoring write, and
    // without a frame to do it in the tree would stay bent at whatever the last gust left.
    if (this.#entries.size > 0) this.#viewer.scene.requestRender();
  }

  /** What decides whether a site may use the GPU path. */
  get gpuMotion(): GpuMotionGate {
    return {
      allowed: this.#gpuAllowed,
      wanted: this.#gpuWanted,
      available: this.#gpu !== undefined,
    };
  }

  /**
   * The viewer's "Motion on GPU" setting: `false` puts every site on the CPU path, for an A/B
   * comparison or on hardware where the shader path misbehaves.
   *
   * Every deformer switches in place (`SplatDeformer.setGpu`): it first restores the measured
   * pose through the path it is leaving — the CPU path's restoring write, or the GPU hook
   * uninstalled — so nothing displaced is left behind, then re-derives on the new path at the
   * next tick. The cost readout starts over, so it never averages the two paths together.
   */
  setGpuMotion(enabled: boolean): void {
    if (this.#destroyed || enabled === this.#gpuWanted) return;
    this.#gpuWanted = enabled;
    const factory = this.#factory();
    for (const entry of this.#entries.values()) {
      entry.deformer.setGpu(factory);
      entry.status = entry.deformer.status;
      entry.motionCost.clear();
      entry.applyCost.clear();
      entry.motionMs = null;
      entry.applyMs = null;
    }
    log.info("motion path switched", { gpu: factory !== undefined });
    this.#publish();
    // The frame that shows the restore, and the first frame on the new path.
    if (this.#entries.size > 0) this.#viewer.scene.requestRender();
  }

  /** The texture factory deformers get: the GPU path when build, viewer and engine all allow. */
  #factory(): MotionTextureFactory | undefined {
    return this.#gpuAllowed && this.#gpuWanted ? this.#gpu : undefined;
  }

  /** Chooses the motion model; see {@link LivingMotionModel}. */
  setMotionModel(model: LivingMotionModel): void {
    if (this.#destroyed || model === this.#model) return;
    this.#model = model;
    this.#publish();
    if (this.#entries.size > 0) this.#viewer.scene.requestRender();
  }

  /**
   * Pins every deformed tree at its measured pose until the returned release is called.
   *
   * This is how `CesiumSceneManager.snapshot()` photographs the survey rather than the
   * simulation. No separate code path is needed on the deformer's side: applying the rig at
   * zero wind restores the exact measured bytes in one frame, because every frame is computed
   * from the canonical positions rather than from the previous frame. Counted, so overlapping
   * holds compose, and releasing twice is harmless.
   */
  holdMeasuredPose(): () => void {
    if (this.#destroyed) return () => undefined;
    this.#holds += 1;
    let released = false;
    return () => {
      if (released || this.#destroyed) return;
      released = true;
      this.#holds = Math.max(0, this.#holds - 1);
    };
  }

  /**
   * Brings the attached deformers into line with the tilesets actually in the scene.
   *
   * Reconciling against `sites.loadedAssets()` rather than following individual events means
   * there is no bookkeeping to get out of step: an asset that is gone from the list has had its
   * tileset destroyed, and its deformer goes with it in the same turn — so nothing can ever
   * write into freed GL objects.
   */
  reconcile(): void {
    if (this.#destroyed) return;
    const live = new Map<string, { slug: string; siteId: string; rigUrl: string }>();
    for (const asset of this.#sites.loadedAssets()) {
      const rigUrl = rigUrlFor(asset.rigPath, asset.representation, asset.sourceUrl);
      if (rigUrl === null) continue;
      live.set(asset.assetId, { slug: asset.siteSlug, siteId: asset.siteId, rigUrl });
    }

    let changed = false;
    for (const [assetId, entry] of this.#entries) {
      if (live.has(assetId)) continue;
      entry.deformer.destroy();
      this.#entries.delete(assetId);
      this.#declined.delete(assetId);
      changed = true;
      log.info("site left the scene; deformer dropped", { site: entry.siteSlug });
    }
    for (const [assetId, candidate] of live) {
      if (this.#entries.has(assetId) || this.#pending.has(assetId)) continue;
      if (this.#declined.has(assetId)) continue;
      this.#pending.add(assetId);
      void this.#attach(assetId, candidate);
    }
    if (changed) this.#afterEntriesChanged();
  }

  destroy(): void {
    if (this.#destroyed) return;
    this.#destroyed = true;
    for (const off of this.#unsubscribe) off();
    this.#unsubscribe.length = 0;
    this.#removeTick?.();
    this.#removeTick = null;
    for (const entry of this.#entries.values()) entry.deformer.destroy();
    this.#entries.clear();
    this.#pending.clear();
    this.#declined.clear();
    this.#holds = 0;
    this.#wasDisplaced = false;
    this.#lastSceneSeconds = null;
    this.#stalledTicks = 0;
    this.#reportedStall = false;
    this.#performance.setAnimating(false);
  }

  async #attach(
    assetId: string,
    candidate: { slug: string; siteId: string; rigUrl: string },
  ): Promise<void> {
    let rig: MotionRig;
    try {
      const response = await fetch(candidate.rigUrl);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      rig = parseRig(await response.text());
    } catch (error) {
      this.#pending.delete(assetId);
      this.#declined.add(assetId);
      log.warn("motion rig could not be loaded", {
        site: candidate.slug,
        url: candidate.rigUrl,
        error: describeError(error),
      });
      return;
    }
    const motion = await this.#loadMotion(rig, candidate);
    const plantBinding = await this.#loadBinding(rig, candidate);
    this.#pending.delete(assetId);
    if (this.#destroyed) return;
    // The site may have unloaded while the rig was in flight.
    const tileset = this.#sites.tilesetFor(candidate.siteId, "gaussian-splat");
    if (tileset === null || this.#entries.has(assetId)) return;

    // A rigged site draws from the aggregated snapshot the deformer was built and verified
    // on (the CPU path rewrites it from a captured packed buffer, and the measured-bytes
    // checks read that capture); incremental slots are for scans that stand still.
    this.#aggregate(candidate.siteId);
    const deformer = new SplatDeformer({
      tileset: splatTilesetOf(tileset),
      rig,
      gpu: this.#factory(),
      ...(plantBinding === undefined ? {} : { plantBinding }),
    });
    this.#entries.set(assetId, {
      siteId: candidate.siteId,
      siteSlug: candidate.slug,
      assetId,
      rig,
      motion,
      deformer,
      status: deformer.status,
      motionCost: new RollingMean(MOTION_COST_WINDOW),
      applyCost: new RollingMean(MOTION_COST_WINDOW),
      motionMs: null,
      applyMs: null,
    });
    log.info("motion rig attached", {
      site: candidate.slug,
      nodes: rig.nodes.length,
      plants: rig.plants?.length ?? 1,
      motionEvidence: motion?.sidecar.motionEvidence ?? "legacy",
    });
    this.#afterEntriesChanged();
  }

  /**
   * The rig's motion sidecar, when it points at one. A sidecar that is missing or does not fit
   * the rig is not a refusal — the tree still moves, under the legacy model — but it is said.
   */
  async #loadMotion(
    rig: MotionRig,
    candidate: { slug: string; rigUrl: string },
  ): Promise<LivingMotion | undefined> {
    if (rig.motionPath === undefined) return undefined;
    const url = new URL(rig.motionPath, candidate.rigUrl).toString();
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const motion = createLivingMotion(rig, parseMotionSidecar(await response.text(), rig));
      // Builds the motion textures now (~0.5 s each, once per session), not on the first frame.
      prepareLivingMotion(motion);
      return motion;
    } catch (error) {
      log.warn("motion sidecar could not be loaded; using the legacy model", {
        site: candidate.slug,
        url,
        error: describeError(error),
      });
      return undefined;
    }
  }

  /**
   * A forest rig's plant binding (`plants.json`), when the rig points at one. Unlike a missing
   * sidecar this has no fallback: without it a wall would be skinned to the crown beside it,
   * so the deformer refuses the rig (reason `binding`) and nothing moves.
   */
  async #loadBinding(
    rig: MotionRig,
    candidate: { slug: string; rigUrl: string },
  ): Promise<PlantBinding | undefined> {
    if (!isForestRig(rig) || rig.bindingPath === undefined) return undefined;
    const url = new URL(rig.bindingPath, candidate.rigUrl).toString();
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return parsePlantBinding(await response.text(), rig);
    } catch (error) {
      log.warn("plant binding could not be loaded; the forest stays still", {
        site: candidate.slug,
        url,
        error: describeError(error),
      });
      return undefined;
    }
  }

  /** Whether an entry runs Living Mode's modal model this frame. */
  #living(entry: LivingEntry): LivingMotion | undefined {
    return this.#model === "auto" ? entry.motion : undefined;
  }

  /** One frame of one rig: Living Mode when it has a sidecar, the legacy model otherwise. */
  #frame(
    entry: LivingEntry,
    t: number,
    wind: WindSettings,
  ): { transforms: NodeTransform[]; flutter: FlutterField } {
    const motion = this.#living(entry);
    if (motion !== undefined) {
      return livingFrame(motion, t, livingWindFromSettings(wind, motion.sidecar));
    }
    return { transforms: deform(entry.rig, t, wind), flutter: flutterField(entry.rig, t, wind) };
  }

  /** The proven worst-case splat displacement for an entry at a wind, metres. */
  #maxDisplacement(entry: LivingEntry, wind: WindSettings): number {
    const motion = this.#living(entry);
    if (motion !== undefined) {
      return livingMaxDisplacement(motion, livingWindFromSettings(wind, motion.sidecar));
    }
    return maxDisplacement(entry.rig, wind);
  }

  /** Starts or stops the tick, re-decides `animating`, and publishes — never from the tick. */
  #afterEntriesChanged(): void {
    if (this.#entries.size > 0 && this.#removeTick === null) {
      // preUpdate fires every widget tick even in request-render mode; preRender would not.
      // Do not re-wrap `tileset.update`: GaussianSplatPrimitive already owns that slot.
      this.#removeTick = this.#viewer.scene.preUpdate.addEventListener(this.#tick);
    } else if (this.#entries.size === 0 && this.#removeTick !== null) {
      this.#removeTick();
      this.#removeTick = null;
      this.#wasDisplaced = false;
    }
    this.#refreshAnimating();
    this.#publish();
    if (this.#entries.size > 0) this.#viewer.scene.requestRender();
  }

  #refreshAnimating(): void {
    this.#performance.setAnimating(this.#wind.strength > 0 && this.#entries.size > 0);
  }

  readonly #tick = (): void => {
    if (this.#destroyed || this.#entries.size === 0) return;
    const wind = this.#holds > 0 ? WIND_CALM : this.#wind;
    const t = sceneSeconds(this.#viewer.clock.currentTime);
    this.#checkClock(t, wind.strength > 0);
    let displaced = false;
    let phaseChanged = false;
    for (const entry of this.#entries.values()) {
      const before = entry.status;
      // Two terms, computed once per rig per frame: where every node is, and how hard every
      // node's splats are shimmering. Both are pure functions of the same `(rig, t, wind)`, so
      // the frame stays reproducible from the clock alone.
      const started = performance.now();
      const frame = this.#frame(entry, t, wind);
      const modelled = performance.now();
      const status = entry.deformer.apply(frame.transforms, frame.flutter);
      const written = performance.now();
      entry.status = status;
      // The readout's samples: animated frames only, so calm (which costs nothing) does not
      // dilute the figure the two paths are compared on.
      if (status.displaced) {
        entry.motionCost.push(written - started);
        entry.applyCost.push(written - modelled);
      }
      if (status.displaced) displaced = true;
      if (
        status.phase !== before.phase ||
        status.reason !== before.reason ||
        status.displaced !== before.displaced
      )
        phaseChanged = true;
    }
    // Ask for a frame while anything is displaced, and for the one frame that restores the
    // measured pose. Never otherwise: at rest the deformer writes nothing at all.
    if (displaced || this.#wasDisplaced) this.#viewer.scene.requestRender();
    this.#wasDisplaced = displaced;
    // Twice a second at most, and only when a rounded figure moved: see MOTION_COST_PUBLISH_MS.
    if (displaced && this.#refreshCosts()) phaseChanged = true;
    // Phase changes are rare and one-shot — attaching, refusing, a snapshot rebuild — so this
    // is not per-frame work even though it is reached from a per-frame listener.
    if (phaseChanged) this.#settlePhases();
  };

  /** Copies the rolling means into the published readout when due. True when one changed. */
  #refreshCosts(): boolean {
    const now = performance.now();
    if (now < this.#nextCostPublish) return false;
    this.#nextCostPublish = now + MOTION_COST_PUBLISH_MS;
    let changed = false;
    for (const entry of this.#entries.values()) {
      const motionMs = roundMs(entry.motionCost.mean);
      const applyMs = roundMs(entry.applyCost.mean);
      if (motionMs === entry.motionMs && applyMs === entry.applyMs) continue;
      entry.motionMs = motionMs;
      entry.applyMs = applyMs;
      changed = true;
    }
    return changed;
  }

  /**
   * Notices a scene clock that has stopped advancing while wind is on.
   *
   * `deform` is a pure function of `t`, so a clock that does not move produces the same
   * transforms for ever and the tree freezes mid-bend — which is exactly what a motion model
   * with no frequency content in it also looks like. Distinguishing them by eye is impossible,
   * and this is the cheapest thing that can tell them apart. One warning per stall, not one per
   * frame; a calm scene is exempt, since nothing is expected to move there anyway.
   */
  #checkClock(t: number, animating: boolean): void {
    if (!animating) {
      this.#lastSceneSeconds = t;
      this.#stalledTicks = 0;
      this.#reportedStall = false;
      return;
    }
    if (this.#lastSceneSeconds === t) {
      this.#stalledTicks += 1;
      if (this.#stalledTicks >= STALLED_CLOCK_TICKS && !this.#reportedStall) {
        this.#reportedStall = true;
        log.warn("scene clock has stopped; the survey is frozen mid-bend, not becalmed", {
          sceneSeconds: t,
          ticks: this.#stalledTicks,
        });
      }
      return;
    }
    this.#lastSceneSeconds = t;
    this.#stalledTicks = 0;
    this.#reportedStall = false;
  }

  /** Retires anything that refused, toasts why, and republishes. */
  #settlePhases(): void {
    let retired = false;
    for (const [assetId, entry] of this.#entries) {
      if (entry.status.phase !== "refused") continue;
      const reason = entry.status.reason ?? "internal";
      const body = REFUSAL_BODY[reason];
      log.warn("deformation refused", { site: entry.siteSlug, reason });
      if (body !== null) {
        this.#events.emit("toast", {
          tone: "warning",
          title: `${entry.siteSlug} is not moving`,
          body: `${body} The survey is shown exactly as it was measured.`,
          id: `living-${assetId}`,
        });
      }
      entry.deformer.destroy();
      this.#entries.delete(assetId);
      this.#declined.add(assetId);
      retired = true;
    }
    if (retired) this.#afterEntriesChanged();
    else this.#publish();
  }

  /**
   * The published view. `animating` is deliberately a property of the wind and the attachments
   * rather than of the last frame's `displaced` flag: a badge that says motion is simulated must
   * not blink off between two uploads, and a momentary {@link holdMeasuredPose} is a
   * photographic device, not a change in the scene's weather.
   */
  #snapshot(): LivingSurveyStatus {
    const wind = this.#wind;
    const gate = this.gpuMotion;
    const sites: LivingSiteStatus[] = [];
    let ready = false;
    for (const entry of this.#entries.values()) {
      if (entry.status.phase === "ready") ready = true;
      sites.push({
        ...livingMotionPath(entry.status, gate),
        motionMs: entry.motionMs,
        applyMs: entry.applyMs,
        siteId: entry.siteId,
        siteSlug: entry.siteSlug,
        assetId: entry.assetId,
        phase: entry.status.phase,
        reason: entry.status.reason,
        numSplats: entry.status.numSplats,
        displaced: entry.status.displaced,
        rigSourceNote: entry.rig.sourceNote,
        motionEvidence: this.#living(entry)?.sidecar.motionEvidence ?? null,
        maxDisplacementM: this.#maxDisplacement(entry, wind),
        sortStaleness:
          this.#living(entry) === undefined
            ? sortStaleness(entry.rig, wind, REFERENCE_GAUSSIAN_SCALE_M)
            : this.#maxDisplacement(entry, wind) / REFERENCE_GAUSSIAN_SCALE_M,
      });
    }
    return { wind, animating: wind.strength > 0 && ready, sites };
  }

  /** Takes a site's splat primitive out of incremental mode, now and for its lifetime. */
  #aggregate(siteId: string): void {
    const tileset = this.#sites.tilesetFor(siteId, "gaussian-splat");
    if (tileset === null) return;
    // Before the primitive exists (no tile loaded yet), the tileset's setting decides.
    (tileset as unknown as { splatIncremental: boolean }).splatIncremental = false;
    const primitive = splatTilesetOf(tileset).gaussianSplatPrimitive;
    if (primitive?.incremental) {
      primitive.incremental = false;
      this.#viewer.scene.requestRender();
    }
  }

  /** Emits the `living` event, but only when something a listener could see has changed. */
  #publish(): void {
    const next = this.#snapshot();
    const previous = this.#published;
    if (previous !== null && sameStatus(previous, next)) return;
    this.#published = next;
    this.#events.emit("living", next);
  }
}

/** Structural equality over the fields the store mirrors. Keeps React off the per-frame path. */
function sameStatus(a: LivingSurveyStatus, b: LivingSurveyStatus): boolean {
  if (a.animating !== b.animating) return false;
  if (a.wind.strength !== b.wind.strength || a.wind.bearingDeg !== b.wind.bearingDeg) return false;
  if (a.sites.length !== b.sites.length) return false;
  return a.sites.every((site, index) => {
    const other = b.sites[index];
    return (
      site.assetId === other?.assetId &&
      site.phase === other.phase &&
      site.reason === other.reason &&
      site.numSplats === other.numSplats &&
      site.displaced === other.displaced &&
      site.motionEvidence === other.motionEvidence &&
      site.maxDisplacementM === other.maxDisplacementM &&
      site.motionPath === other.motionPath &&
      site.cpuReason === other.cpuReason &&
      site.motionMs === other.motionMs &&
      site.applyMs === other.applyMs
    );
  });
}
