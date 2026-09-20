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
  deform,
  maxDisplacement,
  parseRig,
  sortStaleness,
  WIND_CALM,
  type MotionRig,
  type WindSettings,
} from "@twin/world";

import type { Emitter } from "@/lib/emitter";
import { createLogger, describeError } from "@/lib/log";
import {
  REFERENCE_GAUSSIAN_SCALE_M,
  type LivingSiteStatus,
  type LivingSurveyStatus,
} from "@/state/living";

import { rigUrlFor } from "./livingRigs";
import type { PerformanceManager } from "./PerformanceManager";
import type { SiteManager } from "./SiteManager";
import { splatTilesetOf } from "./splatInternals";
import { SplatDeformer, type DeformerReason, type DeformerStatus } from "./SplatDeformer";
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
 * Why a refusal is permanent, in words a person can act on. Only `refused` phases appear here:
 * the `waiting` reasons (`no-primitive`, `no-snapshot`, `no-texture`, `no-bake-transform`,
 * `no-capture`) are the ordinary first few frames after a tile loads and must never toast.
 */
const REFUSAL_BODY: Readonly<Record<DeformerReason, string | null>> = {
  "multi-tile":
    "This capture is tiled into more than one node, so splat indices are not stable and the rig cannot be trusted to move the right points.",
  layout:
    "The splat texture's addressing parameters are not self-consistent on this device, so the tree would be written into the wrong texels.",
  frame:
    "This capture's frame is not east-north-up, so the motion model cannot tell which way is up.",
  upright: "These points do not read as a standing tree, so the rig does not describe them.",
  checksum: "These are not the splats the motion rig was built for.",
  bake: "The capture's placement could not be undone exactly, so the measured pose could not be guaranteed.",
  internal: "The deformer hit an unexpected error and stopped.",
  // Waiting reasons. Normal for the first frames after load; never a toast.
  "no-primitive": null,
  "no-snapshot": null,
  "no-texture": null,
  "no-bake-transform": null,
  "no-capture": null,
};

/** Everything held for one deformed asset. Dropped whole when its site unloads. */
interface LivingEntry {
  readonly siteId: string;
  readonly siteSlug: string;
  readonly assetId: string;
  readonly rig: MotionRig;
  readonly deformer: SplatDeformer;
  status: DeformerStatus;
}

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
  /** Counted holds that pin every tree at its measured pose (see {@link holdMeasuredPose}). */
  #holds = 0;
  #removeTick: (() => void) | null = null;
  /** Whether the previous tick left anything displaced — the restore frame needs one render. */
  #wasDisplaced = false;
  #published: LivingSurveyStatus | null = null;
  #destroyed = false;

  constructor(
    viewer: Viewer,
    events: Emitter<SceneEvents>,
    sites: SiteManager,
    performance: PerformanceManager,
  ) {
    this.#viewer = viewer;
    this.#events = events;
    this.#sites = sites;
    this.#performance = performance;
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
      const rigUrl = rigUrlFor(asset.siteSlug, asset.representation, asset.sourceUrl);
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
    this.#pending.delete(assetId);
    if (this.#destroyed) return;
    // The site may have unloaded while the rig was in flight.
    const tileset = this.#sites.tilesetFor(candidate.siteId, "gaussian-splat");
    if (tileset === null || this.#entries.has(assetId)) return;

    const deformer = new SplatDeformer({ tileset: splatTilesetOf(tileset), rig });
    this.#entries.set(assetId, {
      siteId: candidate.siteId,
      siteSlug: candidate.slug,
      assetId,
      rig,
      deformer,
      status: deformer.status,
    });
    log.info("motion rig attached", { site: candidate.slug, nodes: rig.nodes.length });
    this.#afterEntriesChanged();
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
    let displaced = false;
    let phaseChanged = false;
    for (const entry of this.#entries.values()) {
      const before = entry.status;
      const status = entry.deformer.apply(deform(entry.rig, t, wind));
      entry.status = status;
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
    // Phase changes are rare and one-shot — attaching, refusing, a snapshot rebuild — so this
    // is not per-frame work even though it is reached from a per-frame listener.
    if (phaseChanged) this.#settlePhases();
  };

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
    const sites: LivingSiteStatus[] = [];
    let ready = false;
    for (const entry of this.#entries.values()) {
      if (entry.status.phase === "ready") ready = true;
      sites.push({
        siteId: entry.siteId,
        siteSlug: entry.siteSlug,
        assetId: entry.assetId,
        phase: entry.status.phase,
        reason: entry.status.reason,
        numSplats: entry.status.numSplats,
        displaced: entry.status.displaced,
        rigSourceNote: entry.rig.sourceNote,
        maxDisplacementM: maxDisplacement(entry.rig, wind),
        sortStaleness: sortStaleness(entry.rig, wind, REFERENCE_GAUSSIAN_SCALE_M),
      });
    }
    return { wind, animating: wind.strength > 0 && ready, sites };
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
      site.maxDisplacementM === other.maxDisplacementM
    );
  });
}
