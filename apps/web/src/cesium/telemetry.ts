/**
 * Live mode's first driver (step C3, docs/SCENE_OBJECTS.md §4 "Telemetry on instances"): a
 * pose stream bound to an instance moves that object rigidly in the viewer.
 *
 * Per binding (`telemetry.json`, `lib/telemetry.ts`) its source's readings are brought into
 * the scan frame (`poseToScan`) and played out by a `PoseTrack` (`@twin/world`): a fixed
 * playout delay over a clock-offset estimate, interpolation, bounded dead reckoning, then
 * held, and once stale frozen or faded back to the rest pose. The pose shown becomes the
 * rigid motion `M = pose · rest⁻¹` and is written through the one motion path the object has:
 *
 * - a **skinned** instance: its skin's constant handle (`Z_0 = [R − I | t_o]` about the skin's
 *   origin, the other handles at rest) via `skinningOf(asset).setInstanceHandles`;
 * - any other instance: the rigid part of the motion chain (`splatRigid.ts`), keyed by the
 *   splats' instance ids.
 *
 * A bound instance (with its ancestors and descendants) is claimed (`motionClaims.ts`), so the
 * wind does not also sway it. At rest -- no reading yet, or faded back -- the object's motion
 * is `null`: the measured frame, pixel for pixel.
 */

import {
  fadeTowardRest,
  PoseTrack,
  poseToScan,
  relativeMotion,
  scanFrame,
  translationAbout,
  type Pose,
  type RigidMotion,
  type ScanFrame,
  type TrackState,
} from "@twin/world";
import { Matrix4, type Cesium3DTileset, type Scene } from "cesium";

import type { InstancesDoc } from "@/lib/instances";
import { createLogger } from "@/lib/log";
import { HANDLE_FLOATS, rigidHandle, type SkinEntry } from "@/lib/skin";
import {
  loadTelemetry,
  telemetryRefOf,
  type TelemetryBinding,
  type TelemetryDoc,
} from "@/lib/telemetry";
import {
  createTelemetrySource,
  type ChannelFactory,
  type TelemetrySource,
} from "@/lib/telemetrySources";

import { claimInstances } from "./motionClaims";
import { cesiumMotionTextures } from "./splatGpuTextures";
import type { MotionTextureFactory } from "./splatGpuMotion";
import { instancesHookOf } from "./splatInstances";
import { splatTilesetOf } from "./splatInternals";
import { SplatRigidMotion } from "./splatRigid";
import { skinningOf } from "./splatSkin";

const log = createLogger("telemetry");

/** What the driver needs of a skin part (`SplatSkinning`). */
export interface TelemetrySkinTarget {
  readonly doc: { readonly byInstance: ReadonlyMap<number, SkinEntry> };
  setInstanceHandles(instanceId: number, handles: ArrayLike<number> | null): boolean;
}

/** What the driver needs of the rigid part (`SplatRigidMotion`). */
export interface TelemetryRigidTarget {
  setInstanceMotion(instanceId: number, motion: RigidMotion | null): void;
}

/** How one binding stands this frame. */
export interface BindingStatus {
  readonly instance: number;
  readonly source: string;
  readonly state: TrackState;
  /** Playout time minus the newest reading, ms (null before the first). */
  readonly ageMs: number | null;
  /** The motion path it is written through, or none while it cannot be placed. */
  readonly via: "skin" | "rigid" | "none";
  /** The body pose shown (scan frame), null at rest. */
  readonly pose: Pose | null;
  /** The rest pose it is measured from (scan frame), null while unknown. */
  readonly rest: Pose | null;
  /** Readings that could not be placed (a frame the scan has no placement for). */
  readonly unplaced: number;
  /** The source time shown, ms. */
  readonly playoutMs: number;
}

export interface TelemetryTick {
  /** Something changed on screen: a render is owed. */
  readonly changed: boolean;
  /** Bindings displaced from rest now. */
  readonly moving: number;
}

interface Bound {
  readonly binding: TelemetryBinding;
  readonly track: PoseTrack;
  readonly rest: Pose | null;
  via: "skin" | "rigid" | "none";
  /** The motion last written, or null (rest). */
  written: RigidMotion | null;
  status: BindingStatus;
  unplaced: number;
}

/** The body's rest pose: the binding's, or the instance's base centre (bounds), level. */
export function restPoseOf(
  binding: TelemetryBinding,
  instances: Pick<InstancesDoc, "byId"> | undefined,
  scan: ScanFrame | undefined,
): Pose | null {
  if (binding.rest) return poseToScan(binding.rest, binding.rest.frame, scan) ?? null;
  const instance = instances?.byId.get(binding.instance);
  if (!instance) return null;
  const { min, max } = instance.bounds;
  return {
    position: [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, min[2]],
    orientation: [0, 0, 0, 1],
  };
}

/** `ids`, their ancestors and their descendants: what a driven body's claim covers. */
export function relatedInstances(
  doc: Pick<InstancesDoc, "instances" | "byId"> | undefined,
  ids: Iterable<number>,
): Set<number> {
  const out = new Set<number>();
  const children = new Map<number, number[]>();
  for (const instance of doc?.instances ?? []) {
    if (instance.parent === null) continue;
    const list = children.get(instance.parent) ?? [];
    list.push(instance.id);
    children.set(instance.parent, list);
  }
  for (const id of ids) {
    out.add(id);
    for (let up = doc?.byId.get(id)?.parent ?? null; up !== null;) {
      if (out.has(up)) break;
      out.add(up);
      up = doc?.byId.get(up)?.parent ?? null;
    }
    const stack = [id];
    for (let at = stack.pop(); at !== undefined; at = stack.pop()) {
      for (const child of children.get(at) ?? []) {
        if (out.has(child)) continue;
        out.add(child);
        stack.push(child);
      }
    }
  }
  return out;
}

function sameMotion(a: RigidMotion | null, b: RigidMotion | null): boolean {
  if (a === null || b === null) return a === b;
  for (let i = 0; i < 4; i += 1) if (a.rotation[i] !== b.rotation[i]) return false;
  for (let i = 0; i < 3; i += 1) if (a.translation[i] !== b.translation[i]) return false;
  return true;
}

/** Moves one scan's bound instances from their sources. */
export class TelemetryDriver {
  readonly doc: TelemetryDoc;
  readonly #sources: ReadonlyMap<string, TelemetrySource>;
  readonly #bound: Bound[];
  readonly #unsubscribe: (() => void)[] = [];
  readonly claimed: ReadonlySet<number>;

  constructor(
    doc: TelemetryDoc,
    instances: Pick<InstancesDoc, "instances" | "byId"> | undefined,
    scan: ScanFrame | undefined,
    sources: ReadonlyMap<string, TelemetrySource>,
  ) {
    this.doc = doc;
    this.#sources = sources;
    this.#bound = doc.bindings.map((binding) => {
      const rest = restPoseOf(binding, instances, scan);
      if (rest === null) {
        log.warn("a bound instance has no rest pose; it stays still", {
          instance: binding.instance,
        });
      }
      const bound: Bound = {
        binding,
        track: new PoseTrack(binding),
        rest,
        via: "none",
        written: null,
        unplaced: 0,
        status: {
          instance: binding.instance,
          source: binding.source,
          state: "none",
          ageMs: null,
          via: "none",
          pose: null,
          rest,
          unplaced: 0,
          playoutMs: 0,
        },
      };
      const source = sources.get(binding.source);
      if (source) {
        this.#unsubscribe.push(
          source.subscribe((sample, arrival) => {
            if (binding.stream !== undefined && sample.stream !== binding.stream) return;
            const pose = poseToScan(sample, sample.frame, scan);
            if (!pose) {
              if (bound.unplaced === 0) {
                log.warn("a reading's frame needs the scan's placement; dropped", {
                  instance: binding.instance,
                  frame: sample.frame,
                });
              }
              bound.unplaced += 1;
              return;
            }
            bound.track.push(sample.t, pose, arrival);
          }),
        );
      }
      return bound;
    });
    this.claimed = relatedInstances(
      instances,
      doc.bindings.map((b) => b.instance),
    );
  }

  /** Every binding's standing, as of the last tick. */
  get statuses(): BindingStatus[] {
    return this.#bound.map((b) => b.status);
  }

  status(instanceId: number): BindingStatus | undefined {
    return this.#bound.find((b) => b.binding.instance === instanceId)?.status;
  }

  /**
   * Pumps the sources to local time `nowMs`, plays every track out at it and writes what moved:
   * through `skin` when it skins the instance, otherwise through `rigid()` (made on demand).
   */
  tick(
    nowMs: number,
    skin: TelemetrySkinTarget | undefined,
    rigid: () => TelemetryRigidTarget | undefined,
  ): TelemetryTick {
    for (const source of this.#sources.values()) source.pump?.(nowMs);
    let changed = false;
    let moving = 0;
    for (const b of this.#bound) {
      const reading = b.track.read(nowMs);
      const rest = b.rest;
      const pose =
        rest !== null && reading.pose !== null
          ? fadeTowardRest(rest, reading.pose, reading.weight)
          : null;
      const motion = pose !== null && rest !== null ? relativeMotion(rest, pose) : null;
      const skinEntry = skin?.doc.byInstance.get(b.binding.instance);
      const via: Bound["via"] = rest === null ? "none" : skinEntry ? "skin" : "rigid";
      if (via !== b.via) {
        // The skin loaded after the rigid part took the object (or went away): hand over.
        if (b.written !== null) {
          this.#write(b, b.via, null, skin, rigid);
          changed = true;
        }
        b.via = via;
      }
      if (!sameMotion(motion, b.written)) {
        this.#write(b, via, motion, skin, rigid);
        changed = true;
      }
      if (b.written !== null) moving += 1;
      b.status = {
        instance: b.binding.instance,
        source: b.binding.source,
        state: reading.state,
        ageMs: reading.ageMs,
        via,
        pose,
        rest,
        unplaced: b.unplaced,
        playoutMs: reading.playoutMs,
      };
    }
    return { changed, moving };
  }

  /** Every bound instance back at rest and every track emptied (a replay starts here). */
  reset(
    skin: TelemetrySkinTarget | undefined,
    rigid: () => TelemetryRigidTarget | undefined,
  ): void {
    for (const b of this.#bound) {
      b.track.clear();
      if (b.written !== null) this.#write(b, b.via, null, skin, rigid);
    }
    for (const source of this.#sources.values()) source.reset?.();
  }

  /** Stops listening and closes the sources. */
  close(): void {
    for (const off of this.#unsubscribe.splice(0)) off();
    for (const source of this.#sources.values()) source.close();
  }

  #write(
    b: Bound,
    via: Bound["via"],
    motion: RigidMotion | null,
    skin: TelemetrySkinTarget | undefined,
    rigid: () => TelemetryRigidTarget | undefined,
  ): void {
    const id = b.binding.instance;
    if (via === "skin") {
      const entry = skin?.doc.byInstance.get(id);
      if (skin && entry) {
        if (motion === null) {
          skin.setInstanceHandles(id, null);
        } else {
          // The constant handle carries the whole motion; the elastic ones stay at rest.
          const handles = new Float64Array(entry.handles * HANDLE_FLOATS);
          rigidHandle(motion.rotation, translationAbout(motion, entry.origin), handles, 0);
          skin.setInstanceHandles(id, handles);
        }
      }
    } else if (via === "rigid") {
      rigid()?.setInstanceMotion(id, motion);
    }
    b.written = motion;
  }
}

// ---- Attachment --------------------------------------------------------------------------

/** One scan's telemetry, once attached: the driver and its rigid part (if made). */
export interface TelemetryAttachment {
  readonly driver: TelemetryDriver;
  readonly sources: ReadonlyMap<string, TelemetrySource>;
  readonly rigid: SplatRigidMotion | undefined;
  /** The clock the driver reads, ms. */
  readonly clock: () => number;
}

const ATTACHED = new Map<string, TelemetryAttachment>();

/** `assetId`'s telemetry, once its bindings loaded and its instances arrived. */
export function telemetryOf(assetId: string): TelemetryAttachment | undefined {
  return ATTACHED.get(assetId);
}

export interface AttachTelemetryOptions {
  /** The local clock, ms (epoch by default; a harness steps its own). */
  readonly clock?: () => number;
  readonly factory?: MotionTextureFactory;
  readonly load?: typeof loadTelemetry;
  /** How stream sources open their channel (tests). */
  readonly channel?: ChannelFactory;
}

/** The scan frame of `tileset`: its root's computed transform, once there is one. */
function scanOf(tileset: Cesium3DTileset): ScanFrame | undefined {
  const root = tileset.root as { computedTransform?: Matrix4 } | undefined;
  if (!root?.computedTransform) return undefined;
  return scanFrame(Matrix4.toArray(root.computedTransform));
}

/**
 * Lets telemetry move `tileset`'s objects, when its root declares `extras.telemetry`: the
 * bindings are fetched once; when the scan's instances have arrived the sources open and the
 * driver ticks on every scene update. Returns the disposer. A tileset without telemetry costs
 * nothing.
 */
export function attachTelemetry(
  tileset: Cesium3DTileset,
  scene: Pick<Scene, "preUpdate" | "requestRender">,
  assetId: string,
  options: AttachTelemetryOptions = {},
): () => void {
  const ref = telemetryRefOf((tileset.root as { extras?: unknown } | undefined)?.extras);
  const url = (tileset as unknown as { resource?: { url?: string } }).resource?.url;
  if (!ref || !url) return () => undefined;
  const clock = options.clock ?? (() => Date.now());
  const factory = options.factory ?? cesiumMotionTextures();
  const load = options.load ?? loadTelemetry;
  let doc: TelemetryDoc | undefined;
  let driver: TelemetryDriver | undefined;
  let rigid: SplatRigidMotion | undefined;
  let release: (() => void) | undefined;
  let disposed = false;
  const rigidFor = (): SplatRigidMotion | undefined => {
    if (rigid) return rigid;
    const hook = instancesHookOf(assetId);
    if (!hook || !factory) return undefined;
    rigid = new SplatRigidMotion(hook, factory, splatTilesetOf(tileset));
    const current = ATTACHED.get(assetId);
    if (current) ATTACHED.set(assetId, { ...current, rigid });
    return rigid;
  };
  const start = (): void => {
    const hook = instancesHookOf(assetId);
    const scan = scanOf(tileset);
    if (!doc || !hook || !scan) return;
    const sources = new Map<string, TelemetrySource>();
    for (const [id, config] of doc.sources) {
      if (!doc.bindings.some((b) => b.source === id)) continue;
      sources.set(id, createTelemetrySource(id, config, { clock, scan }, options.channel));
    }
    driver = new TelemetryDriver(doc, hook.doc, scan, sources);
    release = claimInstances(assetId, driver, driver.claimed);
    ATTACHED.set(assetId, { driver, sources, rigid, clock });
    log.info("telemetry attached", {
      asset: assetId,
      bindings: doc.bindings.length,
      sources: sources.size,
    });
    if (doc.issues.length > 0) log.warn("telemetry.json had problems", { first: doc.issues[0] });
  };
  const offUpdate = scene.preUpdate.addEventListener(() => {
    if (disposed) return;
    if (!driver) {
      start();
      if (!driver) return;
    }
    const tick = driver.tick(clock(), skinningOf(assetId), rigidFor);
    const uploaded = rigid?.sync() ?? false;
    if (tick.changed || uploaded) scene.requestRender();
  });
  load(url, ref)
    .then((loaded) => {
      if (disposed) return;
      doc = loaded;
      scene.requestRender();
    })
    .catch((error: unknown) => {
      log.warn("telemetry did not load; nothing is driven", {
        message: error instanceof Error ? error.message : String(error),
      });
    });
  return () => {
    disposed = true;
    offUpdate();
    driver?.reset(skinningOf(assetId), () => rigid);
    driver?.close();
    release?.();
    rigid?.destroy();
    const current = ATTACHED.get(assetId);
    if (current && current.driver === driver) ATTACHED.delete(assetId);
    driver = undefined;
    rigid = undefined;
  };
}
