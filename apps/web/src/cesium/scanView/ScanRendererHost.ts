/**
 * Draws the active splat scan with a dedicated splat renderer -- Spark (three.js) or PlayCanvas
 * (SuperSplat's engine) -- over the globe, as an alternative to CesiumJS's own splat primitive,
 * chosen in the app for comparison (settings `splatRenderer`).
 *
 * The globe stays CesiumJS: terrain, the world, the camera and its controls (orbit, walk, fly),
 * collision (the scan's packaged solids) and every tool. Only the splats move: CesiumJS's splat
 * tileset is hidden (SiteManager.setSplatRenderer), and the chosen renderer draws the same
 * tiles on a transparent canvas laid over the globe's, from Cesium's camera each frame in the
 * scan's own east/north/up metres. The splats write no depth in any renderer, so drawing them
 * over the globe is what CesiumJS did too.
 *
 * Streaming is the scan viewer page's (view/stream.ts): the tile cut follows the view within
 * the device's splat budget, fetched a few at a time, what is on screen kept until what
 * replaces it is ready. A back-end only turns a tile into its mesh and draws a frame.
 *
 * A frame is drawn only when something it is drawn from changed (overlayFrames.ts): the
 * camera, the canvas, the globe's resolution, a tile, a fade, a sort the renderer finished --
 * while detail is still arriving the overlay draws as often as before, and once everything is
 * in and the camera rests it draws nothing, as the globe under it. Its resolution follows the
 * globe's (quality.ts), and the main-thread part of each tile is spent within a frame budget
 * that shrinks while the camera moves (tileWork.ts). A flight can have its destination's
 * tiles fetched ahead (`prefetchScanDestination`).
 *
 * The scan's objects are the renderer's to draw too: hidden and highlighted (scanInstances.ts),
 * moving as the shared drivers move them (scanMotion.ts: the wind's skins, telemetry's rigid
 * motions) and split objects at their poses (scanObjects.ts), and selectable in the scene from
 * the tiles it draws (`registerPickSource`, cesium/sceneSelect). Motion is one more input a
 * frame is drawn from: the drivers ask the globe for a frame as they move something, and a
 * frame is drawn when the motion handed to the renderer changed (`ScanMotionLink.update`) or a
 * split object moved -- so at rest, with nothing moving, still nothing is drawn.
 *
 * A scan seen from afar (SiteManager's far view: drawn small, not engaged) is the same session
 * as the scan up close -- `ScanTarget.far` is not part of its key, so zooming out and back in
 * never restarts the renderer (no reload, no flash) -- drawn with a quarter of the budget
 * (`FAR_BUDGET_SHARE`) and nothing culled for size (quality.ts), and with no objects to select.
 *
 * Under the WebGPU trial a scan with objects or motion is drawn with WebGL2 (PlayCanvas's
 * default renderer): the modifiers that hide, highlight and move them are GLSL only for now
 * (playcanvasBackend.ts, `WorkBufferModifier`), and the readouts say so
 * (`WEBGL2_FOR_OBJECTS_NOTICE`).
 */

import {
  BoundingSphere,
  Camera,
  Cartesian3,
  Intersect,
  Matrix4,
  type Cesium3DTileset,
  type HeadingPitchRange,
  type Scene,
  type CesiumWidget,
} from "cesium";

import { deviceSplatBudget, deviceSplatCeiling, isHandheld } from "@/lib/detail";
import { instancesRefOf } from "@/lib/instances";
import { AdaptiveSplatBudget } from "@/lib/splatBudget";
import { createLogger, describeError } from "@/lib/log";
import { variantsOf } from "@/lib/variants";
import { useSceneObjects } from "@/state/sceneObjects";
import { DEFAULT_SPLAT_RENDERER } from "@/state/settings";
import { TileStreamer, type View } from "@/view/stream";
import { parseTileset, type TileNode } from "@/view/tiles";

import { uniformScale } from "../placement";
import { DEDICATED_PRIORITY, registerPickSource } from "../sceneSelect/pickSources";
import { inverseScaledTransformation } from "../tilesetScale";
import { FrameMeter, type FrameReading } from "./frameMeter";
import { Handover } from "./handover";
import { OverlayFrames, OverlayInputs, type FrameOutcome } from "./overlayFrames";
import { scanPose } from "./pose";
import { globePixelRatio, maxShDegree, overlayPixelRatio, type GlobeResolution } from "./quality";
import { linkScanInstances } from "./scanInstances";
import { declaresMotion, declaresObjects, ScanMotionLink } from "./scanMotion";
import { ScanObjects } from "./scanObjects";
import { countOverlayDraw } from "./stats";
import { TileWork } from "./tileWork";
import type { BackendHooks, GraphicsApi, ScanBackend, SplatRendererKind } from "./types";

const log = createLogger("scan-renderer");

/** The scan a dedicated renderer draws: the site's splat tileset CesiumJS loaded (hidden). */
export interface ScanTarget {
  key: string;
  tileset: Cesium3DTileset;
  /** The scan's asset id: whose objects (state/instances.ts) the renderer draws hidden or lit. */
  assetId?: string;
  /**
   * Seen from afar (SiteManager's far view), not engaged: drawn small, with a quarter of the
   * budget and nothing culled for size. Not part of `key`: the session carries on either way.
   */
  far?: boolean;
}

/** How often the tile cut is re-planned while the camera moves (ms), as the viewer page. */
const REPLAN_MS = 150;
/** Tiles fetched at once. */
const FETCHES_AT_ONCE = 3;
/** Most gaussians streamed in at once, whatever the budget. */
const MAX_STREAMED = 10_000_000;
/** How long after the camera last moved it counts as resting: the frame drawn then is the
 *  full-resolution one (quality.ts cuts resolution while it moves). */
const MOTION_SETTLE_MS = 200;
/**
 * How long the camera must be still (ms) before the cut may refine past the motion budget, to
 * the device's ceiling: long enough that a pause inside a gesture does not fetch a still
 * view's detail only to drop it again.
 */
export const REST_AFTER_MS = 600;
/**
 * How many times its motion budget a still view may hold (within the device's ceiling): 4.5M
 * on a desktop at its 3M budget. On the Camp scan's arrival views 3M left tiles in view at 16
 * px; 6M, the desktop's ceiling, brought the worst to 7 px but took a software-GL page past
 * what it could hold.
 */
export const REST_GROWTH = 1.5;
/** Most gaussians put on screen per re-plan (~4M a second at REPLAN_MS): tiles that land
 *  together go up over a few frames instead of all in one. */
const MAX_SHOWN_PER_UPDATE = 600_000;
/** The next level of detail within this distance is fetched ahead once the view is served. */
const PREFETCH_RADIUS_M = 30;
/** Loaded tiles kept beyond what is drawn, so a look back needs no download. */
const CACHE_FACTOR = 1.5;
/**
 * The share of its budget a scan seen from afar may draw (`ScanTarget.far`): it is a few dozen
 * pixels across. Its cache keeps the full budget's, so zooming back in finds what was drawn.
 */
export const FAR_BUDGET_SHARE = 0.25;

/** Where a scan's package keeps the renderer-native streamed level of detail, beside its
 *  tileset (PlayCanvas's streamed SOG, written by splat-transform). */
export const NATIVE_LOD_PATH = "sog/lod-meta.json";

/**
 * Whether a tileset says where its native level of detail is: `root.extras.nativeLod` as a
 * uri relative to the tileset (or `{ uri }`) says it has one there, `false` that it has none.
 * `undefined` when it says nothing, as every scan packaged before the key existed.
 */
export function declaredNativeLod(extras: unknown): string | false | undefined {
  if (typeof extras !== "object" || extras === null) return undefined;
  const value = (extras as { nativeLod?: unknown }).nativeLod;
  if (value === false) return false;
  if (typeof value === "string" && value !== "") return value;
  if (typeof value === "object" && value !== null) {
    const uri = (value as { uri?: unknown }).uri;
    if (typeof uri === "string" && uri !== "") return uri;
  }
  return undefined;
}

/** Where the scans found to have no native package are remembered (this device only). */
const NO_NATIVE_KEY = "hexapod.scan.noNativeLod";
/** Most tilesets remembered: the oldest go first. */
const NO_NATIVE_MAX = 200;
/** How long an absence is believed: a package can be published beside a tileset later. */
export const NO_NATIVE_TTL_MS = 3 * 24 * 3600 * 1000;

/** Tilesets probed and found without a native package, and when, from browser storage. */
function rememberedWithout(): [string, number][] {
  try {
    const raw: unknown = JSON.parse(localStorage.getItem(NO_NATIVE_KEY) ?? "[]");
    if (!Array.isArray(raw)) return [];
    return raw.filter(
      (e): e is [string, number] =>
        Array.isArray(e) && typeof e[0] === "string" && typeof e[1] === "number",
    );
  } catch {
    return [];
  }
}

function knownWithout(tilesetUrl: string, now: number): boolean {
  return rememberedWithout().some(([url, at]) => url === tilesetUrl && now - at < NO_NATIVE_TTL_MS);
}

function rememberWithout(tilesetUrl: string, now: number): void {
  try {
    const list = rememberedWithout().filter(
      ([url, at]) => url !== tilesetUrl && now - at < NO_NATIVE_TTL_MS,
    );
    list.push([tilesetUrl, now]);
    localStorage.setItem(NO_NATIVE_KEY, JSON.stringify(list.slice(-NO_NATIVE_MAX)));
  } catch {
    // Storage may be blocked: the probe then runs again next time, as it always did.
  }
}

/**
 * The native level of detail of the tileset at `tilesetUrl`, or null. A declared one
 * (`declaredNativeLod`) is taken as it is, and a declared absence costs no request. A scan
 * that declares nothing is probed once -- a HEAD, so nothing is downloaded -- and a scan found
 * without one is remembered on this device for a few days (`NO_NATIVE_TTL_MS`), so the
 * probe's 404 -- which the browser logs as an error whatever the page does with it -- is not
 * repeated on every visit. A scan that has one behaves as it always did.
 */
export async function findNativeLod(
  tilesetUrl: string,
  extras: unknown,
  probe: (url: string) => Promise<boolean> = async (url) =>
    (await fetch(url, { method: "HEAD" }).catch(() => null))?.ok === true,
): Promise<string | null> {
  const declared = declaredNativeLod(extras);
  if (declared === false) return null;
  if (typeof declared === "string") return new URL(declared, tilesetUrl).toString();
  const now = Date.now();
  if (knownWithout(tilesetUrl, now)) return null;
  const lodUrl = new URL(NATIVE_LOD_PATH, tilesetUrl).toString();
  if (await probe(lodUrl)) return lodUrl;
  rememberWithout(tilesetUrl, now);
  return null;
}

/**
 * THE switch for a scan with objects (`extras.instances`) that also has a native streamed
 * package: false streams its 3D Tiles, whose splats carry object ids by checksum, so the
 * objects can be hidden, highlighted and moved; true streams the native package (SOG), whose
 * splats carry none -- every object drawn whatever the panel says, and the panels report the
 * gap. False for now; flipped to measure what the native package would gain such a scan.
 */
export const NATIVE_SOG_FOR_SCANS_WITH_OBJECTS = false;

/** What the readouts say when the WebGPU trial draws a scan with WebGL2 for its objects. */
export const WEBGL2_FOR_OBJECTS_NOTICE = "WebGL2 for scans with objects or motion";

/** The scan tileset's root extras (what it declares: instances, skin, objects, ...). */
function rootExtrasOf(tileset: Cesium3DTileset): unknown {
  return (tileset.root as { extras?: unknown } | undefined)?.extras;
}

/** The scan tileset's root transform as it is declared, column-major (the splats' frame). */
function rootTransformArray(tileset: Cesium3DTileset): number[] {
  const transform = (tileset.root as { transform?: Matrix4 } | undefined)?.transform;
  return Matrix4.toArray(transform ?? Matrix4.IDENTITY);
}

/**
 * Whether a scan declares objects with ids (`instances.json`, or objects variants to pick from,
 * lib/variants.ts): what its tiles must carry ids for.
 */
export function declaresInstances(extras: unknown): boolean {
  return instancesRefOf(extras) !== null || variantsOf(extras).objects.length > 0;
}

/**
 * Whether a scan declares anything only a GLSL modifier draws: objects (`instances.json`,
 * objects variants, split objects) or motion (a skin, skins variants, telemetry). The WebGPU
 * trial draws such a scan with WebGL2.
 */
export function declaresObjectsOrMotion(extras: unknown): boolean {
  return declaresInstances(extras) || declaresMotion(extras) || declaresObjects(extras);
}

/**
 * Calls `changed` when a split object of `assetId`'s scan is given a pose (by hand or by a
 * driver: `useSceneObjects.setPose`), which no camera move shows. Returns the remover.
 */
function watchPoses(assetId: string | undefined, changed: () => void): () => void {
  if (assetId === undefined) return () => undefined;
  return useSceneObjects.subscribe((state, previous) => {
    if (state.poses[assetId] !== previous.poses[assetId]) changed();
  });
}

/** What a session reports of its moving objects. */
function motionStatus(
  link: ScanMotionLink | null,
  backend: ScanBackend<unknown>,
): ScanRendererStatus["motion"] {
  if (!link || !backend.setMotion) return null;
  const tiles = backend.motionTiles?.() ?? { skinned: 0, redrawn: 0 };
  return { updates: link.updates, ...tiles };
}

/** What the page's tests, the debug panel and the developer readouts read. */
export interface ScanRendererStatus {
  kind: SplatRendererKind;
  active: boolean;
  /** The graphics API the renderer draws with now, or null while none draws. */
  api: GraphicsApi | null;
  /**
   * Why it does not draw with what was chosen, in one line, or null: the WebGPU trial on
   * WebGL2 because the browser has no WebGPU, because the WebGPU device was lost, or because
   * the scan has objects or motion (`WEBGL2_FOR_OBJECTS_NOTICE`).
   */
  notice: string | null;
  /** The WebGPU trial draws this scan with WebGL2 for its objects or motion, not for want of
   *  WebGPU (`declaresObjectsOrMotion`). */
  webgl2ForObjects: boolean;
  /** How fast it drew during the latest camera motion (frameMeter.ts), or null. */
  meter: FrameReading | null;
  tiles: number;
  gaussians: number;
  frames: number;
  error: string | null;
  /** The gaussians it may draw now (adaptive), those being fetched, and those held. */
  budget: number;
  /** Whether the renderer streams the scan's own streamed package (runNative). */
  native: boolean;
  loading: number;
  cached: number;
  /** Tiles that can carry object ids, and those the scan's instances.json lists, or null. */
  instances: { tiles: number; matched: number } | null;
  /**
   * The scan's moving objects (scanMotion.ts): motions handed to the renderer, tiles carrying
   * skin weights, and tile redraws for motion; null while the renderer moves nothing.
   */
  motion: { updates: number; skinned: number; redrawn: number } | null;
  /** Split objects drawn beside the scan (scanObjects.ts). */
  objects: number;
  /** Whether the renderer shows what it was last asked for (`ScanBackend.settled`). */
  settled: boolean;
  /** Whether it draws the scan as seen from afar (`ScanTarget.far`). */
  far: boolean;
}

interface Session {
  kind: SplatRendererKind;
  key: string;
  /** The one start this session came from: what a lost device names (`deviceLost`). */
  token: object;
  api: GraphicsApi;
  notice: string | null;
  webgl2ForObjects: boolean;
  stop(): void;
  status(): Omit<
    ScanRendererStatus,
    | "kind"
    | "active"
    | "instances"
    | "api"
    | "notice"
    | "webgl2ForObjects"
    | "meter"
    | "motion"
    | "objects"
    | "settled"
    | "far"
  >;
  meter(now: number): FrameReading | null;
  instances(): { tiles: number; matched: number } | null;
  motion(): ScanRendererStatus["motion"];
  objects(): number;
  settled(): boolean;
  /** Fetches what a camera at `pose` will draw (a flight's destination); null forgets it. */
  prefetch(pose: CameraPose | null): void;
  /** Draws the scan as seen from afar, or up close again (`ScanTarget.far`). */
  setFar(far: boolean): void;
  far(): boolean;
}

/** A session as drawing makes it; `start` adds which start, API and notice it is. */
type SessionCore = Omit<Session, "token" | "api" | "notice" | "webgl2ForObjects">;

/** How the host makes its renderers. */
export interface ScanRendererOptions {
  /** The renderers' modules (tests hand the host their own); fetched on demand otherwise. */
  backends?: (kind: Exclude<SplatRendererKind, "cesium">) => Promise<BackendModule>;
  /** Keeps each drawn frame readable after it is shown (harnesses read pixels back). */
  preserveDrawingBuffer?: boolean;
}

/** A renderer's module: what `loadBackend` fetches (tests hand the host their own). */
export interface BackendModule {
  createBackend(
    canvas: HTMLCanvasElement,
    budget: number,
    hooks: BackendHooks,
  ): Promise<ScanBackend<unknown>>;
}

async function loadBackend(kind: Exclude<SplatRendererKind, "cesium">): Promise<BackendModule> {
  // Each renderer is its own chunk, fetched only when chosen; PlayCanvas's two share one.
  if (kind === "spark") return import("./sparkBackend");
  const playcanvas = await import("./playcanvasBackend");
  return kind === "playcanvas-webgpu"
    ? { createBackend: playcanvas.createWebgpuBackend }
    : playcanvas;
}

/**
 * A renderer's module could not be fetched (`loadBackend`'s dynamic import): its chunk did not
 * arrive, which is the network's doing or a new release's, not the renderer's or the device's.
 */
export class BackendLoadError extends Error {
  constructor(
    readonly kind: Exclude<SplatRendererKind, "cesium">,
    readonly reason: unknown,
  ) {
    super(`The ${kind} renderer's code did not load: ${describeError(reason)}`);
    this.name = "BackendLoadError";
  }
}

/** A camera pose in Earth-fixed coordinates. */
export interface CameraPose {
  position: Cartesian3;
  direction: Cartesian3;
  up: Cartesian3;
}

/**
 * Where a camera flight will end, in any of the forms the flight code has at hand: a pose; a
 * position with Cesium's `flyTo` orientation (heading, pitch, roll in its east-north-up frame);
 * or a bounding sphere with the offset `flyToBoundingSphere` was given.
 */
export type ScanDestination =
  | CameraPose
  | { position: Cartesian3; heading: number; pitch: number; roll?: number }
  | { boundingSphere: BoundingSphere; offset: HeadingPitchRange };

/** How long a destination is fetched for at most: past the longest flight (5.5 s), and past a
 *  site that engages only once the flight is under way. */
export const DESTINATION_PREFETCH_MS = 15_000;

/** Every live host, for `prefetchScanDestination` (the flight code holds none of them). */
const hosts = new Set<ScanRendererHost>();

/**
 * Fetches ahead, into a dedicated splat renderer's tile cache, what the scan will show from
 * where a camera flight is about to end -- call it as the flight starts (SiteManager's fly-to).
 * CesiumJS preloads a flight's destination for the tilesets it draws
 * (`preloadFlightDestinations`), but a scan drawn by PlayCanvas or Spark is streamed by the
 * overlay, whose CesiumJS copy is hidden without preloading: without this the overlay only ever
 * sees the views on the way, fetches each and abandons it, and starts on the destination once
 * the camera is there.
 *
 * Applies to every live renderer host (there is one per globe), and to a session that starts
 * while the flight is under way (the site engaging on approach). Ends on arrival (the camera
 * within about 2% of the flight's distance of the destination), after
 * `DESTINATION_PREFETCH_MS`, or when the returned function is called (a cancelled flight). A
 * scan streamed in PlayCanvas's own format (runNative) chooses its own level of detail and is
 * not prefetched.
 */
export function prefetchScanDestination(destination: ScanDestination): () => void {
  const cancels = [...hosts].map((host) => host.prefetchDestination(destination));
  return () => cancels.forEach((cancel) => cancel());
}

/** The camera a flight ends at, from any `ScanDestination`, with Cesium's own conventions. */
export function destinationPose(scene: Scene, destination: ScanDestination): CameraPose {
  if ("direction" in destination) {
    return {
      position: Cartesian3.clone(destination.position),
      direction: Cartesian3.clone(destination.direction),
      up: Cartesian3.clone(destination.up),
    };
  }
  const probe = new Camera(scene);
  if ("boundingSphere" in destination) {
    probe.viewBoundingSphere(destination.boundingSphere, destination.offset);
    probe.lookAtTransform(Matrix4.IDENTITY);
  } else {
    probe.setView({
      destination: destination.position,
      orientation: {
        heading: destination.heading,
        pitch: destination.pitch,
        roll: destination.roll ?? 0,
      },
    });
  }
  return {
    position: Cartesian3.clone(probe.positionWC),
    direction: Cartesian3.clone(probe.directionWC),
    up: Cartesian3.clone(probe.upWC),
  };
}

type HostViewer = Pick<
  CesiumWidget,
  "camera" | "canvas" | "scene" | "resolutionScale" | "useBrowserRecommendedResolution"
>;

export class ScanRendererHost {
  private kind: SplatRendererKind = DEFAULT_SPLAT_RENDERER;
  private target: ScanTarget | null = null;
  private session: Session | null = null;
  private starting: Promise<void> | null = null;
  private lastError: string | null = null;
  /** A flight's destination being prefetched, and until when (`prefetchScanDestination`). */
  private destination: { pose: CameraPose; until: number } | null = null;
  private destinationTimer: ReturnType<typeof setTimeout> | null = null;
  /**
   * Why the WebGPU trial draws with WebGL2 for the rest of this visit, once WebGPU failed to
   * start or its device was lost: every later session goes straight to WebGL2 rather than
   * meet the same failure per scan. Choosing the renderer again tries WebGPU again.
   */
  private webgpuFailed: string | null = null;
  /**
   * Starts retired, whether or not their session exists yet: their device was lost
   * (`deviceLost`), or a frame threw (`drawFailed`). One retirement each, and a session still
   * starting is dropped as it arrives (`sync`).
   */
  private readonly retiredStarts = new WeakSet<object>();
  /**
   * The renderer and scan whose overlay threw while drawing (`drawFailed`): not started again
   * for that scan until the renderer is chosen again, or another scan is shown. A renderer
   * that throws on a scan throws on every frame of it, and restarting it on the spot would
   * only fetch the scan again to throw again.
   */
  private drawFailure: { kind: SplatRendererKind; key: string } | null = null;
  /**
   * Told, in words for the screen, when a session's overlay threw and was retired: the scan it
   * drew is gone from the view (CesiumJS keeps its own copy hidden), so the operator should
   * hear why. The scene manager raises a toast.
   */
  onFailure: ((message: string) => void) | null = null;

  private readonly backends: (kind: Exclude<SplatRendererKind, "cesium">) => Promise<BackendModule>;
  private readonly preserveDrawingBuffer: boolean;

  constructor(
    private readonly viewer: HostViewer,
    options: ScanRendererOptions = {},
  ) {
    this.backends = options.backends ?? loadBackend;
    this.preserveDrawingBuffer = options.preserveDrawingBuffer === true;
    hosts.add(this);
  }

  get renderer(): SplatRendererKind {
    return this.kind;
  }

  setRenderer(kind: SplatRendererKind): void {
    if (kind === this.kind) return;
    this.kind = kind;
    if (kind === "playcanvas-webgpu") this.webgpuFailed = null;
    // Choosing a renderer is asking for it: one that threw on this scan is tried again.
    this.drawFailure = null;
    this.sync();
  }

  setTarget(target: ScanTarget | null): void {
    if (target?.key === this.target?.key && target?.tileset === this.target?.tileset) {
      // The same scan from nearer or further: the session carries on, and only what it may
      // draw changes (`Session.setFar`).
      if (target && (target.far === true) !== (this.target?.far === true)) {
        this.target = target;
        this.session?.setFar(target.far === true);
      }
      return;
    }
    this.target = target;
    this.sync();
  }

  /**
   * Fetches what the scan will show from `destination` ahead of the camera
   * (`prefetchScanDestination`, which calls this on every host). Returns the cancel.
   */
  prefetchDestination(destination: ScanDestination): () => void {
    const pose = destinationPose(this.viewer.scene, destination);
    const entry = { pose, until: performance.now() + DESTINATION_PREFETCH_MS };
    this.destination = entry;
    this.session?.prefetch(pose);
    if (this.destinationTimer !== null) clearTimeout(this.destinationTimer);
    this.destinationTimer = setTimeout(() => this.endPrefetch(entry), DESTINATION_PREFETCH_MS);
    return () => this.endPrefetch(entry);
  }

  private endPrefetch(entry: { pose: CameraPose; until: number }): void {
    if (this.destination !== entry) return;
    this.destination = null;
    if (this.destinationTimer !== null) clearTimeout(this.destinationTimer);
    this.destinationTimer = null;
    this.session?.prefetch(null);
  }

  status(): ScanRendererStatus {
    const session = this.session;
    const inner = session?.status() ?? {
      tiles: 0,
      gaussians: 0,
      frames: 0,
      error: null,
      budget: 0,
      loading: 0,
      cached: 0,
      native: false,
    };
    return {
      kind: this.kind,
      active: session !== null,
      api: session?.api ?? null,
      notice: session?.notice ?? null,
      webgl2ForObjects: session?.webgl2ForObjects ?? false,
      meter: session?.meter(performance.now()) ?? null,
      ...inner,
      instances: session?.instances() ?? null,
      motion: session?.motion() ?? null,
      objects: session?.objects() ?? 0,
      settled: session?.settled() ?? true,
      far: session?.far() ?? false,
      error: inner.error ?? this.lastError,
    };
  }

  destroy(): void {
    hosts.delete(this);
    if (this.destinationTimer !== null) clearTimeout(this.destinationTimer);
    this.destinationTimer = null;
    this.destination = null;
    this.target = null;
    this.session?.stop();
    this.session = null;
  }

  private sync(): void {
    const failedHere =
      this.drawFailure?.kind === this.kind && this.drawFailure.key === this.target?.key;
    const wanted = this.kind !== "cesium" && this.target !== null && !failedHere;
    const current = this.session;
    if (current && (!wanted || current.kind !== this.kind || current.key !== this.target?.key)) {
      current.stop();
      this.session = null;
    }
    if (!wanted || this.session || this.starting) return;
    const kind = this.kind;
    const target = this.target;
    if (kind === "cesium" || !target) return;
    this.starting = this.start(kind, target)
      .then((session) => {
        // Chosen away from while it started, or its device lost on the way: undone at once
        // (the `finally` below starts the replacement).
        if (
          this.kind !== kind ||
          this.target?.key !== target.key ||
          this.retiredStarts.has(session.token)
        ) {
          session.stop();
        } else {
          this.session = session;
          // Zoomed in or out while it started: drawn as the target says now.
          session.setFar(this.target?.far === true);
          // A flight under way when the scan came in: its destination still counts.
          const destination = this.destination;
          if (destination && performance.now() < destination.until) {
            session.prefetch(destination.pose);
          }
        }
      })
      .catch((error: unknown) => {
        this.lastError = error instanceof Error ? error.message : String(error);
        log.warn("splat renderer failed to start", { kind, error: this.lastError });
      })
      .finally(() => {
        this.starting = null;
        this.sync();
      });
  }

  /** What the globe renders at now: the overlay's resolution follows it (quality.ts). */
  private globeResolution(): GlobeResolution {
    return {
      devicePixelRatio: window.devicePixelRatio || 1,
      browserRecommended: this.viewer.useBrowserRecommendedResolution,
      resolutionScale: this.viewer.resolutionScale,
    };
  }

  /** A new transparent canvas for a renderer, right above the globe's. */
  private overlayCanvas(kind: SplatRendererKind): HTMLCanvasElement {
    const canvas = document.createElement("canvas");
    canvas.dataset.scanRenderer = kind;
    Object.assign(canvas.style, {
      position: "absolute",
      inset: "0",
      width: "100%",
      height: "100%",
      pointerEvents: "none",
    });
    // Right above the globe's canvas, under Cesium's credits and every panel.
    this.viewer.canvas.insertAdjacentElement("afterend", canvas);
    return canvas;
  }

  /**
   * A renderer lost its GPU device for good (`BackendHooks.deviceLost`; WebGPU only): the
   * session it belongs to is replaced by one drawing with WebGL2, on a new canvas, and draws
   * again from the start. Later sessions this visit go straight to WebGL2 (`webgpuFailed`).
   */
  private deviceLost(token: object, reason: string): void {
    if (this.retiredStarts.has(token)) return;
    this.retiredStarts.add(token);
    this.webgpuFailed = reason;
    log.warn("splat renderer lost its device; drawing with WebGL2 instead", { reason });
    this.retire(token);
  }

  /**
   * A frame of the session from start `token` threw (OverlayFrames has stopped it already, and
   * kept the throw out of CesiumJS's frame): the session is retired, its error is what the
   * status says, the scan is not started again with this renderer (`drawFailure`), and whoever
   * listens (`onFailure`) is told. The globe and its other managers never notice.
   */
  private drawFailed(token: object, kind: SplatRendererKind, key: string, error: unknown): void {
    if (this.retiredStarts.has(token)) return;
    this.retiredStarts.add(token);
    const message = `The ${kind} splat renderer stopped: ${describeError(error)}`;
    this.lastError = message;
    this.drawFailure = { kind, key };
    log.error("splat renderer threw while drawing; its overlay is retired", {
      kind,
      error: describeError(error),
    });
    this.retire(token);
    try {
      this.onFailure?.(message);
    } catch {
      // A listener's own failure is not the overlay's to report.
    }
  }

  /**
   * Stops the session from start `token`, if it is the one drawing, and lets `sync` decide what
   * replaces it. Never from inside the renderer's own callback or frame: on the next task. A
   * session still starting is dropped as it arrives (`sync`, `retiredStarts`).
   */
  private retire(token: object): void {
    setTimeout(() => {
      const session = this.session;
      if (session?.token !== token) return;
      this.session = null;
      try {
        session.stop();
      } catch (error) {
        // A renderer that broke while drawing can break again on the way out; its canvas is
        // gone either way (the session's `stop`), and nothing of it is drawn again.
        log.warn("a retired splat renderer did not stop cleanly", { error: describeError(error) });
      }
      this.sync();
    }, 0);
  }

  private async start(
    kind: Exclude<SplatRendererKind, "cesium">,
    target: ScanTarget,
  ): Promise<Session> {
    this.lastError = null;
    const budget = deviceSplatBudget();
    /** This start, as a lost device names it. */
    const token = {};
    // The session's frame driver exists only once the session does; until then a renderer's
    // request for a frame is dropped (nothing is drawn before the first frame anyway).
    const wake = { frame: (_reason: string): void => undefined };
    const work = new TileWork();
    const hooks: BackendHooks = {
      frameWanted: () => wake.frame("renderer"),
      work,
      maxShDegree: maxShDegree(isHandheld()),
      deviceLost: (reason) => this.deviceLost(token, reason),
      preserveDrawingBuffer: this.preserveDrawingBuffer,
    };
    const create = async (
      module: Exclude<SplatRendererKind, "cesium">,
    ): Promise<{ canvas: HTMLCanvasElement; backend: ScanBackend<unknown> }> => {
      // Tagged with the back-end actually made, not the one chosen: the WebGPU trial's WebGL2
      // fallback is PlayCanvas's default renderer, on a canvas of its own.
      const canvas = this.overlayCanvas(module);
      try {
        let loaded: BackendModule;
        try {
          loaded = await this.backends(module);
        } catch (error) {
          throw new BackendLoadError(module, error);
        }
        return { canvas, backend: await loaded.createBackend(canvas, budget, hooks) };
      } catch (error) {
        canvas.remove();
        throw error;
      }
    };
    // The WebGPU trial never leaves a scan undrawn: when WebGPU does not start (or PlayCanvas
    // started nothing but its Null device), or failed earlier this visit, PlayCanvas on WebGL2
    // draws it -- the default renderer, on a canvas no WebGPU context ever touched -- and the
    // developer readouts say why.
    let made: { canvas: HTMLCanvasElement; backend: ScanBackend<unknown> } | null = null;
    /** Why this one session draws with WebGL2 when WebGPU itself was never tried. */
    let loadNotice: string | null = null;
    // A scan with objects or motion is drawn with WebGL2 from the start: what hides, lights and
    // moves them is GLSL only for now (playcanvasBackend.ts). WebGPU is not held against.
    const webgl2ForObjects =
      kind === "playcanvas-webgpu" &&
      target.assetId !== undefined &&
      declaresObjectsOrMotion(rootExtrasOf(target.tileset));
    if (kind === "playcanvas-webgpu" && this.webgpuFailed === null && !webgl2ForObjects) {
      try {
        made = await create(kind);
      } catch (error) {
        if (error instanceof BackendLoadError) {
          // The renderer's code did not arrive (a dropped connection, a new release replacing
          // the chunks): that says nothing about WebGPU, so it is not held against it for the
          // visit. This session draws with WebGL2, and the next one asks for WebGPU again.
          loadNotice = `${error.message}; WebGPU is tried again with the next scan`;
          log.warn("WebGPU splat renderer's code did not load; drawing with WebGL2", {
            error: error.message,
          });
        } else {
          this.webgpuFailed = `WebGPU did not start: ${describeError(error)}`;
          log.warn("WebGPU splat renderer did not start; drawing with WebGL2", {
            error: this.webgpuFailed,
          });
        }
      }
    }
    made ??= await create(kind === "playcanvas-webgpu" ? "playcanvas" : kind);
    const { canvas, backend } = made;
    const api = backend.api ?? "webgl2";
    canvas.dataset.api = api;
    const notice =
      kind !== "playcanvas-webgpu"
        ? null
        : webgl2ForObjects
          ? WEBGL2_FOR_OBJECTS_NOTICE
          : (backend.apiNote ?? this.webgpuFailed ?? loadNotice);
    // A frame that throws retires this session, never the globe (overlayFrames.ts).
    const failed = (error: unknown): void => this.drawFailed(token, kind, target.key, error);
    try {
      const session = await this.run(kind, target, canvas, backend, budget, wake, work, failed);
      return Object.assign(session, { token, api, notice, webgl2ForObjects });
    } catch (error) {
      // Whatever failed before the first frame leaves nothing behind.
      work.stop();
      backend.destroy();
      canvas.remove();
      throw error;
    }
  }

  private async run(
    kind: Exclude<SplatRendererKind, "cesium">,
    target: ScanTarget,
    canvas: HTMLCanvasElement,
    backend: ScanBackend<unknown>,
    budget: number,
    wake: { frame: (reason: string) => void },
    work: TileWork,
    failed: (error: unknown) => void,
  ): Promise<SessionCore> {
    const { viewer } = this;
    const url = new URL(target.tileset.resource.url, location.href).toString();
    if (backend.streamNative) {
      const native = await this.runNative(
        kind,
        target,
        canvas,
        backend,
        url,
        budget,
        wake,
        work,
        failed,
      );
      if (native) return native;
    }
    const response = await fetch(url);
    if (!response.ok) throw new Error(`The scan's tileset answered ${String(response.status)}.`);
    const tree = parseTileset(await response.json());
    let error: string | null = null;
    // As CesiumJS's splats: the device budget to start, then what motion frame times allow.
    const adaptive = new AdaptiveSplatBudget(deviceSplatCeiling(), budget);
    const streamedFor = (drawn: number): number =>
      Math.min(drawn * backend.loadFactor, MAX_STREAMED);
    /** Seen from afar (`ScanTarget.far`): a share of the budget, nothing culled for size. */
    let far = target.far === true;
    /** What the view may draw now: the adaptive budget, cut while the scan is seen from afar.
     *  (A far view draws too little for its frame times to move the adaptive budget.) */
    const drawBudget = (): number =>
      far ? Math.round(adaptive.budget * FAR_BUDGET_SHARE) : adaptive.budget;
    /** The camera has been still for `REST_AFTER_MS`: the view may refine to `restBudget`. */
    let resting = false;
    /**
     * What the cut may hold now: the motion budget, or at rest `REST_GROWTH` times it, within
     * the device's ceiling (lib/detail.ts `deviceSplatCeiling`; a phone's is its budget). A
     * still overlay draws nothing, so what a still view holds costs memory and one sort, not
     * frame time; the next move plans with the motion budget again and swaps back to the
     * ancestors kept loaded. A step on the motion budget rather than the ceiling, so it scales
     * with what this device's frames have shown it can draw. Not the renderer's own budget:
     * PlayCanvas re-copies and re-sorts every splat when that changes, and the tile cut is
     * what decides what it draws.
     */
    const restBudget = (): number =>
      Math.max(adaptive.budget, Math.min(adaptive.ceiling, adaptive.budget * REST_GROWTH));
    const planBudget = (): number => (resting && !far ? restBudget() : drawBudget());
    /** Tiles kept: the full budget's share whether near or far, so zooming in finds them, and
     *  at least what a still view refines to. */
    const cacheBudget = (): number =>
      Math.max(streamedFor(adaptive.budget) * CACHE_FACTOR, streamedFor(restBudget()));
    if (far) backend.setBudget(drawBudget());
    const handover = new Handover<unknown>({
      add: (mesh) => backend.add(mesh),
      remove: (mesh) => backend.remove(mesh),
      isDrawn: (mesh, sinceMs) => backend.isDrawn(mesh, sinceMs),
      ...(backend.fade ? { fade: backend.fade.bind(backend) } : {}),
    });
    const streamer = new TileStreamer<unknown>(
      tree,
      {
        load: (tile, signal) => backend.load(url, tile, signal),
        show: (_tile, mesh) => handover.show(mesh, performance.now()),
        hide: (_tile, mesh) => handover.hide(mesh, performance.now()),
        dispose: (mesh) => {
          handover.forget(mesh);
          backend.dispose(mesh);
        },
        failed: (tile: TileNode, reason: unknown) => {
          error = reason instanceof Error ? reason.message : String(reason);
          log.warn("scan tile did not load; its parent stays", { tile: tile.uri, error });
        },
      },
      {
        budget: streamedFor(drawBudget()),
        cacheBudget: cacheBudget(),
        concurrency: FETCHES_AT_ONCE,
        maxShownPerUpdate: MAX_SHOWN_PER_UPDATE,
        prefetchRadiusM: PREFETCH_RADIUS_M,
      },
    );
    let arrived = true;
    /** Hands the renderer and the streamer what may be drawn now (`drawBudget`, `planBudget`). */
    const applyBudget = (): void => {
      backend.setBudget(drawBudget());
      streamer.setBudget(streamedFor(planBudget()), cacheBudget());
      arrived = true;
    };
    /** The camera came to rest, or moved again: the cut is planned again under its budget. */
    const rest = (next: boolean): void => {
      if (next === resting) return;
      resting = next;
      streamer.setBudget(streamedFor(planBudget()), cacheBudget());
      arrived = true;
    };
    // A tile arrived or failed, a deferred swap is due, a failed tile may be tried again.
    streamer.onArrival = () => {
      arrived = true;
      wake.frame("tiles");
    };
    // The scan's objects move as the shared drivers move them (scanMotion.ts), and its split
    // objects are drawn where their poses put them (scanObjects.ts).
    const extras = rootExtrasOf(target.tileset);
    const motionLink = target.assetId
      ? new ScanMotionLink(target.assetId, backend, { native: false, extras })
      : null;
    const objects = new ScanObjects<unknown>(backend, target.assetId, () => wake.frame("objects"));
    /**
     * Hands the renderer the drivers' latest motion and places the split objects at their
     * poses; true when either changed what a frame shows. Both run every time: a frame drawn
     * for one shows the other as it is now.
     */
    const pollMotion = (): boolean => {
      const handed = motionLink?.update() ?? false;
      const moved = objects.tick();
      return handed || moved;
    };

    const toLocal = new Matrix4();
    const toWorld = new Matrix4();
    const lastEye = new Cartesian3(Number.NaN, 0, 0);
    const lastDirection = new Cartesian3();
    const scratchCentre = new Cartesian3();
    const scratchForward = new Cartesian3();
    const scratchSphere = new BoundingSphere();
    let lastPlan = 0;
    // Motion frames, for the adaptive budget and the resolution: the camera moved since the
    // last frame drawn.
    const frameEye = new Cartesian3(Number.NaN, 0, 0);
    const frameDirection = new Cartesian3();
    let lastMotionFrameAt = 0;
    let lastMotionAt = 0;
    let frames = 0;
    const meter = new FrameMeter();
    const handheld = isHandheld();
    const inputs = new OverlayInputs();
    const inputSize = (): { width: number; height: number; pixelRatio: number } => ({
      width: viewer.canvas.clientWidth,
      height: viewer.canvas.clientHeight,
      pixelRatio: globePixelRatio(this.globeResolution()),
    });

    /** The view from a camera at `position` looking along `direction`, in the scan's frame. */
    const viewFrom = (
      position: Cartesian3,
      direction: Cartesian3,
      up: Cartesian3,
      eye: [number, number, number],
      height: number,
    ): View => {
      const camera = viewer.camera;
      const culling = camera.frustum.computeCullingVolume(position, direction, up);
      const fovy = (camera.frustum as { fovy?: number }).fovy ?? Math.PI / 3;
      // The middle of the view, in the scan's frame: its finest tiles go there first.
      const forward = Matrix4.multiplyByPointAsVector(toLocal, direction, scratchForward);
      Cartesian3.normalize(forward, forward);
      const aspect = viewer.canvas.clientWidth / Math.max(1, height);
      return {
        eye,
        projection: height / (2 * Math.tan(fovy / 2)),
        centre: {
          forward: [forward.x, forward.y, forward.z],
          halfDiagonal: Math.atan(Math.tan(fovy / 2) * Math.hypot(1, aspect)),
        },
        visible: (bounds) => {
          Matrix4.multiplyByPoint(
            toWorld,
            Cartesian3.fromArray(bounds.center, 0, scratchCentre),
            scratchSphere.center,
          );
          // A tile's radius is in the scan's frame: `scale` metres a unit under a runtime scale.
          scratchSphere.radius = bounds.radius * uniformScale(toWorld);
          return culling.computeVisibility(scratchSphere) !== Intersect.OUTSIDE;
        },
      };
    };

    const frame = (): FrameOutcome => {
      const tileset = target.tileset;
      if (tileset.isDestroyed()) return { again: false, by: null };
      Matrix4.clone(tileset.root.computedTransform, toWorld);
      inverseScaledTransformation(toWorld, toLocal);
      const camera = viewer.camera;
      const moved =
        !Cartesian3.equalsEpsilon(camera.positionWC, lastEye, 0, 1e-3) ||
        !Cartesian3.equalsEpsilon(camera.directionWC, lastDirection, 1e-5);
      const now = performance.now();
      const motion =
        !Cartesian3.equalsEpsilon(camera.positionWC, frameEye, 0, 1e-3) ||
        !Cartesian3.equalsEpsilon(camera.directionWC, frameDirection, 1e-5);
      Cartesian3.clone(camera.positionWC, frameEye);
      Cartesian3.clone(camera.directionWC, frameDirection);
      if (motion) lastMotionAt = now;
      // Dynamic resolution, as games do: fewer pixels while the view moves (blending splats
      // is per pixel, and the GPU was what ran out), the full resolution once it rests -- the
      // globe's, at most (quality.ts).
      const moving = now - lastMotionAt < MOTION_SETTLE_MS;
      work.moving = moving;
      // Still long enough: the cut may refine to the rest budget. Moving: back to the motion
      // budget at once, so the first re-plan of a gesture swaps the surplus out.
      rest(now - lastMotionAt >= REST_AFTER_MS);
      const size = inputSize();
      const pose = {
        ...scanPose(viewer.camera, toLocal, {
          width: size.width,
          height: size.height,
          pixelRatio: overlayPixelRatio(this.globeResolution(), { handheld, moving }),
        }),
        farView: far,
      };
      const drawn = Math.min(streamer.drawnGaussians, adaptive.budget);
      // Frames while tiles arrive are slowed by their uploads, not by what is drawn: only a
      // steady view's motion frames say what the GPU can sort and blend -- and only one that
      // follows another motion frame, now that a resting overlay draws nothing in between.
      // Nor one still drawing a still view's surplus, which the first re-plan takes away.
      const steady =
        motion &&
        streamer.loading === 0 &&
        lastMotionFrameAt > 0 &&
        streamer.drawnGaussians <= adaptive.budget;
      let budgetMoved = false;
      if (steady && adaptive.frame(now - lastMotionFrameAt, drawn)) {
        applyBudget();
        budgetMoved = true;
        log.info("splat budget", { kind, budget: adaptive.budget });
      }
      lastMotionFrameAt = motion ? now : 0;
      let replanAt: number | null = null;
      if (arrived || moved) {
        if (now - lastPlan >= REPLAN_MS) {
          lastPlan = now;
          arrived = false;
          Cartesian3.clone(camera.positionWC, lastEye);
          Cartesian3.clone(camera.directionWC, lastDirection);
          streamer.update(
            viewFrom(camera.positionWC, camera.directionWC, camera.upWC, pose.eye, pose.height),
          );
        } else {
          // Held back by the throttle: re-planned once it allows, camera moving or not.
          replanAt = lastPlan + REPLAN_MS;
        }
      }
      // The drivers' motion as it is now (`changed` may have handed it already).
      pollMotion();
      const drawStart = performance.now();
      backend.render(pose);
      if (motion) meter.record(drawStart, performance.now() - drawStart);
      countOverlayDraw(canvas, pose.pixelRatio);
      inputs.commit(camera, size, toWorld);
      const step = handover.tick(performance.now());
      frames += 1;
      const deadlines = [
        moving ? lastMotionAt + MOTION_SETTLE_MS : null,
        // The view the budget held back is refined once the camera has been still a moment;
        // one the budget did not hold back has nothing more to show.
        !resting && streamer.limited ? lastMotionAt + REST_AFTER_MS : null,
        replanAt,
        step.nextAt,
        // Work the renderer held back (a moving object's throttled re-sort): its last pose is
        // sorted by then even when nothing moves any more.
        backend.frameDueBy?.() ?? null,
      ].filter((t): t is number => t !== null);
      return {
        again: step.changed || step.animating || budgetMoved,
        by: deadlines.length > 0 ? Math.min(...deadlines) : null,
      };
    };

    // The root first, as the viewer page: the whole scan, coarse, while the rest streams.
    let root: unknown;
    try {
      root = await backend.load(url, tree.root);
    } catch (reason) {
      streamer.stop();
      motionLink?.dispose();
      throw reason;
    }
    handover.show(root, performance.now());
    streamer.adopt(tree.root, root);
    void objects.load(url, extras, rootTransformArray(target.tileset));
    const driver = new OverlayFrames(
      (listener) => viewer.scene.postRender.addEventListener(listener),
      {
        // What the frame is drawn from moved -- or the objects did: the drivers ask the globe
        // for a frame as they move something (LivingSurveyManager, telemetry.ts), and this is
        // where the overlay hears of it.
        changed: () => {
          const tileset = target.tileset;
          if (tileset.isDestroyed()) return false;
          return (
            inputs.changed(viewer.camera, inputSize(), tileset.root.computedTransform) ||
            pollMotion()
          );
        },
        draw: frame,
        failed,
      },
    );
    wake.frame = (reason) => driver.wake(reason);
    driver.wake("start");
    // A pose set by hand or by a driver (`useSceneObjects.setPose`).
    const unwatchPoses = watchPoses(target.assetId, () => driver.wake("objects"));
    const unlinkInstances = target.assetId
      ? linkScanInstances(target.assetId, backend, false, undefined, () => driver.wake("instances"))
      : () => undefined;
    // Scene selection picks from the tiles this renderer draws (cesium/sceneSelect).
    const unlinkPick =
      target.assetId && backend.pickTiles
        ? registerPickSource(
            target.assetId,
            {
              renderer: backend.name,
              // Objects are selected in an engaged scan, not in one seen from afar.
              tiles: () => (far ? [] : (backend.pickTiles?.() ?? [])),
              toWorld: () =>
                target.tileset.isDestroyed() ? undefined : target.tileset.root.computedTransform,
            },
            DEDICATED_PRIORITY,
          )
        : () => undefined;
    log.info("splat renderer started", { kind, tiles: tree.root.uri });

    return {
      kind,
      key: target.key,
      setFar: (next) => {
        if (next === far) return;
        far = next;
        applyBudget();
        driver.wake("far");
      },
      far: () => far,
      stop: () => {
        unlinkPick();
        unlinkInstances();
        motionLink?.dispose();
        objects.stop();
        unwatchPoses();
        driver.stop();
        work.stop();
        // The canvas goes whatever the renderer does on the way out (a retired one may throw).
        try {
          streamer.stop();
          backend.destroy();
        } finally {
          canvas.remove();
        }
      },
      prefetch: (destination) => {
        if (!destination) {
          streamer.prefetchView(null);
          return;
        }
        const tileset = target.tileset;
        if (tileset.isDestroyed()) return;
        Matrix4.clone(tileset.root.computedTransform, toWorld);
        inverseScaledTransformation(toWorld, toLocal);
        const eye = Matrix4.multiplyByPoint(toLocal, destination.position, new Cartesian3());
        streamer.prefetchView(
          viewFrom(
            destination.position,
            destination.direction,
            destination.up,
            [eye.x, eye.y, eye.z],
            viewer.canvas.clientHeight,
          ),
        );
      },
      instances: () => backend.instanceTiles?.() ?? null,
      motion: () => motionStatus(motionLink, backend),
      objects: () => objects.count,
      settled: () => backend.settled?.() ?? true,
      meter: (now) => meter.reading(now),
      status: () => ({
        tiles: streamer.drawn.length,
        gaussians: streamer.drawnGaussians,
        frames,
        error,
        budget: drawBudget(),
        loading: streamer.loading,
        cached: streamer.loadedGaussians,
        native: false,
      }),
    };
  }

  /**
   * The scan in the renderer's own streamed format, when its package has one: the renderer
   * streams and chooses by itself and this only keeps its camera on Cesium's. Null when the
   * package has none (older scans), and the tileset is streamed here instead. Also null for
   * a scan with objects (`extras.instances`) while `NATIVE_SOG_FOR_SCANS_WITH_OBJECTS` is off:
   * the native package's splats carry no object ids, so the objects could not be hidden or
   * highlighted there (the published camp, which has both, drew every object whatever the
   * panel said); its 3D Tiles carry them by checksum.
   */
  private async runNative(
    kind: Exclude<SplatRendererKind, "cesium">,
    target: ScanTarget,
    canvas: HTMLCanvasElement,
    backend: ScanBackend<unknown>,
    tilesetUrl: string,
    budget: number,
    wake: { frame: (reason: string) => void },
    work: TileWork,
    failed: (error: unknown) => void,
  ): Promise<SessionCore | null> {
    if (!backend.streamNative) return null;
    const extras = rootExtrasOf(target.tileset);
    if (!NATIVE_SOG_FOR_SCANS_WITH_OBJECTS && target.assetId && declaresInstances(extras)) {
      return null;
    }
    const lodUrl = await findNativeLod(tilesetUrl, extras);
    if (lodUrl === null) return null;
    const stream = await backend.streamNative(lodUrl);
    // The renderer chooses its own level of detail within its budget: from afar, a share of it.
    let far = target.far === true;
    const applyBudget = (): void =>
      backend.setBudget(far ? Math.round(budget * FAR_BUDGET_SHARE) : budget);
    if (far) applyBudget();
    const { viewer } = this;
    const toLocal = new Matrix4();
    const handheld = isHandheld();
    const lastEye = new Cartesian3(Number.NaN, 0, 0);
    const lastDirection = new Cartesian3();
    const inputs = new OverlayInputs();
    const inputSize = (): { width: number; height: number; pixelRatio: number } => ({
      width: viewer.canvas.clientWidth,
      height: viewer.canvas.clientHeight,
      pixelRatio: globePixelRatio(this.globeResolution()),
    });
    let lastMotionAt = 0;
    let frames = 0;
    const meter = new FrameMeter();
    // Streaming and sorting are the renderer's: it asks for a frame when it has new detail or
    // a new order (`hooks.frameWanted`); the camera, the canvas and the settle are this one's.
    const frame = (): FrameOutcome => {
      const tileset = target.tileset;
      if (tileset.isDestroyed()) return { again: false, by: null };
      inverseScaledTransformation(tileset.root.computedTransform, toLocal);
      const camera = viewer.camera;
      const now = performance.now();
      const motion =
        !Cartesian3.equalsEpsilon(camera.positionWC, lastEye, 0, 1e-3) ||
        !Cartesian3.equalsEpsilon(camera.directionWC, lastDirection, 1e-5);
      if (motion) {
        lastMotionAt = now;
        Cartesian3.clone(camera.positionWC, lastEye);
        Cartesian3.clone(camera.directionWC, lastDirection);
      }
      const moving = now - lastMotionAt < MOTION_SETTLE_MS;
      work.moving = moving;
      const size = inputSize();
      const ratio = overlayPixelRatio(this.globeResolution(), { handheld, moving });
      const drawStart = performance.now();
      backend.render({
        ...scanPose(camera, toLocal, { width: size.width, height: size.height, pixelRatio: ratio }),
        farView: far,
      });
      if (motion) meter.record(drawStart, performance.now() - drawStart);
      countOverlayDraw(canvas, ratio);
      inputs.commit(camera, size, tileset.root.computedTransform);
      frames += 1;
      return { again: false, by: moving ? lastMotionAt + MOTION_SETTLE_MS : null };
    };
    // The native package carries no tile checksums: no object ids, no skins (the panels say
    // so); split objects are tiles of their own, drawn as in any session.
    const motionLink = target.assetId
      ? new ScanMotionLink(target.assetId, backend, { native: true, extras })
      : null;
    const objects = new ScanObjects<unknown>(backend, target.assetId, () => wake.frame("objects"));
    void objects.load(tilesetUrl, extras, rootTransformArray(target.tileset));
    const driver = new OverlayFrames(
      (listener) => viewer.scene.postRender.addEventListener(listener),
      {
        changed: () => {
          const tileset = target.tileset;
          if (tileset.isDestroyed()) return false;
          return (
            inputs.changed(viewer.camera, inputSize(), tileset.root.computedTransform) ||
            objects.tick()
          );
        },
        draw: () => {
          objects.tick();
          return frame();
        },
        failed,
      },
    );
    wake.frame = (reason) => driver.wake(reason);
    driver.wake("start");
    const unwatchPoses = watchPoses(target.assetId, () => driver.wake("objects"));
    // The native package's splats carry no object ids: the objects panel says so.
    const unlinkInstances = target.assetId
      ? linkScanInstances(target.assetId, backend, true)
      : () => undefined;
    log.info("splat renderer streaming natively", { kind, url: lodUrl });
    return {
      kind,
      key: target.key,
      instances: () => null,
      motion: () => null,
      objects: () => objects.count,
      settled: () => backend.settled?.() ?? true,
      meter: (now) => meter.reading(now),
      // The renderer chooses its own level of detail from its own camera.
      prefetch: () => undefined,
      setFar: (next) => {
        if (next === far) return;
        far = next;
        applyBudget();
        driver.wake("far");
      },
      far: () => far,
      stop: () => {
        unlinkInstances();
        motionLink?.dispose();
        objects.stop();
        unwatchPoses();
        driver.stop();
        work.stop();
        try {
          stream.stop();
          backend.destroy();
        } finally {
          canvas.remove();
        }
      },
      status: () => ({
        tiles: 0,
        gaussians: stream.splats(),
        frames,
        error: null,
        budget: stream.splats(),
        loading: 0,
        cached: 0,
        native: true,
      }),
    };
  }
}
