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
 */

import {
  BoundingSphere,
  Cartesian3,
  Intersect,
  Matrix4,
  type Camera,
  type Cesium3DTileset,
  type Scene,
  type Viewer,
} from "cesium";

import { deviceSplatBudget, deviceSplatCeiling, isHandheld } from "@/lib/detail";
import { AdaptiveSplatBudget } from "@/lib/splatBudget";
import { instancesRefOf } from "@/lib/instances";
import { createLogger } from "@/lib/log";
import { DEFAULT_SPLAT_RENDERER } from "@/state/settings";
import { TileStreamer, type View } from "@/view/stream";
import { parseTileset, type TileNode } from "@/view/tiles";

import { DEDICATED_PRIORITY, registerPickSource } from "../sceneSelect/pickSources";
import { Handover } from "./handover";
import { scanPose } from "./pose";
import { linkScanInstances } from "./scanInstances";
import { ScanMotionLink } from "./scanMotion";
import { ScanObjects } from "./scanObjects";
import type { ScanBackend, ScanPose, SplatRendererKind } from "./types";

const log = createLogger("scan-renderer");

/** The scan a dedicated renderer draws: the site's splat tileset CesiumJS loaded (hidden). */
export interface ScanTarget {
  key: string;
  tileset: Cesium3DTileset;
  /** The scan's asset id: whose objects (state/instances.ts) the renderer draws hidden or lit. */
  assetId?: string;
}

/** How often the tile cut is re-planned while the camera moves (ms), as the viewer page. */
const REPLAN_MS = 150;
/** Tiles fetched at once. */
const FETCHES_AT_ONCE = 3;
/** Most gaussians streamed in at once, whatever the budget. */
const MAX_STREAMED = 10_000_000;
/** Resolution while the camera moves, as a share of the resting one (never below
 *  MIN_MOTION_PIXEL_RATIO), and how long after the last change it counts as resting. */
const MOTION_RESOLUTION = 0.6;
const MIN_MOTION_PIXEL_RATIO = 0.75;
const MOTION_SETTLE_MS = 200;
/** Most gaussians put on screen per re-plan (~4M a second at REPLAN_MS): tiles that land
 *  together go up over a few frames instead of all in one. */
const MAX_SHOWN_PER_UPDATE = 600_000;
/** The next level of detail within this distance is fetched ahead once the view is served. */
const PREFETCH_RADIUS_M = 30;
/** Loaded tiles kept beyond what is drawn, so a look back needs no download. */
const CACHE_FACTOR = 1.5;

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

/** What the page's tests and the debug panel read. */
export interface ScanRendererStatus {
  kind: SplatRendererKind;
  active: boolean;
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
}

interface Session {
  kind: SplatRendererKind;
  key: string;
  stop(): void;
  status(): Omit<ScanRendererStatus, "kind" | "active" | "instances" | "motion" | "objects">;
  instances(): { tiles: number; matched: number } | null;
  motion(): ScanRendererStatus["motion"];
  objects(): number;
}

/** How the host makes its renderers. */
export interface ScanRendererOptions {
  /** Keeps each drawn frame readable after it is shown (harnesses read pixels back). */
  preserveDrawingBuffer?: boolean;
}

interface BackendModule {
  createBackend(
    canvas: HTMLCanvasElement,
    budget: number,
    options?: ScanRendererOptions,
  ): Promise<ScanBackend<unknown>>;
}

/** Frames in a row the overlay waits for the globe's own while the camera moves. */
const MAX_WAIT_FOR_GLOBE = 2;

/**
 * Calls `draw` once a frame, in step with the globe: right after CesiumJS renders (its
 * `postRender`, so the overlay is drawn from exactly the camera the globe was), and on the
 * frames it does not (request-render mode: a still camera) from the animation frame. Drawing
 * only from the animation frame put the overlay a camera pose behind whenever its callback ran
 * before CesiumJS's -- the order of animation-frame callbacks, which a render-loop restart
 * (render-error recovery) flips -- and the scan slid on the map as the view moved.
 */
function driveWithGlobe(scene: Scene, camera: Camera, draw: () => void): () => void {
  const eye = new Cartesian3(Number.NaN, 0, 0);
  const direction = new Cartesian3();
  let globeDrew = false;
  let waited = 0;
  let raf = 0;
  const drawNow = (): void => {
    Cartesian3.clone(camera.positionWC, eye);
    Cartesian3.clone(camera.directionWC, direction);
    draw();
  };
  const removePostRender = scene.postRender.addEventListener(() => {
    globeDrew = true;
    waited = 0;
    drawNow();
  });
  const tick = (): void => {
    raf = requestAnimationFrame(tick);
    if (globeDrew) {
      globeDrew = false;
      return;
    }
    const moved =
      !Cartesian3.equalsEpsilon(camera.positionWC, eye, 0, 1e-3) ||
      !Cartesian3.equalsEpsilon(camera.directionWC, direction, 1e-5);
    // Moved, and the globe has not drawn it yet: it will this frame.
    if (moved && waited++ < MAX_WAIT_FOR_GLOBE) return;
    waited = 0;
    drawNow();
  };
  raf = requestAnimationFrame(tick);
  return () => {
    cancelAnimationFrame(raf);
    removePostRender();
  };
}

function loadBackend(kind: Exclude<SplatRendererKind, "cesium">): Promise<BackendModule> {
  // Each renderer is its own chunk, fetched only when chosen.
  return kind === "spark" ? import("./sparkBackend") : import("./playcanvasBackend");
}

/** The scan tileset's root extras (what it declares: instances, skin, objects, ...). */
function rootExtrasOf(tileset: Cesium3DTileset): unknown {
  return (tileset.root as { extras?: unknown } | undefined)?.extras;
}

/** The scan tileset's root transform as it is declared, column-major (the splats' frame). */
function rootTransformArray(tileset: Cesium3DTileset): number[] {
  const transform = (tileset.root as { transform?: Matrix4 } | undefined)?.transform;
  return Matrix4.toArray(transform ?? Matrix4.IDENTITY);
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

export class ScanRendererHost {
  private kind: SplatRendererKind = DEFAULT_SPLAT_RENDERER;
  private target: ScanTarget | null = null;
  private session: Session | null = null;
  private starting: Promise<void> | null = null;
  private lastError: string | null = null;

  constructor(
    private readonly viewer: Pick<Viewer, "camera" | "canvas" | "scene">,
    private readonly options: ScanRendererOptions = {},
  ) {}

  get renderer(): SplatRendererKind {
    return this.kind;
  }

  setRenderer(kind: SplatRendererKind): void {
    if (kind === this.kind) return;
    this.kind = kind;
    this.sync();
  }

  setTarget(target: ScanTarget | null): void {
    if (target?.key === this.target?.key && target?.tileset === this.target?.tileset) return;
    this.target = target;
    this.sync();
  }

  status(): ScanRendererStatus {
    const inner = this.session?.status() ?? {
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
      active: this.session !== null,
      ...inner,
      instances: this.session?.instances() ?? null,
      motion: this.session?.motion() ?? null,
      objects: this.session?.objects() ?? 0,
      error: inner.error ?? this.lastError,
    };
  }

  destroy(): void {
    this.target = null;
    this.session?.stop();
    this.session = null;
  }

  private sync(): void {
    const wanted = this.kind !== "cesium" && this.target !== null;
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
        // Chosen away from while it started: undone at once.
        if (this.kind !== kind || this.target?.key !== target.key) session.stop();
        else this.session = session;
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

  private async start(
    kind: Exclude<SplatRendererKind, "cesium">,
    target: ScanTarget,
  ): Promise<Session> {
    this.lastError = null;
    const { viewer } = this;
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
    viewer.canvas.insertAdjacentElement("afterend", canvas);
    const budget = deviceSplatBudget();
    let backend: ScanBackend<unknown>;
    try {
      backend = await (await loadBackend(kind)).createBackend(canvas, budget, this.options);
    } catch (error) {
      canvas.remove();
      throw error;
    }
    try {
      return await this.run(kind, target, canvas, backend, budget);
    } catch (error) {
      // Whatever failed before the first frame leaves nothing behind.
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
  ): Promise<Session> {
    const { viewer } = this;
    const url = new URL(target.tileset.resource.url, location.href).toString();
    if (backend.streamNative) {
      const native = await this.runNative(kind, target, canvas, backend, url);
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
    const streamed = streamedFor(adaptive.budget);
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
        budget: streamed,
        cacheBudget: streamed * CACHE_FACTOR,
        concurrency: FETCHES_AT_ONCE,
        maxShownPerUpdate: MAX_SHOWN_PER_UPDATE,
        prefetchRadiusM: PREFETCH_RADIUS_M,
      },
    );
    let arrived = true;
    streamer.onArrival = () => {
      arrived = true;
    };
    // The scan's objects move as the shared drivers move them (scanMotion.ts), and its split
    // objects are drawn where their poses put them (scanObjects.ts).
    const extras = rootExtrasOf(target.tileset);
    const motionLink = target.assetId
      ? new ScanMotionLink(target.assetId, backend, { native: false, extras })
      : null;
    const objects = new ScanObjects<unknown>(backend, target.assetId);

    const toLocal = new Matrix4();
    const toWorld = new Matrix4();
    const lastEye = new Cartesian3(Number.NaN, 0, 0);
    const lastDirection = new Cartesian3();
    const scratchCentre = new Cartesian3();
    const scratchSphere = new BoundingSphere();
    let lastPlan = 0;
    // Motion frames, for the adaptive budget: the camera moved since the last frame.
    const frameEye = new Cartesian3(Number.NaN, 0, 0);
    const frameDirection = new Cartesian3();
    let lastFrameAt = 0;
    let lastMotionAt = 0;
    let frames = 0;
    let stopDriving: (() => void) | null = null;
    const pixelRatio = Math.min(window.devicePixelRatio || 1, isHandheld() ? 1.5 : 2);

    const view = (pose: ScanPose): View => {
      const camera = viewer.camera;
      const culling = camera.frustum.computeCullingVolume(
        camera.positionWC,
        camera.directionWC,
        camera.upWC,
      );
      return {
        eye: pose.eye,
        projection: pose.height / (2 * Math.tan(pose.fovy / 2)),
        visible: (bounds) => {
          Matrix4.multiplyByPoint(
            toWorld,
            Cartesian3.fromArray(bounds.center, 0, scratchCentre),
            scratchSphere.center,
          );
          scratchSphere.radius = bounds.radius;
          return culling.computeVisibility(scratchSphere) !== Intersect.OUTSIDE;
        },
      };
    };

    const tick = (): void => {
      const tileset = target.tileset;
      if (tileset.isDestroyed()) return;
      Matrix4.clone(tileset.root.computedTransform, toWorld);
      Matrix4.inverseTransformation(toWorld, toLocal);
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
      // is per pixel, and the GPU was what ran out), the full resolution the moment it rests.
      const moving = now - lastMotionAt < MOTION_SETTLE_MS;
      const pose = scanPose(viewer.camera, toLocal, {
        width: viewer.canvas.clientWidth,
        height: viewer.canvas.clientHeight,
        pixelRatio: moving
          ? Math.max(MIN_MOTION_PIXEL_RATIO, pixelRatio * MOTION_RESOLUTION)
          : pixelRatio,
      });
      const drawn = Math.min(streamer.drawnGaussians, adaptive.budget);
      // Frames while tiles arrive are slowed by their uploads, not by what is drawn: only a
      // steady view's motion frames say what the GPU can sort and blend.
      const steady = motion && streamer.loading === 0 && lastFrameAt > 0;
      if (steady && adaptive.frame(now - lastFrameAt, drawn)) {
        backend.setBudget(adaptive.budget);
        const next = streamedFor(adaptive.budget);
        streamer.setBudget(next, next * CACHE_FACTOR);
        arrived = true;
        log.info("splat budget", { kind, budget: adaptive.budget });
      }
      lastFrameAt = now;
      if ((arrived || moved) && now - lastPlan >= REPLAN_MS) {
        lastPlan = now;
        arrived = false;
        Cartesian3.clone(camera.positionWC, lastEye);
        Cartesian3.clone(camera.directionWC, lastDirection);
        streamer.update(view(pose));
      }
      motionLink?.update();
      objects.tick();
      backend.render(pose);
      handover.tick(performance.now());
      frames += 1;
    };

    // The root first, as the viewer page: the whole scan, coarse, while the rest streams.
    let root: unknown;
    try {
      root = await backend.load(url, tree.root);
    } catch (reason) {
      streamer.stop();
      throw reason;
    }
    handover.show(root, performance.now());
    streamer.adopt(tree.root, root);
    void objects.load(url, extras, rootTransformArray(target.tileset));
    stopDriving = driveWithGlobe(viewer.scene, viewer.camera, tick);
    const unlinkInstances = target.assetId
      ? linkScanInstances(target.assetId, backend, false)
      : () => undefined;
    // Scene selection picks from the tiles this renderer draws (cesium/sceneSelect).
    const unlinkPick =
      target.assetId && backend.pickTiles
        ? registerPickSource(
            target.assetId,
            {
              renderer: kind,
              tiles: () => backend.pickTiles?.() ?? [],
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
      stop: () => {
        unlinkPick();
        unlinkInstances();
        motionLink?.dispose();
        objects.stop();
        stopDriving?.();
        streamer.stop();
        backend.destroy();
        canvas.remove();
      },
      instances: () => backend.instanceTiles?.() ?? null,
      motion: () => motionStatus(motionLink, backend),
      objects: () => objects.count,
      status: () => ({
        tiles: streamer.drawn.length,
        gaussians: streamer.drawnGaussians,
        frames,
        error,
        budget: adaptive.budget,
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
   * a scan with objects (`extras.instances`): the native package's splats carry no object ids,
   * so the objects could not be hidden or highlighted there (the published camp, which has
   * both, drew every object whatever the panel said); its 3D Tiles carry them by checksum.
   */
  private async runNative(
    kind: Exclude<SplatRendererKind, "cesium">,
    target: ScanTarget,
    canvas: HTMLCanvasElement,
    backend: ScanBackend<unknown>,
    tilesetUrl: string,
  ): Promise<Session | null> {
    if (!backend.streamNative) return null;
    const extras = rootExtrasOf(target.tileset);
    if (target.assetId && instancesRefOf(extras) !== null) return null;
    const lodUrl = await findNativeLod(tilesetUrl, extras);
    if (lodUrl === null) return null;
    const stream = await backend.streamNative(lodUrl);
    const { viewer } = this;
    const toLocal = new Matrix4();
    const pixelRatio = Math.min(window.devicePixelRatio || 1, isHandheld() ? 1.5 : 2);
    const lastEye = new Cartesian3(Number.NaN, 0, 0);
    const lastDirection = new Cartesian3();
    let lastMotionAt = 0;
    let frames = 0;
    const tick = (): void => {
      const tileset = target.tileset;
      if (tileset.isDestroyed()) return;
      Matrix4.inverseTransformation(tileset.root.computedTransform, toLocal);
      const camera = viewer.camera;
      const now = performance.now();
      if (
        !Cartesian3.equalsEpsilon(camera.positionWC, lastEye, 0, 1e-3) ||
        !Cartesian3.equalsEpsilon(camera.directionWC, lastDirection, 1e-5)
      ) {
        lastMotionAt = now;
        Cartesian3.clone(camera.positionWC, lastEye);
        Cartesian3.clone(camera.directionWC, lastDirection);
      }
      const moving = now - lastMotionAt < MOTION_SETTLE_MS;
      backend.render(
        scanPose(camera, toLocal, {
          width: viewer.canvas.clientWidth,
          height: viewer.canvas.clientHeight,
          pixelRatio: moving
            ? Math.max(MIN_MOTION_PIXEL_RATIO, pixelRatio * MOTION_RESOLUTION)
            : pixelRatio,
        }),
      );
      frames += 1;
    };
    // The native package carries no tile checksums: no object ids, no skins (the panels say
    // so); split objects are tiles of their own, drawn as in any session.
    const motion = target.assetId
      ? new ScanMotionLink(target.assetId, backend, { native: true, extras })
      : null;
    const objects = new ScanObjects<unknown>(backend, target.assetId);
    void objects.load(tilesetUrl, extras, rootTransformArray(target.tileset));
    const stopDriving = driveWithGlobe(viewer.scene, viewer.camera, () => {
      objects.tick();
      tick();
    });
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
      stop: () => {
        unlinkInstances();
        motion?.dispose();
        objects.stop();
        stopDriving();
        stream.stop();
        backend.destroy();
        canvas.remove();
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
