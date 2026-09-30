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
  type Cesium3DTileset,
  type Viewer,
} from "cesium";

import { deviceSplatBudget, isHandheld } from "@/lib/detail";
import { createLogger } from "@/lib/log";
import { TileStreamer, type View } from "@/view/stream";
import { parseTileset, type TileNode } from "@/view/tiles";

import { scanPose } from "./pose";
import type { ScanBackend, ScanPose, SplatRendererKind } from "./types";

const log = createLogger("scan-renderer");

/** The scan a dedicated renderer draws: the site's splat tileset CesiumJS loaded (hidden). */
export interface ScanTarget {
  key: string;
  tileset: Cesium3DTileset;
}

/** How often the tile cut is re-planned while the camera moves (ms), as the viewer page. */
const REPLAN_MS = 150;
/** Tiles fetched at once. */
const FETCHES_AT_ONCE = 3;
/** Most gaussians streamed in at once, whatever the budget (as the viewer page). */
const MAX_STREAMED = 6_000_000;
/** Loaded tiles kept beyond what is drawn, so a look back needs no download. */
const CACHE_FACTOR = 1.5;

/** What the page's tests and the debug panel read. */
export interface ScanRendererStatus {
  kind: SplatRendererKind;
  active: boolean;
  tiles: number;
  gaussians: number;
  frames: number;
  error: string | null;
}

interface Session {
  kind: SplatRendererKind;
  key: string;
  stop(): void;
  status(): Omit<ScanRendererStatus, "kind" | "active">;
}

interface BackendModule {
  createBackend(canvas: HTMLCanvasElement, budget: number): Promise<ScanBackend<unknown>>;
}

function loadBackend(kind: Exclude<SplatRendererKind, "cesium">): Promise<BackendModule> {
  // Each renderer is its own chunk, fetched only when chosen.
  return kind === "spark" ? import("./sparkBackend") : import("./playcanvasBackend");
}

export class ScanRendererHost {
  private kind: SplatRendererKind = "cesium";
  private target: ScanTarget | null = null;
  private session: Session | null = null;
  private starting: Promise<void> | null = null;
  private lastError: string | null = null;

  constructor(private readonly viewer: Pick<Viewer, "camera" | "canvas">) {}

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
    const inner = this.session?.status() ?? { tiles: 0, gaussians: 0, frames: 0, error: null };
    return {
      kind: this.kind,
      active: this.session !== null,
      ...inner,
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
      backend = await (await loadBackend(kind)).createBackend(canvas, budget);
    } catch (error) {
      canvas.remove();
      throw error;
    }
    const url = new URL(target.tileset.resource.url, location.href).toString();
    const response = await fetch(url);
    if (!response.ok) {
      backend.destroy();
      canvas.remove();
      throw new Error(`The scan's tileset answered ${String(response.status)}.`);
    }
    const tree = parseTileset(await response.json());
    let error: string | null = null;
    const streamed = Math.min(budget * backend.loadFactor, MAX_STREAMED);
    const streamer = new TileStreamer<unknown>(
      tree,
      {
        load: (tile) => backend.load(url, tile),
        show: (_tile, mesh) => backend.add(mesh),
        hide: (_tile, mesh) => backend.remove(mesh),
        dispose: (mesh) => backend.dispose(mesh),
        failed: (tile: TileNode, reason: unknown) => {
          error = reason instanceof Error ? reason.message : String(reason);
          log.warn("scan tile did not load; its parent stays", { tile: tile.uri, error });
        },
      },
      { budget: streamed, cacheBudget: streamed * CACHE_FACTOR, concurrency: FETCHES_AT_ONCE },
    );
    let arrived = true;
    streamer.onArrival = () => {
      arrived = true;
    };

    const toLocal = new Matrix4();
    const toWorld = new Matrix4();
    const lastEye = new Cartesian3(Number.NaN, 0, 0);
    const lastDirection = new Cartesian3();
    const scratchCentre = new Cartesian3();
    const scratchSphere = new BoundingSphere();
    let lastPlan = 0;
    let frames = 0;
    let running = true;
    let raf = 0;
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
      if (!running) return;
      raf = requestAnimationFrame(tick);
      const tileset = target.tileset;
      if (tileset.isDestroyed()) return;
      Matrix4.clone(tileset.root.computedTransform, toWorld);
      Matrix4.inverseTransformation(toWorld, toLocal);
      const pose = scanPose(viewer.camera, toLocal, {
        width: viewer.canvas.clientWidth,
        height: viewer.canvas.clientHeight,
        pixelRatio,
      });
      const camera = viewer.camera;
      const moved =
        !Cartesian3.equalsEpsilon(camera.positionWC, lastEye, 0, 1e-3) ||
        !Cartesian3.equalsEpsilon(camera.directionWC, lastDirection, 1e-5);
      const now = performance.now();
      if ((arrived || moved) && now - lastPlan >= REPLAN_MS) {
        lastPlan = now;
        arrived = false;
        Cartesian3.clone(camera.positionWC, lastEye);
        Cartesian3.clone(camera.directionWC, lastDirection);
        streamer.update(view(pose));
      }
      backend.render(pose);
      frames += 1;
    };

    // The root first, as the viewer page: the whole scan, coarse, while the rest streams.
    const root = await backend.load(url, tree.root);
    backend.add(root);
    streamer.adopt(tree.root, root);
    raf = requestAnimationFrame(tick);
    log.info("splat renderer started", { kind, tiles: tree.root.uri });

    return {
      kind,
      key: target.key,
      stop: () => {
        running = false;
        cancelAnimationFrame(raf);
        streamer.stop();
        backend.destroy();
        canvas.remove();
      },
      status: () => ({
        tiles: streamer.drawn.length,
        gaussians: streamer.drawnGaussians,
        frames,
        error,
      }),
    };
  }
}
