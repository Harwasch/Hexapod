/**
 * A scan's tiles streamed into Spark (three.js) with the view: shared by the scan viewer page
 * (main.ts) and the globe's Spark splat renderer (cesium/SparkSplatOverlay.ts), so both draw a
 * tileset the same way.
 */

import { fetchSplatResource } from "@/lib/splatFetch";
import { type SparkRenderer, SplatFileType, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";

import { tileUrl } from "@/lib/tileProxy";
import { deviceSplatBudget, isHandheld } from "@/lib/detail";
import { maxShDegree } from "@/lib/shDegree";

import { spzFromGlb } from "./glb";
import { TileStreamer, type View } from "./stream";
import { LOAD_FACTOR, parseTileset, planAdditive, type TileNode, type TileTree } from "./tiles";

/**
 * One tile as a Spark mesh: the SPZ inside the tile's GLB, handed over as it is, with Spark's
 * level of detail on. `lod: true` has Spark build a merged LoD tree over the tile in a worker
 * ("quick" tiny-lod: 1-3 s per million splats, per Spark's lod-getting-started docs; a 100k
 * tile is a fraction of a second) and draw from it within `SparkRenderer.lodSplatCount`,
 * which is shared by every mesh in the scene. The SPZ bytes come back too, for a page that
 * reads the tile's splat centres.
 *
 * The mesh is placed at the tile's frame (`TileNode.offset`, zero but for a tile the packer
 * moved because the scan is wider than SPZ's +/-2048 m), so it sits in the scan's frame like
 * every other tile. Centres are kept as 32-bit floats (`extSplats`) unless asked otherwise:
 * Spark's packed splats hold them as half floats, 6 cm apart 64-128 m out and a metre apart
 * a few kilometres out, which a large scan's tiles are.
 */
export async function loadSplatTile(
  tilesetUrl: string,
  tile: TileNode,
  options: { lod?: boolean; extSplats?: boolean; signal?: AbortSignal } = {},
): Promise<{ mesh: SplatMesh; bytes: Uint8Array }> {
  const response = await fetchSplatResource(
    new URL(tile.uri, new URL(tilesetUrl, location.href)).toString(),
    {
      signal: options.signal,
    },
  );
  if (!response.ok) throw new Error(`The scan's data answered ${String(response.status)}.`);
  const bytes = spzFromGlb(await response.arrayBuffer());
  const mesh = new SplatMesh({
    fileBytes: bytes,
    fileType: SplatFileType.SPZ,
    lod: options.lod ?? true,
    extSplats: options.extSplats ?? true,
  });
  try {
    await mesh.initialized;
  } catch (error) {
    mesh.dispose();
    throw error;
  }
  const cap = maxShDegree(isHandheld());
  if (mesh.maxSh > cap) {
    mesh.maxSh = cap;
    mesh.updateGenerator();
  }
  mesh.position.set(...(tile.offset ?? [0, 0, 0]));
  return { mesh, bytes };
}

export interface Streaming {
  readonly settled: boolean;
  /** Once a frame, before rendering: re-plans when the camera moved or a tile arrived. */
  frame(): void;
  stop(): void;
}

/** How often the cut is re-planned while the camera keeps moving. */
const REPLAN_MS = 150;
/** Tiles fetched at once: enough to keep a connection busy, few enough that the nearest
 *  ones are not queued behind a dozen others when the camera turns. */
const FETCHES_AT_ONCE = 3;
/** Most gaussians streamed in at once, whatever the budget: a desktop's 3M draw budget times
 *  LOAD_FACTOR would hold 12M, which is past what Spark's LoD trees want in memory. */
const MAX_STREAMED = 6_000_000;
/** Loaded tiles kept beyond what is drawn, so a look back needs no download. */
const CACHE_FACTOR = 1.5;

/**
 * Streams a REPLACE scan with the camera (stream.ts): the cut follows the view within
 * LOAD_FACTOR times the Detail budget. Re-planning is a pass over a few hundred tiles, so it
 * runs at most every REPLAN_MS and only when the camera moved or a tile arrived; fetching
 * and decoding happen off the frame (fetch, then Spark's worker), so the camera never waits
 * for them -- what is on screen stays until what replaces it is ready.
 */
export function streamWithView(
  tileset: { url: string; tree: TileTree },
  scan: THREE.Group,
  camera: THREE.PerspectiveCamera,
  renderer: THREE.WebGLRenderer,
  budget: number,
  first: { root: TileNode; mesh: SplatMesh },
  display: {
    show: (tile: TileNode, mesh: SplatMesh) => void;
    hide: (tile: TileNode, mesh: SplatMesh) => void;
  },
  report: (drawn: { tiles: TileNode[]; gaussians: number }) => void,
  /** How a tile becomes a mesh, when the page wants more than the mesh (its splat centres). */
  loadMesh?: (tile: TileNode) => Promise<SplatMesh>,
  fullResolution = false,
): Streaming {
  const streamer = new TileStreamer<SplatMesh>(
    tileset.tree,
    {
      load: (tile) =>
        loadMesh ? loadMesh(tile) : loadSplatTile(tileset.url, tile).then((t) => t.mesh),
      show: display.show,
      hide: display.hide,
      dispose: (mesh) => mesh.dispose(),
      failed: (tile) => console.warn(`Tile ${tile.uri} did not load; its parent stays.`),
    },
    {
      budget: fullResolution ? Infinity : Math.min(budget * LOAD_FACTOR, MAX_STREAMED),
      targetErrorPx: fullResolution ? -Infinity : undefined,
      cacheBudget: fullResolution
        ? Infinity
        : Math.min(budget * LOAD_FACTOR, MAX_STREAMED) * CACHE_FACTOR,
      concurrency: FETCHES_AT_ONCE,
    },
  );
  streamer.adopt(first.root, first.mesh);
  const frustum = new THREE.Frustum();
  const matrix = new THREE.Matrix4();
  const toScan = new THREE.Matrix4();
  const sphere = new THREE.Sphere();
  const eye = new THREE.Vector3();
  const lastPose = new THREE.Matrix4();
  let arrived = true;
  let lastPlan = 0;
  streamer.onArrival = () => {
    arrived = true;
  };
  const view = (): View => {
    scan.updateMatrixWorld();
    camera.updateMatrixWorld();
    toScan.copy(scan.matrixWorld).invert();
    eye.copy(camera.position).applyMatrix4(toScan);
    frustum.setFromProjectionMatrix(
      matrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse),
    );
    const height = renderer.domElement.clientHeight || window.innerHeight;
    return {
      eye: [eye.x, eye.y, eye.z],
      projection: (height * (camera.projectionMatrix.elements[5] ?? 1)) / 2,
      visible: (bounds) => {
        sphere.center.set(...bounds.center).applyMatrix4(scan.matrixWorld);
        sphere.radius = bounds.radius;
        return frustum.intersectsSphere(sphere);
      },
    };
  };
  return {
    get settled() {
      return !arrived && streamer.settled;
    },
    frame: () => {
      const now = performance.now();
      const moved = !lastPose.equals(camera.matrixWorld);
      if (!(arrived || moved) || now - lastPlan < REPLAN_MS) return;
      lastPlan = now;
      arrived = false;
      lastPose.copy(camera.matrixWorld);
      if (streamer.update(view()) || moved) {
        report({ tiles: streamer.drawn, gaussians: streamer.drawnGaussians });
      }
    },
    stop: () => streamer.stop(),
  };
}

/** Explicit Spark defaults, shared with training runs that use gsplat's AA filter. */
export function splatAntialiasing(antialiased?: boolean | null): {
  preBlurAmount: number;
  blurAmount: number;
} {
  return antialiased === false
    ? { preBlurAmount: 0.3, blurAmount: 0 }
    : { preBlurAmount: 0, blurAmount: 0.3 };
}

export async function fetchSplatTileset(
  source: string,
  signal?: AbortSignal,
): Promise<{ url: string; tree: TileTree }> {
  const url = await tileUrl(source);
  const response = await fetchSplatResource(url, { signal });
  if (!response.ok) throw new Error(`The tileset answered ${String(response.status)}.`);
  return { url, tree: parseTileset(await response.json()) };
}

/**
 * One scan's loading lifecycle, shared by the gallery and training console. The parent
 * group owns the coordinate frame. parseTileset deliberately removes the root's ECEF
 * placement; decoded SPZ positions retain the input PLY frame, plus child offsets.
 * The gallery turns ENU z-up to y-up; a solve-local caller attaches directly to its frame.
 */
export function openScanStream(
  source: string | { url: string; tree: TileTree },
  scan: THREE.Group,
  camera: THREE.PerspectiveCamera,
  renderer: THREE.WebGLRenderer,
  spark: SparkRenderer,
  display: {
    /** Diagnostic only: all leaves, no tile budget or per-mesh LOD. */
    fullResolution?: boolean;
    show?: (tile: TileNode, mesh: SplatMesh) => void;
    hide?: (tile: TileNode, mesh: SplatMesh) => void;
    report?: (drawn: { tiles: TileNode[]; gaussians: number }) => void;
    load?: (url: string, tile: TileNode, signal: AbortSignal) => Promise<SplatMesh>;
  } = {},
): Streaming & {
  ready: Promise<{ tileset: { url: string; tree: TileTree }; mesh: SplatMesh }>;
} {
  const abort = new AbortController();
  const budget = spark.lodSplatCount ?? deviceSplatBudget();
  let stopped = false;
  let complete = false;
  let streaming: Streaming | null = null;
  const additive = new Map<TileNode, SplatMesh>();
  const show = (tile: TileNode, mesh: SplatMesh): void => {
    scan.add(mesh);
    display.show?.(tile, mesh);
  };
  const hide = (tile: TileNode, mesh: SplatMesh): void => {
    scan.remove(mesh);
    display.hide?.(tile, mesh);
  };
  const report = (tiles: TileNode[]): void =>
    display.report?.({
      tiles,
      gaussians: tiles.reduce(
        (sum, tile) => sum + (Number.isFinite(tile.gaussians) ? tile.gaussians : 0),
        0,
      ),
    });
  const ready = (async () => {
    const tileset =
      typeof source === "string" ? await fetchSplatTileset(source, abort.signal) : source;
    const load = async (tile: TileNode): Promise<SplatMesh> => {
      const mesh = display.load
        ? await display.load(tileset.url, tile, abort.signal)
        : (
            await loadSplatTile(tileset.url, tile, {
              signal: abort.signal,
              lod: !display.fullResolution,
            })
          ).mesh;
      if (stopped) {
        mesh.dispose();
        throw new DOMException("Scan closed", "AbortError");
      }
      return mesh;
    };
    const root = tileset.tree.root;
    const mesh = await load(root);
    show(root, mesh);
    report([root]);
    if (tileset.tree.refine === "REPLACE") {
      streaming = streamWithView(
        tileset,
        scan,
        camera,
        renderer,
        budget,
        { root, mesh },
        { show, hide },
        (drawn) => display.report?.(drawn),
        load,
        display.fullResolution,
      );
    } else {
      additive.set(root, mesh);
      const [, ...rest] = planAdditive(
        tileset.tree,
        display.fullResolution ? Infinity : budget * LOAD_FACTOR,
      );
      void (async () => {
        for (const step of rest) {
          for (const tile of step.add) {
            if (stopped) return;
            const more = await load(tile);
            additive.set(tile, more);
            show(tile, more);
          }
          report([...additive.keys()]);
        }
        complete = true;
      })().catch(() => {
        /* Keep the loaded overview if a later ADD tile fails. */
      });
    }
    return { tileset, mesh };
  })();
  return {
    get settled() {
      return streaming?.settled ?? complete;
    },
    ready,
    frame: () => streaming?.frame(),
    stop: () => {
      stopped = true;
      abort.abort();
      streaming?.stop();
      for (const [tile, mesh] of additive) {
        hide(tile, mesh);
        mesh.dispose();
      }
      additive.clear();
    },
  };
}
