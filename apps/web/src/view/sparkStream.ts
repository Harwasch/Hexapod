/**
 * A scan's tiles streamed into Spark (three.js) with the view: shared by the scan viewer page
 * (main.ts) and the globe's Spark splat renderer (cesium/SparkSplatOverlay.ts), so both draw a
 * tileset the same way.
 */

import { SplatFileType, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";

import { spzFromGlb } from "./glb";
import { TileStreamer, type View } from "./stream";
import { LOAD_FACTOR, type TileNode, type TileTree } from "./tiles";

/**
 * One tile as a Spark mesh: the SPZ inside the tile's GLB, handed over as it is, with Spark's
 * level of detail on. `lod: true` has Spark build a merged LoD tree over the tile in a worker
 * ("quick" tiny-lod: 1-3 s per million splats, per Spark's lod-getting-started docs; a 100k
 * tile is a fraction of a second) and draw from it within `SparkRenderer.lodSplatCount`,
 * which is shared by every mesh in the scene. The SPZ bytes come back too, for a page that
 * reads the tile's splat centres.
 */
export async function loadSplatTile(
  tilesetUrl: string,
  tile: TileNode,
  options: { lod?: boolean; extSplats?: boolean; signal?: AbortSignal } = {},
): Promise<{ mesh: SplatMesh; bytes: Uint8Array }> {
  const response = await fetch(new URL(tile.uri, new URL(tilesetUrl, location.href)).toString(), {
    signal: options.signal,
  });
  if (!response.ok) throw new Error(`The scan's data answered ${String(response.status)}.`);
  const bytes = spzFromGlb(await response.arrayBuffer());
  const mesh = new SplatMesh({
    fileBytes: bytes,
    fileType: SplatFileType.SPZ,
    lod: options.lod ?? true,
    extSplats: options.extSplats ?? false,
  });
  await mesh.initialized;
  return { mesh, bytes };
}

export interface Streaming {
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
      budget: Math.min(budget * LOAD_FACTOR, MAX_STREAMED),
      cacheBudget: Math.min(budget * LOAD_FACTOR, MAX_STREAMED) * CACHE_FACTOR,
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
      projection: height / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))),
      visible: (bounds) => {
        sphere.center.set(...bounds.center).applyMatrix4(scan.matrixWorld);
        sphere.radius = bounds.radius;
        return frustum.intersectsSphere(sphere);
      },
    };
  };
  return {
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
