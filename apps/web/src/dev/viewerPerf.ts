/** Opt-in build entry only. Uses the console's backend, without the globe or UI. */
import { Application } from "playcanvas";

import { createBackend } from "@/cesium/scanView/playcanvasBackend";
import { maxShDegree } from "@/cesium/scanView/quality";
import { TileWork } from "@/cesium/scanView/tileWork";
import type { ScanPose } from "@/cesium/scanView/types";
import { parseTileset, type Sphere, type TileNode, type TileTree } from "@/view/tiles";

/** A full-detail cut: never draw REPLACE parents on top of their children. */
export function fullDetail(tree: TileTree): TileNode[] {
  const tiles: TileNode[] = [];
  const visit = (tile: TileNode): void => {
    if (tile.uri && (tree.refine === "ADD" || tile.children.length === 0)) tiles.push(tile);
    tile.children.forEach(visit);
  };
  visit(tree.root);
  return tiles;
}

/** Z-up scan frame; complete orbit at 3 radii, then dolly to 0.15 radii. */
export function cameraPath(bounds: Sphere, orbit: number, dolly: number): ScanPose[] {
  if (!Number.isInteger(orbit) || orbit < 2 || !Number.isInteger(dolly) || dolly < 1)
    throw new Error("The path needs at least two orbit frames and one dolly frame.");
  if (!Number.isFinite(bounds.radius) || bounds.radius <= 0)
    throw new Error("The scan needs a positive bounding radius.");
  return Array.from({ length: orbit + dolly }, (_, i) => {
    const angle = i < orbit ? (i / (orbit - 1)) * Math.PI * 2 : Math.PI * 2;
    const radii = i < orbit ? 3 : 3 - (2.85 * (i - orbit + 1)) / dolly;
    const elevation = Math.PI / 9;
    const direction: [number, number, number] = [
      -Math.cos(angle) * Math.cos(elevation),
      -Math.sin(angle) * Math.cos(elevation),
      -Math.sin(elevation),
    ];
    const eye = bounds.center.map(
      (v, axis) => v - (direction[axis] ?? 0) * radii * bounds.radius,
    ) as [number, number, number];
    return {
      eye,
      direction,
      up: [0, 0, 1],
      fovy: Math.PI / 3,
      near: bounds.radius * 0.001,
      far: bounds.radius * 10,
      width: window.innerWidth,
      height: window.innerHeight,
      pixelRatio: 1,
    };
  });
}

export interface PerfOptions {
  tilesetUrl: string;
  degree: number;
  warmup: number;
  orbit: number;
  dolly: number;
  timeoutMs: number;
  /** Finish the GPU in each frame callback: serialized CPU + GPU cost, no pipelining. */
  sync?: boolean;
}

// The callback's own clock, not rAF's argument: with the frame-rate limit off (headless,
// uncapped) consecutive callbacks can carry the same frame timestamp.
const frame = (): Promise<number> =>
  new Promise((resolve) => requestAnimationFrame(() => resolve(performance.now())));

export async function prepare(canvas: HTMLCanvasElement, options: PerfOptions) {
  const response = await fetch(options.tilesetUrl);
  if (!response.ok) throw new Error(`Tileset HTTP ${response.status}`);
  const tree = parseTileset(await response.json());
  const bounds = tree.root.bounds;
  if (!bounds) throw new Error("The tileset needs a local box or sphere.");
  const path = cameraPath(bounds, options.orbit, options.dolly);
  const first = path[0];
  if (!first) throw new Error("Empty camera path");
  const tiles = fullDetail(tree);
  const budget = tiles.reduce((sum, tile) => sum + tile.gaussians, 0);
  if (!Number.isFinite(budget) || budget <= 0) throw new Error("Missing gaussian counts");
  const work = new TileWork();
  const backend = await createBackend(canvas, budget, {
    frameWanted: () => undefined,
    work,
    maxShDegree: maxShDegree(false),
  });
  // The backend uses this same event internally but exposes no public settled getter.
  // Subscribe only in this benchmark entry; ordinary app behaviour is unchanged.
  const system = Application.getApplication(canvas.id)?.systems.gsplat;
  if (!system) throw new Error("Missing PlayCanvas splat system");
  let idle = false;
  system.on("frame:ready", (_camera: unknown, _layer: unknown, ready: boolean, loading: number) => {
    idle = ready && loading === 0;
  });
  const gl = canvas.getContext("webgl2");
  if (!gl) throw new Error("No WebGL2 context on the renderer canvas");
  const info = gl.getExtension("WEBGL_debug_renderer_info");
  const renderer = String(gl.getParameter(info ? info.UNMASKED_RENDERER_WEBGL : gl.RENDERER));
  // Sequential loads bound worker/upload pressure; all loads finish before timing.
  const meshes = [];
  for (const tile of tiles) {
    const mesh = await backend.load(options.tilesetUrl, tile);
    const resource = mesh.gsplat?.resource as unknown as { shBands: number; numSplats: number };
    if (resource.shBands !== options.degree || resource.numSplats !== tile.gaussians)
      throw new Error(`Decoded tile differs from requested SH ${options.degree} / count`);
    backend.add(mesh);
    meshes.push(mesh);
  }
  const loadedAt = performance.now();
  let stable = 0;
  while (stable < 5) {
    if (performance.now() - loadedAt > options.timeoutMs)
      throw new Error("PlayCanvas did not finish loading/sorting/drawing");
    await frame();
    backend.render(first);
    const drawn = meshes.every((mesh) => backend.isDrawn(mesh, performance.now() - loadedAt));
    stable = work.queued === 0 && drawn && idle ? stable + 1 : 0;
  }
  for (let i = 0; i < options.warmup; i++) {
    await frame();
    backend.render(first);
  }
  return {
    renderer,
    bounds,
    splats: budget,
    tiles: tiles.length,
    shDegree: options.degree,
    maxShDegree: maxShDegree(false),
    pixelRatio: 1,
    async measure() {
      const timer = gl.getExtension("EXT_disjoint_timer_query_webgl2") as {
        TIME_ELAPSED_EXT: number;
        GPU_DISJOINT_EXT: number;
      } | null;
      const queries: WebGLQuery[] = [];
      let disjoint = false;
      // Uncapped headless frames are pipelined and PlayCanvas draws splats in its own tick,
      // outside `backend.render`: finishing the GPU at the start of each callback makes the
      // interval everything one frame queued, CPU and GPU, whatever drew it.
      const tick = options.sync
        ? (): Promise<number> =>
            new Promise((resolve) =>
              requestAnimationFrame(() => {
                gl.finish();
                resolve(performance.now());
              }),
            )
        : frame;
      const timestamps: number[] = [await tick()];
      for (const pose of path) {
        const query = timer ? gl.createQuery() : null;
        if (timer && query) gl.beginQuery(timer.TIME_ELAPSED_EXT, query);
        backend.render(pose);
        if (timer && query) {
          gl.endQuery(timer.TIME_ELAPSED_EXT);
          queries.push(query);
        }
        timestamps.push(await tick());
        if (gl.isContextLost()) throw new Error("WebGL context lost during measurement");
        if (timer && gl.getParameter(timer.GPU_DISJOINT_EXT)) disjoint = true;
      }
      // Read after the measured path, never force GPU completion inside a timed frame.
      const until = performance.now() + options.timeoutMs;
      while (
        timer &&
        !disjoint &&
        queries.some((q) => !gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE))
      ) {
        if (performance.now() > until) break;
        await frame();
        if (gl.getParameter(timer.GPU_DISJOINT_EXT)) disjoint = true;
      }
      const valid =
        timer &&
        !disjoint &&
        queries.length === path.length &&
        queries.every((q) => gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE));
      const gpuMs = valid
        ? queries.map((q) => Number(gl.getQueryParameter(q, gl.QUERY_RESULT)) / 1e6)
        : null;
      for (const query of queries) gl.deleteQuery(query);
      return {
        timestamps,
        orbit: options.orbit,
        dolly: options.dolly,
        gpuMs,
        gpuNote: valid
          ? "Timer queries around backend.render; excludes asynchronous CPU sorting."
          : timer
            ? "GPU queries disjoint, incomplete or timed out; discarded."
            : "EXT_disjoint_timer_query_webgl2 unavailable.",
      };
    },
    destroy() {
      backend.destroy();
      work.stop();
    },
  };
}
