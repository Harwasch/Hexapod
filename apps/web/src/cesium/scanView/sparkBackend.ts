/**
 * The Spark back-end of the scan renderer (ScanRendererHost): three.js with Spark's splat
 * renderer drawing the tiles the streamer chose, as they are.
 *
 * Two things Spark does by default are off here:
 *
 * - **Its own level of detail.** `lod: true` had Spark build a merged tree over every tile and
 *   draw at most `lodSplatCount` splats from all of them, which put a second, coarser level of
 *   detail on top of the tileset's own: blocks of merged, grid-aligned splats where the
 *   tileset had sent the real ones, and a tree to build (one tile at a time, in Spark's
 *   worker) before a new tile could show. The tileset already is a level-of-detail hierarchy,
 *   and the streamer picks from it within the budget, so Spark draws what it is given.
 * - **16-bit positions.** Spark's packed splats keep centres as half floats, which are 6 cm
 *   apart at 64-128 m from the origin -- and a tile's splats are in the scan's frame, up to a
 *   hundred metres out: a visible lattice. `extSplats` keeps them as 32-bit floats.
 */

import { SparkRenderer, type SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";

import { loadSplatTile } from "@/view/sparkStream";

import type { ScanBackend, ScanPose } from "./types";

export function createBackend(canvas: HTMLCanvasElement): Promise<ScanBackend<SplatMesh>> {
  const renderer = new THREE.WebGLRenderer({
    canvas,
    alpha: true,
    antialias: false,
    premultipliedAlpha: true,
    powerPreference: "high-performance",
  });
  renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  const spark = new SparkRenderer({ renderer, enableLod: false });
  scene.add(spark);
  const camera = new THREE.PerspectiveCamera();
  const target = new THREE.Vector3();
  const direction = new THREE.Vector3();
  let size = { width: 0, height: 0, pixelRatio: 0 };

  const backend: ScanBackend<SplatMesh> = {
    name: "spark",
    loadFactor: 1,
    load: async (tilesetUrl, tile, signal) =>
      (await loadSplatTile(tilesetUrl, tile, { lod: false, extSplats: true, signal })).mesh,
    add: (mesh) => scene.add(mesh),
    remove: (mesh) => scene.remove(mesh),
    dispose: (mesh) => mesh.dispose(),
    // Spark shows a new mesh once a sort that includes it has run: until then the displayed
    // mapping holds none of its splats.
    isDrawn: (mesh) =>
      spark.display.mapping.some(
        (entry) => (entry.node === mesh || entry.node.parent === mesh) && entry.count > 0,
      ),
    fade: (mesh, alpha) => {
      mesh.opacity = alpha;
    },
    // What is drawn is what the streamer shows; its budget is the draw budget.
    setBudget: () => undefined,
    render: (pose: ScanPose) => {
      if (
        pose.width !== size.width ||
        pose.height !== size.height ||
        pose.pixelRatio !== size.pixelRatio
      ) {
        size = { width: pose.width, height: pose.height, pixelRatio: pose.pixelRatio };
        renderer.setPixelRatio(pose.pixelRatio);
        renderer.setSize(pose.width, pose.height, false);
      }
      camera.position.set(...pose.eye);
      camera.up.set(...pose.up);
      camera.lookAt(target.copy(camera.position).add(direction.set(...pose.direction)));
      camera.fov = THREE.MathUtils.radToDeg(pose.fovy);
      camera.aspect = pose.width / Math.max(1, pose.height);
      camera.near = pose.near;
      camera.far = pose.far;
      camera.updateProjectionMatrix();
      renderer.render(scene, camera);
    },
    destroy: () => {
      renderer.setAnimationLoop(null);
      renderer.dispose();
    },
  };
  return Promise.resolve(backend);
}
