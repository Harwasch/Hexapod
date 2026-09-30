/**
 * The Spark back-end of the scan renderer (ScanRendererHost): three.js with Spark's splat
 * renderer, as the scan viewer page draws a scan. Spark keeps a level-of-detail tree per tile
 * and draws at most `lodSplatCount` splats a frame across all of them, so it is streamed more
 * than it draws (`loadFactor`) and chooses.
 */

import { SparkRenderer, type SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";

import { loadSplatTile } from "@/view/sparkStream";
import { LOAD_FACTOR } from "@/view/tiles";

import type { ScanBackend, ScanPose } from "./types";

export function createBackend(
  canvas: HTMLCanvasElement,
  budget: number,
): Promise<ScanBackend<SplatMesh>> {
  const renderer = new THREE.WebGLRenderer({
    canvas,
    alpha: true,
    antialias: false,
    premultipliedAlpha: true,
    powerPreference: "high-performance",
  });
  renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  const spark = new SparkRenderer({ renderer, lodSplatCount: budget });
  scene.add(spark);
  const camera = new THREE.PerspectiveCamera();
  const target = new THREE.Vector3();
  let size = { width: 0, height: 0, pixelRatio: 0 };

  const backend: ScanBackend<SplatMesh> = {
    name: "spark",
    loadFactor: LOAD_FACTOR,
    load: async (tilesetUrl, tile) => (await loadSplatTile(tilesetUrl, tile)).mesh,
    add: (mesh) => scene.add(mesh),
    remove: (mesh) => scene.remove(mesh),
    dispose: (mesh) => mesh.dispose(),
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
      camera.lookAt(target.set(...pose.eye).add(new THREE.Vector3(...pose.direction)));
      camera.fov = THREE.MathUtils.radToDeg(pose.fovy);
      camera.aspect = pose.width / Math.max(1, pose.height);
      camera.near = pose.near;
      camera.far = pose.far;
      camera.updateProjectionMatrix();
      renderer.render(scene, camera);
    },
    // Spark shows a new mesh once its level-of-detail tree is built and a sort has run: until
    // then the displayed mapping holds none of its splats.
    isDrawn: (mesh) =>
      spark.display.mapping.some(
        (entry) => (entry.node === mesh || entry.node.parent === mesh) && entry.count > 0,
      ),
    setBudget: (drawn) => {
      spark.lodSplatCount = drawn;
    },
    destroy: () => {
      renderer.setAnimationLoop(null);
      renderer.dispose();
    },
  };
  return Promise.resolve(backend);
}
