/**
 * A bare CesiumJS viewer with the console's camera stack -- CameraController, the splat
 * motion gate and the splat collider -- over one splat tileset: the driver for the
 * navigation end-to-end check (e2e/splatNavigation.spec.ts).
 *
 * What it answers, in a real engine: does the camera stop at a splat surface it is driven
 * into, slide rather than stick, pass through with Space held, and does a wheel over a splat
 * zoom to it and stop short?
 *
 * Loaded dynamically by the spec; nothing imports it, so it never reaches the production bundle.
 */

import {
  BoundingSphere,
  Cartesian2,
  Cartesian3,
  CesiumWidget,
  Color,
  HeadingPitchRange,
  Math as CesiumMath,
  Matrix4,
  Ray,
  SceneTransforms,
} from "cesium";
import type { SiteAsset } from "@twin/contracts";

import { Emitter } from "@/lib/emitter";

import { CameraController } from "@/cesium/CameraController";
import { ExploreController } from "@/cesium/ExploreController";
import { SplatCollider } from "@/cesium/SplatCollider";
import { createSiteTileset } from "@/cesium/providers/tiles";
import { SplatMotionGate } from "@/cesium/splatMotionGate";
import { installSplatDecoder } from "@/cesium/splatDecoder";
import { installSplatSorter } from "@/cesium/splatSorter";
import type { SceneEvents } from "@/cesium/types";

export interface CollisionHarness {
  /** Resolves once the collider has the tileset's splats. */
  ready(): Promise<void>;
  /** Puts the camera `range` metres from the centre, level, looking at it. */
  place(range: number, headingDeg?: number): void;
  /** Distance from the camera to the tileset's centre. */
  distanceToCentre(): number;
  /** The first splat hit straight ahead, in metres, or null. */
  hitAhead(): number | null;
  /**
   * Moves the camera `stepM` forward (and `sideways` right) a frame for `frames` frames;
   * resolves to the closest it came to a splat surface, as a share of its clearance there
   * (1 or more: never inside; null: nothing near).
   */
  drive(stepM: number, frames: number, sideways?: number): Promise<number | null>;
  /** What the collider holds, and what the splat primitive draws. */
  debug(): unknown;
  /** How close the camera is to the nearest splat surface, as a share of its clearance
   *  there (null: nothing within twice the clearance). */
  clearanceRatio(): number | null;
  /** The camera's world position. */
  eye(): [number, number, number];
  /** Puts the camera `range` metres from the centre, `pitchDeg` below level (map view). */
  lookDown(range: number, pitchDeg: number, headingDeg?: number): void;
  /** The splat surface under a window position (CSS pixels), or null. */
  surfaceAt(x: number, y: number): [number, number, number] | null;
  /** Where a world point is drawn, in window CSS pixels, or null. */
  windowOf(point: [number, number, number]): [number, number] | null;
  /** The map camera's heading, radians. */
  heading(): number;
  /** First-person exploring, as the console's G key starts it. */
  explore: {
    enter(): void;
    exit(): void;
    mode(): string;
    heading(): number;
    height(): number;
  };
  /** The toasts raised so far (titles). */
  toasts: string[];
}

export async function startCollisionHarness(
  container: HTMLElement,
  tilesetUrl: string,
  options: { hidden?: boolean } = {},
): Promise<CollisionHarness> {
  // The same widget the app runs on (CesiumSceneManager), so the managers see what they get there.
  const viewer = new CesiumWidget(container, {
    baseLayer: false,
    requestRenderMode: false,
    msaaSamples: 1,
  });
  const { scene } = viewer;
  scene.globe.show = false;
  if (scene.skyBox) scene.skyBox.show = false;
  if (scene.skyAtmosphere) scene.skyAtmosphere.show = false;
  scene.backgroundColor = Color.fromCssColorString("#10141a");
  const events = new Emitter<SceneEvents>();
  const toasts: string[] = [];
  events.on("toast", (toast) => toasts.push(toast.title));
  const camera = new CameraController(viewer, events);
  const gate = new SplatMotionGate(scene, events);
  installSplatSorter();
  installSplatDecoder();
  const collider = new SplatCollider(scene, () => gate.holding);
  camera.setCollider(collider);
  const explore = new ExploreController(viewer, events);
  explore.setCollider(collider);
  explore.setCameraController(camera);
  const asset = {
    representation: "gaussian-splat",
    source: { type: "3d-tiles-url", url: tilesetUrl },
    renderConfig: {},
  } as unknown as SiteAsset;
  const tileset = await createSiteTileset(asset, { maximumScreenSpaceError: 1 });
  // `hidden`: drawn by another splat renderer (scanView) from the start, so CesiumJS never
  // loads a tile of it -- and its solids must count all the same.
  tileset.show = !options.hidden;
  tileset.preloadWhenHidden = false;
  if (options.hidden) collider.setSolidWhileHidden(tileset);
  scene.primitives.add(tileset);
  const sphere = BoundingSphere.clone(tileset.boundingSphere);

  const frame = (): Promise<void> =>
    new Promise((done) => {
      const off = scene.postRender.addEventListener(() => {
        off();
        done();
      });
    });

  const place = (range: number, headingDeg = 0): void => {
    viewer.camera.lookAt(
      sphere.center,
      new HeadingPitchRange(CesiumMath.toRadians(headingDeg), 0, range),
    );
    viewer.camera.lookAtTransform(Matrix4.IDENTITY);
    // A placement is a jump, not a motion to check against the surfaces between.
    camera.setCollider(collider);
  };

  return {
    toasts,
    async ready() {
      place(sphere.radius * 3);
      for (let i = 0; i < 600 && !collider.active; i++) await frame();
      // Let the finest tiles in and the grid catch up.
      for (let i = 0; i < 120; i++) await frame();
    },
    place,
    debug: () => ({
      collider: collider.describe(),
      selected: (tileset as unknown as { _selectedTiles: unknown[] })._selectedTiles.length,
      holding: gate.holding,
    }),
    explore: {
      enter: () => explore.enter(),
      exit: () => explore.exit(),
      mode: () => explore.status.mode,
      heading: () => viewer.camera.heading,
      height: () => viewer.camera.positionCartographic.height,
    },
    clearanceRatio: () => {
      const position = viewer.camera.positionWC;
      const clearance = collider.clearance(position);
      if (clearance <= 0) return null;
      const d = collider.distanceToSurface(position, clearance * 2);
      return d === null ? null : d / clearance;
    },
    eye: () => {
      const p = viewer.camera.positionWC;
      return [p.x, p.y, p.z];
    },
    lookDown: (range, pitchDeg, headingDeg = 0) => {
      viewer.camera.lookAt(
        sphere.center,
        new HeadingPitchRange(
          CesiumMath.toRadians(headingDeg),
          CesiumMath.toRadians(-pitchDeg),
          range,
        ),
      );
      viewer.camera.lookAtTransform(Matrix4.IDENTITY);
      camera.setCollider(collider);
    },
    surfaceAt: (x, y) => {
      const ray = viewer.camera.getPickRay(new Cartesian2(x, y));
      const hit = ray ? collider.raycast(ray) : undefined;
      return hit ? [hit.point.x, hit.point.y, hit.point.z] : null;
    },
    windowOf: (point) => {
      const at = SceneTransforms.worldToWindowCoordinates(scene, new Cartesian3(...point));
      return at ? [at.x, at.y] : null;
    },
    heading: () => viewer.camera.heading,
    distanceToCentre: () => Cartesian3.distance(viewer.camera.positionWC, sphere.center),
    hitAhead: () =>
      collider.raycast(new Ray(viewer.camera.positionWC, viewer.camera.directionWC))?.distance ??
      null,
    async drive(stepM, frames, sideways = 0) {
      let closest: number | null = null;
      for (let i = 0; i < frames; i++) {
        viewer.camera.moveForward(stepM);
        if (sideways) viewer.camera.moveRight(sideways);
        await frame();
        const position = viewer.camera.positionWC;
        const clearance = collider.clearance(position);
        if (clearance <= 0) continue;
        const d = collider.distanceToSurface(position, clearance * 2);
        if (d === null) continue;
        closest = Math.min(closest ?? Number.POSITIVE_INFINITY, d / clearance);
      }
      return closest;
    },
  };
}
