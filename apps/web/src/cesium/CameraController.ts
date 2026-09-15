import {
  type BoundingSphere,
  Cartesian2,
  Cartesian3,
  Cartographic,
  EasingFunction,
  HeadingPitchRange,
  Math as CesiumMath,
  Rectangle,
  type Scene,
  type Viewer,
} from "cesium";

import type { CameraBookmark } from "@twin/contracts";
import { metersPerPixel, scaleBandForAltitude } from "@twin/geo";

import type { CameraPose } from "@/state/viewer";

import type { Emitter } from "@/lib/emitter";
import { throttle } from "@/lib/throttle";

import type { SceneEvents } from "./types";

export interface FlyOptions {
  durationS?: number;
  /** Degrees. */
  heading?: number;
  /** Degrees (negative looks down). */
  pitch?: number;
  onComplete?: () => void;
}

const scratchCarto = new Cartographic();

function normalizeDegrees(value: number): number {
  const wrapped = ((value % 360) + 360) % 360;
  return wrapped > 180 ? wrapped - 360 : wrapped;
}
const scratchCenter = new Cartesian2();

/** Owns every camera movement so easing, limits and pose reporting live in one place. */
export class CameraController {
  private readonly scene: Scene;
  private readonly unsubscribe: (() => void)[] = [];
  private lastPose: CameraPose | null = null;
  private readonly reportPose = throttle(() => this.emitPose(), 100);
  private moving = false;

  constructor(
    private readonly viewer: Viewer,
    private readonly events: Emitter<SceneEvents>,
  ) {
    this.scene = viewer.scene;
    const camera = viewer.camera;
    camera.percentageChanged = 0.002;
    const controller = this.scene.screenSpaceCameraController;
    // Allow inspection at centimetre range; collision keeps us above terrain.
    controller.minimumZoomDistance = 0.05;
    controller.maximumZoomDistance = 40_000_000;
    controller.enableCollisionDetection = true;
    controller.inertiaSpin = 0.85;
    controller.inertiaTranslate = 0.85;
    controller.inertiaZoom = 0.8;
    controller.zoomFactor = 4;
    controller.minimumCollisionTerrainHeight = 0;
    this.unsubscribe.push(
      camera.changed.addEventListener(() => this.reportPose()),
      camera.moveStart.addEventListener(() => {
        this.moving = true;
        this.reportPose();
      }),
      camera.moveEnd.addEventListener(() => {
        this.moving = false;
        this.emitPose();
      }),
    );
  }

  get isMoving(): boolean {
    return this.moving;
  }

  /** Current pose, computed on demand. */
  pose(): CameraPose {
    const camera = this.viewer.camera;
    const carto = camera.positionCartographic;
    const longitude = CesiumMath.toDegrees(carto.longitude);
    const latitude = CesiumMath.toDegrees(carto.latitude);
    const surface = this.scene.globe.getHeight(carto);
    const altitude = Math.max(0, carto.height - (surface ?? 0));
    const distance = this.distanceToSurfaceAtCenter() ?? altitude;
    const fovy =
      "fovy" in camera.frustum
        ? (camera.frustum as { fovy: number }).fovy
        : CesiumMath.PI_OVER_THREE;
    return {
      longitude,
      latitude,
      height: carto.height,
      heading: ((CesiumMath.toDegrees(camera.heading) % 360) + 360) % 360,
      pitch: CesiumMath.toDegrees(camera.pitch),
      roll: normalizeDegrees(CesiumMath.toDegrees(camera.roll)),
      altitude,
      scaleBand: scaleBandForAltitude(altitude),
      metersPerPixel: metersPerPixel(distance, fovy, this.viewer.canvas.clientHeight || 1),
    };
  }

  private distanceToSurfaceAtCenter(): number | null {
    const canvas = this.viewer.canvas;
    scratchCenter.x = canvas.clientWidth / 2;
    scratchCenter.y = canvas.clientHeight / 2;
    const ray = this.viewer.camera.getPickRay(scratchCenter);
    if (!ray) return null;
    const hit = this.scene.globe.pick(ray, this.scene);
    if (!hit) return null;
    return Cartesian3.distance(this.viewer.camera.positionWC, hit);
  }

  private emitPose(): void {
    const pose = this.pose();
    this.lastPose = pose;
    this.events.emit("camera", pose);
  }

  get lastKnownPose(): CameraPose | null {
    return this.lastPose;
  }

  /** Instant view without animation (used for the initial Earth view). */
  setView(longitude: number, latitude: number, height: number, heading = 0, pitch = -90): void {
    this.viewer.camera.setView({
      destination: Cartesian3.fromDegrees(longitude, latitude, height),
      orientation: {
        heading: CesiumMath.toRadians(heading),
        pitch: CesiumMath.toRadians(pitch),
        roll: 0,
      },
    });
    this.emitPose();
  }

  flyTo(longitude: number, latitude: number, height: number, options: FlyOptions = {}): void {
    const pose = this.pose();
    const heading = options.heading ?? pose.heading;
    const pitch = options.pitch ?? -45;
    this.viewer.camera.flyTo({
      destination: Cartesian3.fromDegrees(longitude, latitude, height),
      orientation: {
        heading: CesiumMath.toRadians(heading),
        pitch: CesiumMath.toRadians(pitch),
        roll: 0,
      },
      duration:
        options.durationS ?? this.durationFor(Cartesian3.fromDegrees(longitude, latitude, height)),
      easingFunction: EasingFunction.QUADRATIC_IN_OUT,
      complete: options.onComplete,
    });
  }

  flyToRectangle(
    west: number,
    south: number,
    east: number,
    north: number,
    options: FlyOptions = {},
  ): void {
    this.viewer.camera.flyTo({
      destination: Rectangle.fromDegrees(west, south, east, north),
      duration: options.durationS ?? 2.2,
      easingFunction: EasingFunction.QUADRATIC_IN_OUT,
      complete: options.onComplete,
    });
  }

  flyToBookmark(bookmark: CameraBookmark, options: FlyOptions = {}): void {
    this.flyTo(bookmark.longitude, bookmark.latitude, bookmark.height, {
      heading: bookmark.heading,
      pitch: bookmark.pitch,
      ...options,
    });
  }

  /** Oblique approach to a sphere: the standard "arrive at a site" move. */
  flyToBoundingSphere(
    sphere: BoundingSphere,
    options: FlyOptions & { rangeMultiplier?: number } = {},
  ): void {
    const range = Math.max(sphere.radius * (options.rangeMultiplier ?? 3.2), 30);
    this.viewer.camera.flyToBoundingSphere(sphere, {
      offset: new HeadingPitchRange(
        CesiumMath.toRadians(options.heading ?? 100),
        CesiumMath.toRadians(options.pitch ?? -25),
        range,
      ),
      duration: options.durationS ?? this.durationFor(sphere.center),
      easingFunction: EasingFunction.QUADRATIC_IN_OUT,
      complete: options.onComplete,
    });
  }

  /** Rotates the view so north is up while keeping the position and tilt. */
  resetNorth(): void {
    const camera = this.viewer.camera;
    camera.flyTo({
      destination: camera.positionWC.clone(),
      orientation: { heading: 0, pitch: camera.pitch, roll: 0 },
      duration: 0.8,
      easingFunction: EasingFunction.QUADRATIC_IN_OUT,
    });
  }

  /** Looks straight down from the current position (feels like a 2D map). */
  topDown(): void {
    const camera = this.viewer.camera;
    const target = this.distanceToSurfaceAtCenter();
    const carto = Cartographic.clone(camera.positionCartographic, scratchCarto);
    // Move above the current look-at point so the same area stays in view.
    const centerRay = camera.getPickRay(
      new Cartesian2(this.viewer.canvas.clientWidth / 2, this.viewer.canvas.clientHeight / 2),
    );
    const hit = centerRay ? this.scene.globe.pick(centerRay, this.scene) : undefined;
    const destinationCarto = hit ? Cartographic.fromCartesian(hit) : carto;
    const height = Math.max(
      carto.height,
      (target ?? carto.height) + (destinationCarto.height ?? 0),
    );
    camera.flyTo({
      destination: Cartesian3.fromRadians(
        destinationCarto.longitude,
        destinationCarto.latitude,
        height,
      ),
      orientation: { heading: 0, pitch: -CesiumMath.PI_OVER_TWO, roll: 0 },
      duration: 1,
      easingFunction: EasingFunction.QUADRATIC_IN_OUT,
    });
  }

  zoomBy(factor: number): void {
    const distance = this.distanceToSurfaceAtCenter() ?? this.pose().altitude;
    const amount = distance * factor;
    if (amount > 0) this.viewer.camera.zoomIn(amount);
    else this.viewer.camera.zoomOut(-amount);
  }

  /** Home: the whole planet, oriented towards the given longitude. */
  flyHome(longitude = -110, latitude = 30): void {
    this.viewer.camera.flyTo({
      destination: Cartesian3.fromDegrees(longitude, latitude, 18_000_000),
      orientation: { heading: 0, pitch: -CesiumMath.PI_OVER_TWO, roll: 0 },
      duration: 2.5,
      easingFunction: EasingFunction.QUADRATIC_IN_OUT,
    });
  }

  cancelFlight(): void {
    this.viewer.camera.cancelFlight();
  }

  private durationFor(destination: Cartesian3): number {
    const distance = Cartesian3.distance(this.viewer.camera.positionWC, destination);
    // ~1.5 s for a local hop, ~5 s from orbit; never sluggish.
    return CesiumMath.clamp(1.2 + Math.log10(Math.max(distance, 10)) * 0.55, 1.2, 5.5);
  }

  destroy(): void {
    this.reportPose.cancel();
    for (const off of this.unsubscribe) off();
  }
}
