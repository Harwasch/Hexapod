import {
  CameraEventType,
  Transforms,
  Matrix4,
  Ray,
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
/** Zoom floor and near plane for whole sites versus hand-sized objects. */
const SITE_MIN_ZOOM_M = 0.6;
const SITE_NEAR_M = 1.0;
const OBJECT_MIN_ZOOM_M = 0.005;
const OBJECT_NEAR_M = 0.01;
/** Each wheel notch closes (or opens) this fraction of the distance to the point under it. */
const OBJECT_ZOOM_STEP = 0.82;
/** Radians of orbit per pixel of drag at object scale (about 0.35° per pixel). */
const OBJECT_ORBIT_RATE = 0.006;
/** Site-scale orbit: a drag across a 1440 px window turns about 260°; the same drag tilts about 90°. */
const ORBIT_HEADING_RATE = 0.0032;
const ORBIT_TILT_RATE = 0.0018;
/** The orbit never tips below the horizon or past straight down. */
const MIN_PITCH_DEG = -89.5;
const MAX_PITCH_DEG = -1;
/** Heights outside this band are terrain tiles still loading, not a place to pivot on. */
const PLAUSIBLE_HEIGHT_M: [number, number] = [-500, 9000];
const POSE_SETTLE_DELAYS_MS = [600, 2000];
/** After a gesture ends, the drawn surface under the camera is checked once, this much later. */
const FLOOR_CHECK_DELAY_MS = 250;
/** Largest correction the floor check applies; more than this is a mis-sample, not the ground. */
const MAX_FLOOR_LIFT_M = 40;
const IDLE_POSE_REFRESH_MS = 3000;
/** Below this bounding radius a fly-to may arrive closer than the site floor of 30 m. */
const OBJECT_ARRIVAL_RADIUS_M = 30;
/** Arrival tilt for fly-tos: mostly looking at the ground, still showing facades. */
const DEFAULT_ARRIVAL_PITCH = -45;
/** A flight that arcs at least this far above both of its ends looks straight down at the top. */
const LOOK_DOWN_CLIMB_M = 150;

const scratchPick = new Cartesian3();
const scratchDirection = new Cartesian3();
const scratchWindow = new Cartesian2();
const scratchRay = new Ray();
const scratchUp = new Cartesian3();
const scratchForward = new Cartesian3();
const scratchRight = new Cartesian3();

function preventDefault(event: Event): void {
  event.preventDefault();
}

/** Component of a vector perpendicular to `up`, normalised; null when it points straight up or down. */
function horizontal(vector: Cartesian3, up: Cartesian3, result: Cartesian3): Cartesian3 | null {
  const along = Cartesian3.dot(vector, up);
  Cartesian3.subtract(vector, Cartesian3.multiplyByScalar(up, along, result), result);
  if (Cartesian3.magnitude(result) < 1e-6) return null;
  return Cartesian3.normalize(result, result);
}
const scratchFrame = new Matrix4();
const scratchCenter = new Cartesian2();

/** Owns every camera movement so easing, limits and pose reporting live in one place. */
export class CameraController {
  private readonly scene: Scene;
  private readonly unsubscribe: (() => void)[] = [];
  private lastPose: CameraPose | null = null;
  private readonly reportPose = throttle(() => this.emitPose(), 100);
  private moving = false;
  private objectScale = false;
  private readonly settleTimers = new Set<ReturnType<typeof setTimeout>>();
  private orbitPivot: Cartesian3 | null = null;
  private orbitLast: { x: number; y: number } | null = null;
  private orbitRate = { heading: ORBIT_HEADING_RATE, tilt: ORBIT_TILT_RATE };
  private lastSurfaceHeight: number | undefined;

  constructor(
    private readonly viewer: Viewer,
    private readonly events: Emitter<SceneEvents>,
  ) {
    this.scene = viewer.scene;
    const camera = viewer.camera;
    camera.percentageChanged = 0.002;
    const controller = this.scene.screenSpaceCameraController;
    // Allow inspection at centimetre range; collision keeps us above terrain.
    // Close enough to read centimetre detail, far enough that a scroll does not pass through
    // a splat surface that has no collision geometry. Object-scale sites lower this.
    controller.minimumZoomDistance = SITE_MIN_ZOOM_M;
    controller.maximumZoomDistance = 40_000_000;
    controller.enableCollisionDetection = true;
    controller.inertiaSpin = 0.85;
    controller.inertiaTranslate = 0.85;
    controller.inertiaZoom = 0.8;
    controller.zoomFactor = 4;
    // Cesium tests terrain collision, and tilts around the terrain rather than the ellipsoid,
    // only while the camera is *below* minimumCollisionTerrainHeight (15 km by default).
    // Setting it to 0 would switch both off and make every tilt pivot on sea level.
    // Tilt and orbit are handled here instead (Google Maps style, around the view centre);
    // Cesium keeps pinch tilt for touch, wheel and pinch zoom, and left-drag pan.
    controller.tiltEventTypes = [CameraEventType.PINCH];
    controller.zoomEventTypes = [CameraEventType.WHEEL, CameraEventType.PINCH];
    const canvas = viewer.canvas;
    canvas.addEventListener("pointerdown", this.onOrbitStart);
    canvas.addEventListener("contextmenu", preventDefault);
    window.addEventListener("pointermove", this.onOrbitMove);
    window.addEventListener("pointerup", this.onOrbitEnd);
    this.unsubscribe.push(
      () => {
        canvas.removeEventListener("pointerdown", this.onOrbitStart);
        canvas.removeEventListener("contextmenu", preventDefault);
        window.removeEventListener("pointermove", this.onOrbitMove);
        window.removeEventListener("pointerup", this.onOrbitEnd);
      },
      // `camera.changed` fires only when position or orientation moved past
      // `percentageChanged`; `moveStart` also fires when the frustum changes, which a canvas
      // resize does. Starting motion from `changed` keeps a resolution switch from looking like
      // a camera move (that feedback loop made the UI pulse once a second).
      camera.changed.addEventListener(() => {
        if (!this.moving) {
          this.moving = true;
          this.events.emit("motion", true);
        }
        this.reportPose();
      }),
      camera.moveEnd.addEventListener(() => {
        if (this.moving) {
          this.moving = false;
          this.events.emit("motion", false);
        }
        this.emitPose();
        // Terrain and depth under the camera keep arriving after it stops; refresh the
        // altitude and scale readouts once they have.
        for (const delay of POSE_SETTLE_DELAYS_MS) {
          const timer = setTimeout(() => {
            this.settleTimers.delete(timer);
            if (!this.moving) this.emitPose();
          }, delay);
          this.settleTimers.add(timer);
        }
        const floorTimer = setTimeout(() => {
          this.settleTimers.delete(floorTimer);
          if (!this.moving) this.keepAboveDrawnSurface();
        }, FLOOR_CHECK_DELAY_MS);
        this.settleTimers.add(floorTimer);
      }),
    );
  }

  get isMoving(): boolean {
    return this.moving;
  }

  /**
   * Camera floor for meshes: Cesium's own collision against 3D Tiles ray-casts every loaded
   * tile's triangles on the CPU each frame (hundreds of milliseconds on a city mesh), so the
   * tilesets never enable it. Instead, once the camera has come to rest, the drawn surface
   * straight below it is read from the depth buffer (one pick pass), and a camera that ended
   * up under it or too close is eased back up. Wheel zoom already stops at the surface under
   * the cursor, so this only catches a pan or orbit that dipped below the ground.
   */
  private keepAboveDrawnSurface(): void {
    if (!this.scene.sampleHeightSupported) return;
    const camera = this.viewer.camera;
    if (!Matrix4.equals(camera.transform, Matrix4.IDENTITY)) return;
    const carto = Cartographic.clone(camera.positionCartographic, scratchCarto);
    let surface: number | undefined;
    try {
      surface = this.scene.sampleHeight(carto);
    } catch {
      return;
    }
    if (surface === undefined || !Number.isFinite(surface)) return;
    if (surface < PLAUSIBLE_HEIGHT_M[0] || surface > PLAUSIBLE_HEIGHT_M[1]) return;
    const clearance = this.scene.screenSpaceCameraController.minimumZoomDistance;
    const floor = surface + clearance;
    if (carto.height >= floor) return;
    const lift = floor - carto.height;
    // Only ever a small correction: a big difference means the sample hit something else
    // (a roof edge, a tree) rather than the ground the camera is over.
    if (lift > MAX_FLOOR_LIFT_M) return;
    camera.flyTo({
      destination: Cartesian3.fromRadians(carto.longitude, carto.latitude, floor),
      orientation: { heading: camera.heading, pitch: camera.pitch, roll: camera.roll },
      duration: 0.35,
      easingFunction: EasingFunction.QUADRATIC_OUT,
    });
  }

  /**
   * Object scale: for a model a few metres across the camera must get within millimetres of
   * its surface. The default near plane (1 m) and zoom floor (0.6 m) would clip and stop it,
   * so both drop while such a site is active and return to their site-scale values after.
   */
  setObjectScale(on: boolean): void {
    if (this.objectScale === on) return;
    this.objectScale = on;
    const controller = this.scene.screenSpaceCameraController;
    controller.minimumZoomDistance = on ? OBJECT_MIN_ZOOM_M : SITE_MIN_ZOOM_M;
    const frustum = this.viewer.camera.frustum;
    if ("near" in frustum) frustum.near = on ? OBJECT_NEAR_M : SITE_NEAR_M;
    // Cesium's wheel zoom is tuned for a planet: it never moves less than 20 m per notch and
    // refuses to zoom within 1 m of the floor. Objects get a proportional zoom-to-cursor.
    controller.enableZoom = !on;
    // Likewise its rotation rate is tied to height above the ellipsoid, so a drag beside a
    // rock barely turns. Objects get an orbit around the point that was clicked.
    controller.enableRotate = !on;
    const canvas = this.viewer.canvas;
    if (on) canvas.addEventListener("wheel", this.onObjectWheel, { passive: false });
    else canvas.removeEventListener("wheel", this.onObjectWheel);
    this.scene.requestRender();
  }

  /**
   * Google Maps mapping: Ctrl+drag, right-drag and middle-drag orbit the point in the middle
   * of the view (the thing you are looking at), left-drag pans. At object scale a plain
   * left-drag orbits the point that was clicked, because Cesium's rotation is tuned for a
   * planet and barely turns beside a rock.
   */
  private readonly onOrbitStart = (event: PointerEvent): void => {
    if (!this.scene.screenSpaceCameraController.enableInputs) return;
    const around =
      event.button === 1 || event.button === 2 || (event.button === 0 && event.ctrlKey);
    if (around) {
      this.orbitPivot = this.pivotAtCenter();
      this.orbitRate = this.objectScale
        ? { heading: OBJECT_ORBIT_RATE, tilt: OBJECT_ORBIT_RATE }
        : { heading: ORBIT_HEADING_RATE, tilt: ORBIT_TILT_RATE };
    } else if (event.button === 0 && this.objectScale) {
      scratchWindow.x = event.offsetX;
      scratchWindow.y = event.offsetY;
      this.orbitPivot = this.plausiblePick(scratchWindow) ?? this.pivotAtCenter();
      this.orbitRate = { heading: OBJECT_ORBIT_RATE, tilt: OBJECT_ORBIT_RATE };
    } else return;
    if (!this.orbitPivot) return;
    event.preventDefault();
    this.orbitLast = { x: event.clientX, y: event.clientY };
  };

  private readonly onOrbitMove = (event: PointerEvent): void => {
    if (!this.orbitPivot || !this.orbitLast) return;
    const dx = event.clientX - this.orbitLast.x;
    const dy = event.clientY - this.orbitLast.y;
    this.orbitLast = { x: event.clientX, y: event.clientY };
    if (dx === 0 && dy === 0) return;
    // Dragging up tips the view towards the horizon, dragging down towards straight down.
    this.orbit(this.orbitPivot, dx * this.orbitRate.heading, -dy * this.orbitRate.tilt);
  };

  private readonly onOrbitEnd = (): void => {
    this.orbitPivot = null;
    this.orbitLast = null;
  };

  /**
   * Turns the camera around a pivot, keeping its distance: `headingRad` moves the camera to
   * the left around it (the foreground follows a rightward drag, as in Cesium and Google
   * Maps), `pitchDeltaRad` is added to the camera pitch (positive tips towards the horizon,
   * negative towards straight down) and stops at both.
   */
  orbit(pivot: Cartesian3, headingRad: number, pitchDeltaRad: number): void {
    const camera = this.viewer.camera;
    const pitch = CesiumMath.toDegrees(camera.pitch);
    const nextPitch = CesiumMath.clamp(
      pitch + CesiumMath.toDegrees(pitchDeltaRad),
      MIN_PITCH_DEG,
      MAX_PITCH_DEG,
    );
    const allowedTilt = CesiumMath.toRadians(nextPitch - pitch);
    if (headingRad === 0 && allowedTilt === 0) return;
    // Orbit in the pivot's east-north-up frame, then drop the transform again so the rest of
    // the app keeps seeing a world-frame camera.
    camera.lookAtTransform(Transforms.eastNorthUpToFixedFrame(pivot, undefined, scratchFrame));
    camera.rotateLeft(headingRad);
    camera.rotateUp(allowedTilt);
    camera.lookAtTransform(Matrix4.IDENTITY);
    this.scene.requestRender();
  }

  /**
   * The point the view is centred on: whatever the depth buffer has there (a model, a
   * building, the ground), else the terrain, else a point along the view direction at the
   * current height. Gaussian splats write no depth, so under a splat this is its ground.
   */
  pivotAtCenter(): Cartesian3 | null {
    const canvas = this.viewer.canvas;
    scratchWindow.x = canvas.clientWidth / 2;
    scratchWindow.y = canvas.clientHeight / 2;
    const picked = this.plausiblePick(scratchWindow);
    if (picked) return picked;
    const camera = this.viewer.camera;
    const ray = camera.getPickRay(scratchWindow, scratchRay);
    if (!ray) return null;
    const pose = this.pose();
    const distance = pose.altitude / Math.max(Math.sin(CesiumMath.toRadians(-pose.pitch)), 0.05);
    return Ray.getPoint(ray, Math.max(distance, 1), new Cartesian3());
  }

  /** Depth pick, then terrain pick, both rejected when they land on a placeholder tile. */
  private plausiblePick(window: Cartesian2): Cartesian3 | null {
    const candidates: Cartesian3[] = [];
    if (this.scene.pickPositionSupported) {
      const depth = this.scene.pickPosition(window, new Cartesian3());
      if (depth) candidates.push(depth);
    }
    const ray = this.viewer.camera.getPickRay(window, scratchRay);
    const ground = ray ? this.scene.globe.pick(ray, this.scene, new Cartesian3()) : undefined;
    if (ground) candidates.push(ground);
    const cameraPosition = this.viewer.camera.positionWC;
    let best: Cartesian3 | null = null;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const candidate of candidates) {
      const height = Cartographic.fromCartesian(candidate, undefined, scratchCarto).height;
      if (height < PLAUSIBLE_HEIGHT_M[0] || height > PLAUSIBLE_HEIGHT_M[1]) continue;
      const distance = Cartesian3.distance(cameraPosition, candidate);
      if (distance < bestDistance) {
        best = candidate;
        bestDistance = distance;
      }
    }
    return best;
  }

  /** Slides the camera parallel to the ground by a screen-space amount (pixels at the view centre). */
  pan(dxPx: number, dyPx: number): void {
    const camera = this.viewer.camera;
    const pivot = this.pivotAtCenter();
    const distance = pivot ? Cartesian3.distance(camera.positionWC, pivot) : this.pose().altitude;
    const fovy =
      "fovy" in camera.frustum
        ? (camera.frustum as { fovy: number }).fovy
        : CesiumMath.PI_OVER_THREE;
    const metersPerPx = metersPerPixel(distance, fovy, this.viewer.canvas.clientHeight || 1);
    const up = Cartesian3.normalize(camera.positionWC, scratchUp);
    // Forward and right projected onto the local horizontal plane.
    const forward =
      horizontal(camera.directionWC, up, scratchForward) ??
      horizontal(camera.upWC, up, scratchForward);
    const right = horizontal(camera.rightWC, up, scratchRight);
    if (!forward || !right) return;
    camera.move(right, dxPx * metersPerPx);
    camera.move(forward, -dyPx * metersPerPx);
    this.scene.requestRender();
  }

  /** Moves toward (factor < 1) or away from (factor > 1) the view centre, keeping it centred. */
  zoomToward(pivot: Cartesian3, factor: number): void {
    const camera = this.viewer.camera;
    const toPivot = Cartesian3.subtract(pivot, camera.positionWC, scratchDirection);
    const distance = Cartesian3.magnitude(toPivot);
    if (!(distance > 0)) return;
    const floor = this.scene.screenSpaceCameraController.minimumZoomDistance;
    const next = Math.max(floor, distance * factor);
    const step = distance - next;
    if (Math.abs(step) < 1e-6) return;
    Cartesian3.normalize(toPivot, toPivot);
    camera.move(toPivot, step);
    this.scene.requestRender();
  }

  /** Zoom toward the point under the cursor by a fixed fraction of the remaining distance. */
  private readonly onObjectWheel = (event: WheelEvent): void => {
    event.preventDefault();
    const camera = this.viewer.camera;
    scratchWindow.x = event.offsetX;
    scratchWindow.y = event.offsetY;
    let target = this.scene.pickPositionSupported
      ? this.scene.pickPosition(scratchWindow, scratchPick)
      : undefined;
    if (!target) {
      const ray = camera.getPickRay(scratchWindow, scratchRay);
      target = ray ? this.scene.globe.pick(ray, this.scene, scratchPick) : undefined;
    }
    if (!target) return;
    const toTarget = Cartesian3.subtract(target, camera.positionWC, scratchDirection);
    const distance = Cartesian3.magnitude(toTarget);
    if (!(distance > 0)) return;
    const factor = event.deltaY < 0 ? OBJECT_ZOOM_STEP : 1 / OBJECT_ZOOM_STEP;
    const next = Math.max(OBJECT_MIN_ZOOM_M, distance * factor);
    const step = distance - next;
    if (Math.abs(step) < 1e-5) return;
    Cartesian3.normalize(toTarget, toTarget);
    camera.move(toTarget, step);
    this.scene.requestRender();
  };

  get isObjectScale(): boolean {
    return this.objectScale;
  }

  /** Current pose, computed on demand. */
  pose(): CameraPose {
    const camera = this.viewer.camera;
    const carto = camera.positionCartographic;
    const longitude = CesiumMath.toDegrees(carto.longitude);
    const latitude = CesiumMath.toDegrees(carto.latitude);
    // Terrain tiles still loading report placeholder heights kilometres below sea level;
    // treat those as unknown rather than turning a 700 m view into a "7 km" readout.
    const sampled = this.scene.globe.getHeight(carto);
    if (sampled !== undefined && sampled > -500 && sampled < 9000) this.lastSurfaceHeight = sampled;
    // Very close to the ground the exact terrain tile may not be rendered yet; the last
    // plausible height nearby is a far better guess than sea level.
    const surface = this.lastSurfaceHeight ?? 0;
    const altitude = Math.max(0, carto.height - surface);
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
    // At object scale the thing under the crosshair is the model, not the globe: read the
    // depth buffer so the mm/px readout describes the object the user is looking at.
    if (this.objectScale && this.scene.pickPositionSupported) {
      const picked = this.scene.pickPosition(scratchCenter, scratchPick);
      if (picked) return Cartesian3.distance(this.viewer.camera.positionWC, picked);
    }
    const ray = this.viewer.camera.getPickRay(scratchCenter);
    if (!ray) return null;
    const hit = this.scene.globe.pick(ray, this.scene);
    if (!hit) return null;
    // A ray can land on a terrain tile that is still a placeholder kilometres below sea
    // level; that distance would turn a 40 m view into a "97 m/px" readout.
    const height = Cartographic.fromCartesian(hit).height;
    if (height < -500 || height > 9000) return null;
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

  /** Terrain under a resting camera keeps refining; keep the readouts honest while idle. */
  private readonly idleRefresh = setInterval(() => {
    if (!this.moving && this.lastPose) this.emitPose();
  }, IDLE_POSE_REFRESH_MS);

  /** Re-reads the pose without a camera move, e.g. once a model under the camera has loaded. */
  refreshPose(): void {
    if (!this.moving) this.emitPose();
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
    const pitch = options.pitch ?? DEFAULT_ARRIVAL_PITCH;
    const destination = Cartesian3.fromDegrees(longitude, latitude, height);
    this.viewer.camera.flyTo({
      destination,
      orientation: {
        heading: CesiumMath.toRadians(heading),
        pitch: CesiumMath.toRadians(pitch),
        roll: 0,
      },
      duration: options.durationS ?? this.durationFor(destination),
      easingFunction: EasingFunction.QUADRATIC_IN_OUT,
      pitchAdjustHeight: this.pitchAdjustHeight(height),
      complete: options.onComplete,
    });
  }

  /**
   * Any flight that arcs well above both of its ends points the camera straight down at the
   * top of the arc and eases back to the arrival pitch on the way down (Cesium's
   * pitchAdjustHeight). Without it the pitch is interpolated linearly, so a long hop spends
   * its high part looking at the horizon. Measured against the higher end so a short hop or
   * a plain descent keeps a steady tilt instead of nodding.
   */
  private pitchAdjustHeight(arrivalHeight: number): number {
    const higherEnd = Math.max(arrivalHeight, this.viewer.camera.positionCartographic.height);
    return Math.max(higherEnd * 1.5, higherEnd + LOOK_DOWN_CLIMB_M);
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
      pitchAdjustHeight: this.pitchAdjustHeight(this.pose().altitude),
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

  /** Steep approach to a sphere, looking down at it: the standard "arrive at a site" move. */
  flyToBoundingSphere(
    sphere: BoundingSphere,
    options: FlyOptions & { rangeMultiplier?: number } = {},
  ): void {
    // Whole sites never arrive closer than 30 m; a hand-sized object arrives at a few
    // times its own radius so it fills the view.
    const floor = sphere.radius < OBJECT_ARRIVAL_RADIUS_M ? 0.3 : 30;
    const range = Math.max(sphere.radius * (options.rangeMultiplier ?? 3.2), floor);
    const pitch = CesiumMath.toRadians(options.pitch ?? DEFAULT_ARRIVAL_PITCH);
    const arrivalHeight =
      Cartographic.fromCartesian(sphere.center).height + range * Math.sin(-pitch);
    this.viewer.camera.flyToBoundingSphere(sphere, {
      offset: new HeadingPitchRange(CesiumMath.toRadians(options.heading ?? 100), pitch, range),
      duration: options.durationS ?? this.durationFor(sphere.center),
      easingFunction: EasingFunction.QUADRATIC_IN_OUT,
      pitchAdjustHeight: this.pitchAdjustHeight(arrivalHeight),
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
    for (const timer of this.settleTimers) clearTimeout(timer);
    clearInterval(this.idleRefresh);
    this.setObjectScale(false);
    for (const off of this.unsubscribe) off();
  }
}
