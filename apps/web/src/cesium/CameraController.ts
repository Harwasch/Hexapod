import {
  type Camera as CesiumCamera,
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
  sampleTerrainMostDetailed,
  type Scene,
  type CesiumWidget,
} from "cesium";

import type { CameraBookmark } from "@twin/contracts";
import { metersPerPixel, scaleBandForAltitude } from "@twin/geo";

import type { CameraPose } from "@/state/viewer";

import type { Emitter } from "@/lib/emitter";
import { isTyping } from "@/lib/hotkeys";
import { throttle } from "@/lib/throttle";

import { CameraOwnership } from "./cameraOwnership";
import type { ArrivalPose } from "./flightRetarget";
import { Glide, type GlideCarry, type GlidePose } from "./glide";
import { plausibleGround } from "./placement";
import type { ScanDestination } from "./scanView/ScanRendererHost";
import type { SplatCollider } from "./SplatCollider";
import type { SceneEvents } from "./types";

export interface FlyOptions {
  durationS?: number;
  /** Degrees. */
  heading?: number;
  /** Degrees (negative looks down). */
  pitch?: number;
  onComplete?: () => void;
  /** Called instead when the flight is replaced or cancelled before it arrives. */
  onCancel?: () => void;
}

/** What `glide` reports back: arrival, or the flight given up for another, or for a hand. */
export interface GlideOptions {
  /** Seconds; by default the path's own length decides (glide.ts). */
  durationS?: number;
  onComplete?: () => void;
  /** Called instead when another flight, a gesture or a key takes the camera first. */
  onCancel?: () => void;
}

/** A flight `glide` started: pointed at a better pose on the way, or stopped. */
export interface GlideHandle {
  /** Moves where the flight is going, smoothly, however far along it is. */
  retarget(pose: GlidePose): void;
  /** Stops it where it is (its `onCancel` is called). */
  cancel(): void;
  /** Whether it is still flying. */
  readonly active: boolean;
}

/** The glide under way: its flight, its callbacks, and what stands in for it on the camera. */
interface ActiveGlide {
  glide: Glide;
  options: GlideOptions;
  /** The last two poses it gave the camera, Earth-fixed, with their times (`motion`). */
  trail: { at: number; position: Cartesian3; heading: number; pitch: number }[];
  /** Stands in for a Cesium flight on the camera (`_currentFlight`), so a `camera.flyTo` from
   *  anywhere, or `cancelFlight`, ends this glide first, as it would a flight of its own. */
  sentinel: { cancelTween: () => void };
  handle: GlideHandle & { active: boolean };
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
/** Never sample the floor more often than this; a pause inside a drag is not a rest. */
const FLOOR_CHECK_MIN_INTERVAL_MS = 1500;
/** Roll below this (about 0.03°) is float noise; above it the horizon is visibly tilted. */
const ROLL_TOLERANCE_RAD = 0.0005;
/** Largest correction the floor check applies; more than this is a mis-sample, not the ground. */
const MAX_FLOOR_LIFT_M = 40;
/** How long the floor check's lift takes (s). */
const FLOOR_LIFT_S = 0.6;
/**
 * A camera this far under the terrain at full detail (m) once a flight has landed is
 * underground: black, or the inside of the earth. Less is a scan's own ground or the
 * photorealistic world's, a few metres off the terrain model, which the floor check handles.
 */
const UNDERGROUND_M = 25;
/** A camera found underground is brought up to this far over the terrain (m), in this long (s). */
const SURFACED_M = 30;
const SURFACE_S = 1.2;

/**
 * Where a camera that landed at `height` over terrain sampled at `terrain` is brought up to,
 * or null when it is not underground (or the terrain is no answer worth believing).
 */
export function surfacedHeight(height: number, terrain: number | undefined): number | null {
  const ground = plausibleGround(terrain);
  if (ground === undefined || height >= ground - UNDERGROUND_M) return null;
  return ground + SURFACED_M;
}
/** Wheel events closer than this belong to one gesture. */
const WHEEL_GESTURE_MS = 250;
/** A cursor that moved less than this (Manhattan pixels) is still over the same point. */
const WHEEL_SAME_POINT_PX = 4;
const IDLE_POSE_REFRESH_MS = 3000;
/** Below this bounding radius a fly-to may arrive closer than the site floor of 30 m. */
const OBJECT_ARRIVAL_RADIUS_M = 30;
/** Each wheel notch (100 px of delta) toward a splat under the cursor closes this share of
 *  the distance that is left; farther than the clearance, never closer. */
const SURFACE_ZOOM_STEP = 0.75;
/** Arrival tilt for fly-tos: mostly looking at the ground, still showing facades. */
const DEFAULT_ARRIVAL_PITCH = -45;
/** Arrival tilt for an object of a scan: lower, so its sides show as well as its top. */
const OBJECT_ARRIVAL_PITCH = -35;
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
const scratchBefore = new Cartesian3();
const scratchBeforeDirection = new Cartesian3();
const scratchCenter = new Cartesian2();
const scratchGrabRay = new Ray();
const scratchGrabHit = new Cartesian3();
const scratchGrabStep = new Cartesian3();

/** Below this camera height a left-drag grabs the point under the cursor and keeps it there
 *  (Google Maps); above it Cesium's globe spin, which is what a planet-scale drag wants. */
const GRAB_PAN_MAX_ALTITUDE_M = 30_000;
/** A drag ray meeting the grabbed plane further than this many times the grab distance (near
 *  the horizon) is not followed: the ground there moves kilometres per pixel. */
const GRAB_PAN_MAX_REACH = 25;
/** Pan inertia after release: the drag's velocity over this window... */
const PAN_VELOCITY_WINDOW_MS = 80;
/** ...decays with this time constant (ms), and stops below this speed (fraction of the grab
 *  distance per second). */
const PAN_INERTIA_TAU_MS = 220;
const PAN_INERTIA_MIN_SPEED = 0.05;

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
  /** A left-drag pan in progress: the grabbed point and the plane it slides on. */
  private grab: { point: Cartesian3; normal: Cartesian3; reach: number } | null = null;
  private panSamples: { at: number; step: Cartesian3 }[] = [];
  private panInertia = 0;
  private pointerHeld = false;
  private lastFloorCheckAt = 0;
  private lastSurfaceHeight: number | undefined;
  private collider: SplatCollider | null = null;
  /** Space held: the camera passes through splat surfaces instead of stopping at them. */
  private passThrough = false;
  private passKeyEnabled = true;
  /** Where the camera last was that the collision check accepted. */
  private lastGood: Cartesian3 | null = null;
  private wheelOcclusion: { x: number; y: number; at: number; occluded: boolean } | null = null;
  private hinted = false;
  /** Every flight this controller has started (`flights`). */
  private flightCount = 0;
  /**
   * Fetches ahead what a dedicated splat renderer will show where a flight ends, returning its
   * cancel (`prefetchScanDestination`, set by the scene manager; SiteManager's fly-to does the
   * same for its own glides).
   */
  private prefetch: ((destination: ScanDestination) => () => void) | null = null;
  /** The glide under way, if any (`glide`). */
  private current: ActiveGlide | null = null;
  /** The clock glides run on (ms; `setGlideClock`). */
  private glideClock: () => number = () => performance.now();
  /** Who may move the camera: the person once they have touched it (cameraOwnership.ts). */
  private readonly ownership = new CameraOwnership();

  constructor(
    private readonly viewer: CesiumWidget,
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
    // Shift+drag orbits here (Google Maps), so Cesium's free look on Shift+drag goes.
    controller.lookEventTypes = [];
    controller.zoomEventTypes = [CameraEventType.WHEEL, CameraEventType.PINCH];
    const canvas = viewer.canvas;
    // The person's hand on the map, first of all: it takes the camera (cameraOwnership.ts).
    // In the capture phase on the canvas's container, so it sees every wheel notch, including
    // the ones the splat zoom below takes for itself.
    const container = viewer.container as HTMLElement;
    container.addEventListener("wheel", this.onUserWheel, { capture: true, passive: true });
    canvas.addEventListener("pointerdown", this.onPointerHeld);
    window.addEventListener("pointerup", this.onPointerReleased);
    window.addEventListener("pointercancel", this.onPointerReleased);
    canvas.addEventListener("pointerdown", this.onOrbitStart);
    canvas.addEventListener("pointerdown", this.onPanStart);
    canvas.addEventListener("contextmenu", preventDefault);
    window.addEventListener("pointermove", this.onOrbitMove);
    window.addEventListener("pointerup", this.onOrbitEnd);
    window.addEventListener("pointermove", this.onPanMove);
    window.addEventListener("pointerup", this.onPanEnd);
    window.addEventListener("pointercancel", this.onPanEnd);
    canvas.addEventListener("wheel", this.stopPanInertia, { passive: true });
    window.addEventListener("pointermove", this.onUserDrag);
    // Capture, on the canvas's container: a wheel over a splat is handled here before
    // Cesium's own zoom (which listens on the canvas) ever sees it.
    container.addEventListener("wheel", this.onSurfaceWheel, { capture: true, passive: false });
    // CesiumJS's terrain collision acts only on the person's own motion once they have the
    // camera, never on it at rest (`collide`).
    // `update` is CesiumJS's own (private in its typings): called once a frame by the scene.
    const internal = controller as unknown as { update: () => void };
    const update = internal.update.bind(controller);
    internal.update = () => this.collide(update);
    window.addEventListener("keydown", this.onPassKey);
    window.addEventListener("keyup", this.onPassKey);
    window.addEventListener("blur", this.onPassBlur);
    const removeGuard = this.scene.preRender.addEventListener(this.guard);
    // A glide moves the camera at the start of every frame, before anything reads it.
    const removeGlide = this.scene.preUpdate.addEventListener(this.stepGlide);
    // A hand on the map takes the camera from a glide at once: a press, a wheel, a pinch.
    canvas.addEventListener("pointerdown", this.interruptGlide);
    canvas.addEventListener("wheel", this.interruptGlide, { passive: true });
    this.unsubscribe.push(
      () => {
        // The prototype's update again.
        delete (controller as { update?: unknown }).update;
        container.removeEventListener("wheel", this.onUserWheel, { capture: true });
        window.removeEventListener("pointermove", this.onUserDrag);
        container.removeEventListener("wheel", this.onSurfaceWheel, { capture: true });
        window.removeEventListener("keydown", this.onPassKey);
        window.removeEventListener("keyup", this.onPassKey);
        window.removeEventListener("blur", this.onPassBlur);
        removeGuard();
        removeGlide();
        canvas.removeEventListener("pointerdown", this.interruptGlide);
        canvas.removeEventListener("wheel", this.interruptGlide);
      },
      () => {
        canvas.removeEventListener("pointerdown", this.onPointerHeld);
        window.removeEventListener("pointerup", this.onPointerReleased);
        window.removeEventListener("pointercancel", this.onPointerReleased);
        canvas.removeEventListener("pointerdown", this.onOrbitStart);
        canvas.removeEventListener("pointerdown", this.onPanStart);
        canvas.removeEventListener("contextmenu", preventDefault);
        window.removeEventListener("pointermove", this.onOrbitMove);
        window.removeEventListener("pointerup", this.onOrbitEnd);
        window.removeEventListener("pointermove", this.onPanMove);
        window.removeEventListener("pointerup", this.onPanEnd);
        window.removeEventListener("pointercancel", this.onPanEnd);
        canvas.removeEventListener("wheel", this.stopPanInertia);
        this.stopPanInertia();
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
        this.levelHorizon();
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

  /**
   * The splats as surfaces (SplatCollider): the cursor's pick and the orbit pivot land on
   * them, a wheel toward one stops short of it, and the camera cannot move into one --
   * unless Space is held, to pass through to what is on the other side.
   */
  setCollider(collider: SplatCollider | null): void {
    this.collider = collider;
    this.lastGood = null;
  }

  /**
   * Space passes through surfaces on the map; while exploring it jumps instead, so the
   * explore controller turns this off for its duration.
   */
  setPassKeyEnabled(enabled: boolean): void {
    this.passKeyEnabled = enabled;
    if (!enabled) this.passThrough = false;
  }

  /** Whether Space is held to pass through surfaces. */
  get passingThrough(): boolean {
    return this.passThrough;
  }

  private readonly onPassKey = (event: KeyboardEvent): void => {
    if (event.code !== "Space" || !this.passKeyEnabled) return;
    if (event.type === "keyup") {
      this.passThrough = false;
      return;
    }
    // Space in a field types a space, and on a focused button presses it: leave both be.
    const target = event.target;
    if (isTyping(target)) return;
    if (target instanceof HTMLElement && target.closest("button, a, [role='button']")) return;
    event.preventDefault();
    this.passThrough = true;
  };

  private readonly onPassBlur = (): void => {
    this.passThrough = false;
  };

  /**
   * Once a frame, before drawing: a camera that moved into a splat surface is put back at
   * the surface, sliding along it (SplatCollider.resolve). Every way the camera moves --
   * Cesium's drag, pinch and inertia, the orbit here, the keyboard, explore mode -- comes
   * through here, so one check covers them all. Flights and jumps are left alone, and so is
   * everything while Space is held.
   */
  private readonly guard = (): void => {
    const camera = this.viewer.camera;
    const position = camera.positionWC;
    const flying = (camera as unknown as { _currentFlight?: unknown })._currentFlight !== undefined;
    if (
      !this.collider?.active ||
      this.passThrough ||
      flying ||
      !Matrix4.equals(camera.transform, Matrix4.IDENTITY) ||
      !this.lastGood
    ) {
      this.lastGood = Cartesian3.clone(position, this.lastGood ?? undefined);
      return;
    }
    if (Cartesian3.equalsEpsilon(position, this.lastGood, 0, 1e-7)) return;
    const { position: allowed, blocked } = this.collider.resolve(this.lastGood, position);
    if (blocked) {
      Cartesian3.clone(allowed, camera.position);
      this.hintPassThrough();
    }
    Cartesian3.clone(allowed, this.lastGood);
  };

  /** The first time a surface stops the camera, say how to go through it. Once a session. */
  private hintPassThrough(): void {
    if (this.hinted) return;
    this.hinted = true;
    this.events.emit("toast", {
      id: "pass-through",
      tone: "info",
      title: "Hold Space to pass through",
      body: "The camera stops at scanned surfaces. Hold Space while you move to go through one.",
    });
  }

  /**
   * Wheel toward a splat under the cursor: along the ray through the cursor (so the point
   * stays under it), each notch closing a share of the distance left, never nearer than
   * the clearance -- or through it with Space held. Anything else (the ground, a mesh, a
   * splat behind something solid) is left to Cesium's zoom.
   */
  private readonly onSurfaceWheel = (event: WheelEvent): void => {
    const collider = this.collider;
    if (!collider?.active || !this.scene.screenSpaceCameraController.enableInputs) return;
    if (event.target !== this.viewer.canvas) return;
    const rect = this.viewer.canvas.getBoundingClientRect();
    scratchWindow.x = event.clientX - rect.left;
    scratchWindow.y = event.clientY - rect.top;
    const camera = this.viewer.camera;
    const ray = camera.getPickRay(scratchWindow, scratchRay);
    if (!ray) return;
    const hit = collider.raycast(ray);
    if (!hit) return;
    if (this.occludedAt(scratchWindow, hit.distance)) return;
    event.preventDefault();
    event.stopPropagation();
    // Stopped here, it never reaches the canvas's own listener that ends a glide.
    this.endGlide("cancel");
    const pixels = event.deltaMode === 1 ? event.deltaY * 33 : event.deltaY;
    const notches = CesiumMath.clamp(pixels / 100, -3, 3);
    const next = hit.distance * Math.pow(SURFACE_ZOOM_STEP, -notches);
    const clearance = collider.clearance(hit.point);
    let step = hit.distance - next;
    if (!this.passThrough) step = Math.min(step, hit.distance - clearance);
    // Passing through: at the surface already, a notch carries the camera past it.
    else if (notches < 0 && hit.distance - next < clearance * 4) step = hit.distance + clearance;
    if (Math.abs(step) < 1e-6) return;
    camera.move(ray.direction, step);
    // A deliberate move toward a surface is not a collision to undo.
    this.lastGood = Cartesian3.clone(camera.positionWC, this.lastGood ?? undefined);
    this.scene.requestRender();
  };

  /**
   * Whether something solid (the depth buffer, the terrain) is nearer than the splat under
   * the cursor. The depth read is a pick pass and a GPU read-back, so it is made once per
   * wheel gesture at a point, not per notch: zooming along the ray scales both distances
   * alike, so the answer holds until the cursor moves or the wheel rests.
   */
  private occludedAt(window: Cartesian2, splatDistance: number): boolean {
    const now = performance.now();
    const last = this.wheelOcclusion;
    if (
      last &&
      now - last.at < WHEEL_GESTURE_MS &&
      Math.abs(last.x - window.x) + Math.abs(last.y - window.y) < WHEEL_SAME_POINT_PX
    ) {
      last.at = now;
      return last.occluded;
    }
    const solid = this.solidPick(window);
    const occluded =
      solid !== null && Cartesian3.distance(this.viewer.camera.positionWC, solid) < splatDistance;
    this.wheelOcclusion = { x: window.x, y: window.y, at: now, occluded };
    return occluded;
  }

  /**
   * The terrain's point under the cursor, found on the CPU. It used to be the nearer of that
   * and a depth read-back, but a read-back waits for the GPU to finish its frame -- 100 to
   * 300 ms of a busy one, on every wheel gesture -- and the question here is only whether
   * the ground is in front of the splat, which the terrain alone answers.
   */
  private solidPick(window: Cartesian2): Cartesian3 | null {
    const candidates: Cartesian3[] = [];
    const ray = this.viewer.camera.getPickRay(window, new Ray());
    const ground = ray ? this.scene.globe.pick(ray, this.scene, new Cartesian3()) : undefined;
    if (ground) candidates.push(ground);
    let best: Cartesian3 | null = null;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const candidate of candidates) {
      const height = Cartographic.fromCartesian(candidate, undefined, scratchCarto).height;
      if (height < PLAUSIBLE_HEIGHT_M[0] || height > PLAUSIBLE_HEIGHT_M[1]) continue;
      const distance = Cartesian3.distance(this.viewer.camera.positionWC, candidate);
      if (distance < bestDistance) {
        best = candidate;
        bestDistance = distance;
      }
    }
    return best;
  }

  get isMoving(): boolean {
    return this.moving;
  }

  /**
   * Google Maps never rolls the view: the horizon stays level whatever the gesture. Any
   * roll that crept in (touch tilt, inertia, an orbit near straight down) is removed here.
   */
  levelHorizon(): void {
    const camera = this.viewer.camera;
    if (!Matrix4.equals(camera.transform, Matrix4.IDENTITY)) return;
    // Cesium reports roll in [0, 2π): a hair of negative roll reads as nearly a full turn.
    const roll = camera.roll > Math.PI ? camera.roll - CesiumMath.TWO_PI : camera.roll;
    if (Math.abs(roll) < ROLL_TOLERANCE_RAD) return;
    camera.setView({ orientation: { heading: camera.heading, pitch: camera.pitch, roll: 0 } });
  }

  /**
   * Camera floor for meshes: Cesium's own collision against 3D Tiles ray-casts every loaded
   * tile's triangles on the CPU each frame (hundreds of milliseconds on a city mesh), so the
   * tilesets never enable it. Instead, once a camera the app put somewhere (a fly-to) has
   * come to rest, the drawn surface straight below it is read from the depth buffer (one pick
   * pass), and a camera that ended up under it or too close is eased back up.
   *
   * Never once the person has the camera (cameraOwnership.ts): it used to run after every
   * gesture too, and lifted a close-up the wheel had just made -- up to 40 m, onto whatever
   * the pick pass found above the camera (on production, 19.5 m onto a coarse world tile
   * beside the Pumpkin scan, 3 s after the wheel stopped).
   */
  private keepAboveDrawnSurface(): void {
    if (!this.ownership.mayCorrect) return;
    if (!this.scene.sampleHeightSupported) return;
    // A height sample is a pick pass plus a GPU read-back (measured as a third of a slow
    // drag's main-thread time when it fired between the mouse events of one gesture).
    const now = performance.now();
    if (this.pointerHeld || now - this.lastFloorCheckAt < FLOOR_CHECK_MIN_INTERVAL_MS) return;
    this.lastFloorCheckAt = now;
    const camera = this.viewer.camera;
    if (!Matrix4.equals(camera.transform, Matrix4.IDENTITY)) return;
    // Over a scan the splats' own solids keep the camera off them every frame (the guard in
    // preRender), with no read-back: nothing to check here.
    if (this.overSplatSurface()) return;
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
    // Eased in as well as out: the camera is at rest, and a lift that leaves at full speed
    // (Cesium's quadratic-out, as it was) is a jolt straight after a fly-to has landed.
    this.glide(
      {
        longitude: CesiumMath.toDegrees(carto.longitude),
        latitude: CesiumMath.toDegrees(carto.latitude),
        height: floor,
        heading: CesiumMath.toDegrees(camera.heading),
        pitch: CesiumMath.toDegrees(camera.pitch),
        ground: { height: surface, measured: true },
      },
      { durationS: FLOOR_LIFT_S },
    );
  }

  /** Whether a splat's solids lie straight below the camera, within a floor correction. */
  private overSplatSurface(): boolean {
    const collider = this.collider;
    if (!collider?.active) return false;
    const camera = this.viewer.camera;
    const down = this.scene.globe.ellipsoid.geodeticSurfaceNormal(
      camera.positionWC,
      new Cartesian3(),
    );
    Cartesian3.negate(down, down);
    const ray = new Ray(Cartesian3.clone(camera.positionWC), down);
    return collider.raycast(ray, MAX_FLOOR_LIFT_M * 10) !== null;
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
   * Google Maps mapping: Shift+drag, Ctrl+drag, right-drag and middle-drag orbit the point in
   * the middle of the view (the thing you are looking at), left-drag pans (onPanStart). At object scale a plain
   * left-drag orbits the point that was clicked, because Cesium's rotation is tuned for a
   * planet and barely turns beside a rock.
   */
  private readonly onPointerHeld = (): void => {
    this.pointerHeld = true;
    this.ownership.hold(true, performance.now());
  };

  private readonly onPointerReleased = (): void => {
    if (this.pointerHeld) this.ownership.hold(false, performance.now());
    this.pointerHeld = false;
  };

  /** A wheel notch over the map: the person has the camera. */
  private readonly onUserWheel = (): void => {
    this.ownership.input(performance.now());
  };

  /** A drag on the map (or a pinch) is still moving the camera while the pointer moves. */
  private readonly onUserDrag = (): void => {
    if (this.pointerHeld) this.ownership.input(performance.now());
  };

  /**
   * CesiumJS's controller update (`ScreenSpaceCameraController.update`), once a frame: the
   * person's gestures and their inertia, and the terrain collision. On a camera the person
   * has and is not moving, the collision is left out for the frame: it would lift the camera
   * when a finer terrain tile lands under a close-up after the wheel has stopped -- the
   * camera moving by itself a moment after the person stopped it (cameraOwnership.ts).
   */
  private collide(update: () => void): void {
    const controller = this.scene.screenSpaceCameraController;
    const camera = this.viewer.camera;
    const collide = controller.enableCollisionDetection;
    if (collide && !this.ownership.collisionAllowed(performance.now()))
      controller.enableCollisionDetection = false;
    const before = Cartesian3.clone(camera.positionWC, scratchBefore);
    const direction = Cartesian3.clone(camera.directionWC, scratchBeforeDirection);
    try {
      update();
    } finally {
      controller.enableCollisionDetection = collide;
    }
    // Coasting on the gesture's inertia is still the person moving it.
    this.ownership.controllerMoved(
      !Cartesian3.equalsEpsilon(before, camera.positionWC, 0, 1e-6) ||
        !Cartesian3.equalsEpsilon(direction, camera.directionWC, 1e-9),
    );
  }

  /**
   * Whether the person has touched the camera since the app last flew it: from then nothing
   * but they move it (cameraOwnership.ts). A fly-to's settle on a better pose asks first.
   */
  get userHasCamera(): boolean {
    return this.ownership.userHasCamera;
  }

  /**
   * The person takes the camera without a gesture on the map: walking or flying in explore
   * mode, whose own physics moves it each frame.
   */
  takenByUser(): void {
    this.ownership.input(performance.now());
  }

  private readonly onOrbitStart = (event: PointerEvent): void => {
    if (!this.scene.screenSpaceCameraController.enableInputs) return;
    const around =
      event.button === 1 ||
      event.button === 2 ||
      (event.button === 0 && (event.ctrlKey || event.shiftKey));
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
   * Left-drag pan, Google Maps style: the point under the cursor when the drag starts (a
   * splat surface, the terrain, a building) stays under the cursor, sliding on the level
   * plane through it. Cesium's own left-drag spins the globe around the terrain point under
   * the cursor -- under a splat that is the ground beneath it, or nothing towards the
   * horizon, where it rotates the whole view -- so near the ground this takes over, and the
   * planet-scale spin stays for high up. Mouse and pen only; touch keeps Cesium's gestures.
   */
  private readonly onPanStart = (event: PointerEvent): void => {
    this.stopPanInertia();
    this.grab = null;
    const controller = this.scene.screenSpaceCameraController;
    if (event.button !== 0 || this.objectScale) return;
    if (!controller.enableInputs) return;
    const plain = !event.ctrlKey && !event.shiftKey && !event.altKey && !event.metaKey;
    const near = this.pose().altitude < GRAB_PAN_MAX_ALTITUDE_M;
    scratchWindow.x = event.offsetX;
    scratchWindow.y = event.offsetY;
    const point =
      plain && near && event.pointerType !== "touch" ? this.plausiblePick(scratchWindow) : null;
    // Decided per gesture, and left as it is after: re-enabling Cesium's spin on release
    // would hand it the drag's last movement as inertia.
    controller.enableRotate = !point;
    if (!point) return;
    const camera = this.viewer.camera;
    this.grab = {
      point,
      normal: this.scene.globe.ellipsoid.geodeticSurfaceNormal(point, new Cartesian3()),
      reach: Cartesian3.distance(camera.positionWC, point),
    };
    this.panSamples = [];
  };

  private readonly onPanMove = (event: PointerEvent): void => {
    const grab = this.grab;
    if (!grab) return;
    const rect = this.viewer.canvas.getBoundingClientRect();
    scratchWindow.x = event.clientX - rect.left;
    scratchWindow.y = event.clientY - rect.top;
    const camera = this.viewer.camera;
    const ray = camera.getPickRay(scratchWindow, scratchGrabRay);
    if (!ray) return;
    const facing = Cartesian3.dot(ray.direction, grab.normal);
    if (Math.abs(facing) < 1e-4) return;
    const offset = Cartesian3.subtract(grab.point, ray.origin, scratchGrabHit);
    const t = Cartesian3.dot(offset, grab.normal) / facing;
    if (t <= 0 || t > grab.reach * GRAB_PAN_MAX_REACH) return;
    const hit = Ray.getPoint(ray, t, scratchGrabHit);
    const step = Cartesian3.subtract(grab.point, hit, new Cartesian3());
    this.movePan(step);
    const now = performance.now();
    this.panSamples.push({ at: now, step });
    while ((this.panSamples[0]?.at ?? now) < now - PAN_VELOCITY_WINDOW_MS) this.panSamples.shift();
  };

  private readonly onPanEnd = (): void => {
    const grab = this.grab;
    this.grab = null;
    if (!grab) return;
    const now = performance.now();
    const recent = this.panSamples.filter((sample) => sample.at >= now - PAN_VELOCITY_WINDOW_MS);
    this.panSamples = [];
    const first = recent[0];
    if (!first || recent.length < 2) return;
    // Metres per millisecond over the last few moves, then coasting to a stop.
    const velocity = new Cartesian3();
    for (const sample of recent) Cartesian3.add(velocity, sample.step, velocity);
    Cartesian3.divideByScalar(velocity, Math.max(now - first.at, 16), velocity);
    const minSpeed = (grab.reach * PAN_INERTIA_MIN_SPEED) / 1000;
    let last = now;
    const coast = (time: number): void => {
      const dt = Math.min(time - last, 50);
      last = time;
      Cartesian3.multiplyByScalar(velocity, Math.exp(-dt / PAN_INERTIA_TAU_MS), velocity);
      if (Cartesian3.magnitude(velocity) < minSpeed) {
        this.panInertia = 0;
        return;
      }
      this.movePan(Cartesian3.multiplyByScalar(velocity, dt, scratchGrabStep));
      this.panInertia = requestAnimationFrame(coast);
    };
    this.panInertia = requestAnimationFrame(coast);
  };

  private readonly stopPanInertia = (): void => {
    if (this.panInertia) cancelAnimationFrame(this.panInertia);
    this.panInertia = 0;
  };

  private movePan(step: Cartesian3): void {
    const camera = this.viewer.camera;
    Cartesian3.add(camera.position, step, camera.position);
    this.scene.requestRender();
  }

  /**
   * Turns the camera around a pivot, keeping its distance: `headingRad` moves the camera to
   * the left around it (the foreground follows a rightward drag, as in Cesium and Google
   * Maps), `pitchDeltaRad` is added to the camera pitch (positive tips towards the horizon,
   * negative towards straight down) and stops at both.
   */
  orbit(pivot: Cartesian3, headingRad: number, pitchDeltaRad: number): void {
    this.ownership.input(performance.now());
    this.endGlide("cancel");
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
    // Turning around the frame's up axis (not the camera's own up) is what keeps the horizon
    // level: rotating around a pitched camera's up vector rolls the view a little every drag.
    camera.lookAtTransform(Transforms.eastNorthUpToFixedFrame(pivot, undefined, scratchFrame));
    const previousAxis = camera.constrainedAxis;
    camera.constrainedAxis = Cartesian3.UNIT_Z;
    camera.rotateLeft(headingRad);
    camera.rotateUp(allowedTilt);
    camera.constrainedAxis = previousAxis;
    camera.lookAtTransform(Matrix4.IDENTITY);
    this.levelHorizon();
    this.scene.requestRender();
  }

  /**
   * The point the view is centred on: whatever the depth buffer has there (a model, a
   * building, the ground), else the terrain, else a point along the view direction at the
   * current height. Gaussian splats write no depth, so under a splat this is its ground.
   */
  /** The ground point in the middle of the view as lon/lat degrees, with the viewport size. */
  viewCenter(): { longitude: number; latitude: number; width: number; height: number } | null {
    const pivot = this.pivotAtCenter();
    if (!pivot) return null;
    const carto = Cartographic.fromCartesian(pivot, undefined, scratchCarto);
    return {
      longitude: CesiumMath.toDegrees(carto.longitude),
      latitude: CesiumMath.toDegrees(carto.latitude),
      width: this.viewer.canvas.clientWidth,
      height: this.viewer.canvas.clientHeight,
    };
  }

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

  /**
   * Splat pick (splats write no depth: SplatCollider), terrain pick and -- only when no splat
   * is under the cursor -- a depth pick, the nearest that is not on a placeholder tile. The
   * first two run on the CPU; the depth pick waits for the GPU (100 to 300 ms of a busy
   * frame at every gesture start), so it is kept for meshes and the world, which only depth
   * describes.
   */
  private plausiblePick(window: Cartesian2): Cartesian3 | null {
    const candidates: Cartesian3[] = [];
    const ray = this.viewer.camera.getPickRay(window, scratchRay);
    const splat = ray ? this.collider?.raycast(ray) : undefined;
    if (splat) candidates.push(splat.point);
    const ground = ray ? this.scene.globe.pick(ray, this.scene, new Cartesian3()) : undefined;
    if (ground) candidates.push(ground);
    if (!splat && this.scene.pickPositionSupported) {
      const depth = this.scene.pickPosition(window, new Cartesian3());
      if (depth) candidates.push(depth);
    }
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
    this.ownership.input(performance.now());
    this.endGlide("cancel");
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
    this.ownership.input(performance.now());
    this.endGlide("cancel");
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
    this.endGlide("cancel");
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
    // At object scale the thing under the crosshair is the model, not the globe: a splat's
    // solids answer on the CPU; otherwise read the depth buffer (a mesh), so the mm/px
    // readout describes the object the user is looking at.
    const centreRay = this.objectScale ? this.viewer.camera.getPickRay(scratchCenter) : undefined;
    const splatHit = centreRay ? this.collider?.raycast(centreRay) : undefined;
    if (splatHit) return splatHit.distance;
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
    this.ownership.appFlew();
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

  /**
   * How many flights this controller has started. Code that flew the camera and may steer it
   * again later (SiteManager settling on a better pose after landing) compares it with the
   * count after its own flight: a newer flight -- an object flown to, a search result -- has
   * the camera now.
   */
  get flights(): number {
    return this.flightCount;
  }

  /** Where a flight's destination prefetch comes from (`prefetchScanDestination`). */
  setDestinationPrefetch(prefetch: ((destination: ScanDestination) => () => void) | null): void {
    this.prefetch = prefetch;
  }

  flyTo(longitude: number, latitude: number, height: number, options: FlyOptions = {}): void {
    this.ownership.appFlew();
    this.flightCount += 1;
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
      cancel: options.onCancel,
    });
  }

  /**
   * Flies to `pose` as one continuous flight that can be pointed elsewhere on the way
   * (glide.ts): the site fly-to, whose destination keeps improving while it flies. The camera
   * leaves and lands at rest, never goes below the ground the globe has under it, and a
   * `retarget` moves the destination without a jump in position or speed. Anything else that
   * takes the camera -- another flight (Cesium's `flyTo` included), a press or a wheel on the
   * map, the keyboard, explore mode -- ends it, with `onCancel`.
   *
   * Cesium's tilesets preload the destination's tiles during it, as for one of Cesium's own
   * flights (`preloadFlightDestinations`): the glide stands in for one on the camera.
   */
  glide(pose: GlidePose, options: GlideOptions = {}): GlideHandle {
    this.ownership.appFlew();
    this.flightCount += 1;
    // A glide that replaces one under way takes its motion over rather than stopping dead.
    const carry = this.motion();
    this.endGlide("cancel");
    const camera = this.viewer.camera;
    camera.cancelFlight();
    const start = camera.positionCartographic;
    const globe = this.scene.globe;
    const glide = new Glide(
      {
        longitude: CesiumMath.toDegrees(start.longitude),
        latitude: CesiumMath.toDegrees(start.latitude),
        height: start.height,
        heading: CesiumMath.toDegrees(camera.heading),
        pitch: CesiumMath.toDegrees(camera.pitch),
      },
      pose,
      {
        ground: (longitude, latitude) =>
          globe.getHeight(Cartographic.fromDegrees(longitude, latitude, 0, scratchCarto)),
      },
      this.glideClock(),
      { durationS: options.durationS, startGround: this.lastSurfaceHeight, carry },
    );
    const handle = {
      active: true,
      retarget: (next: GlidePose) => {
        if (this.current?.handle !== handle) return;
        glide.retarget(next);
        this.preloadDestination(glide.aim);
      },
      cancel: () => {
        if (this.current?.handle === handle) this.endGlide("cancel");
      },
    };
    const sentinel = {
      cancelTween: () => {
        if (this.current?.sentinel === sentinel) this.endGlide("cancel");
      },
    };
    this.current = { glide, options, sentinel, handle, trail: [] };
    (camera as unknown as { _currentFlight?: unknown })._currentFlight = sentinel;
    this.preloadDestination(glide.aim);
    this.scene.requestRender();
    return handle;
  }

  /**
   * The clock glides run on: `performance.now` (null), or a test's, which steps a fixed time
   * per frame so a flight's path is measured frame by frame at 60 fps on any machine (e2e
   * siteFlight.spec.ts, under software GL).
   */
  setGlideClock(clock: (() => number) | null): void {
    this.glideClock = clock ?? (() => performance.now());
  }

  /** Whether a glide (`glide`) has the camera. */
  get gliding(): boolean {
    return this.current !== null;
  }

  /**
   * Tells Cesium's tilesets where the glide is going, as `camera.flyTo` does for its own
   * flights, so the destination's tiles load during the flight (the preload flight pass).
   */
  private preloadDestination(pose: ArrivalPose): void {
    const scene = this.scene as Scene & {
      preloadFlightCamera?: CesiumCamera;
      preloadFlightCullingVolume?: unknown;
    };
    const preload = scene.preloadFlightCamera;
    if (!preload) return;
    preload.setView({
      destination: Cartesian3.fromDegrees(pose.longitude, pose.latitude, pose.height),
      orientation: {
        heading: CesiumMath.toRadians(pose.heading),
        pitch: CesiumMath.toRadians(pose.pitch),
        roll: 0,
      },
    });
    scene.preloadFlightCullingVolume = preload.frustum.computeCullingVolume(
      preload.positionWC,
      preload.directionWC,
      preload.upWC,
    );
  }

  /** Each frame, first thing: the glide's pose for this moment, and its arrival. */
  private readonly stepGlide = (): void => {
    const current = this.current;
    if (!current) return;
    const at = this.glideClock();
    const pose = current.glide.step(at);
    const position = Cartesian3.fromDegrees(pose.longitude, pose.latitude, pose.height);
    this.viewer.camera.setView({
      destination: position,
      orientation: {
        heading: CesiumMath.toRadians(pose.heading),
        pitch: CesiumMath.toRadians(pose.pitch),
        roll: 0,
      },
    });
    current.trail = [
      ...current.trail.slice(-1),
      { at, position, heading: pose.heading, pitch: pose.pitch },
    ];
    if (current.glide.done) this.endGlide("complete");
  };

  /** How the glide under way is moving now, from its last two frames; none without one. */
  private motion(): GlideCarry | undefined {
    const [a, b] = this.current?.trail ?? [];
    if (!a || !b) return undefined;
    const dt = (b.at - a.at) / 1000;
    if (!(dt > 0)) return undefined;
    const turn = ((((b.heading - a.heading + 180) % 360) + 360) % 360) - 180;
    return {
      velocity: {
        x: (b.position.x - a.position.x) / dt,
        y: (b.position.y - a.position.y) / dt,
        z: (b.position.z - a.position.z) / dt,
      },
      heading: turn / dt,
      pitch: (b.pitch - a.pitch) / dt,
    };
  }

  /** A press or a wheel on the map: the person takes the camera. */
  private readonly interruptGlide = (): void => {
    if (this.current) this.endGlide("cancel");
  };

  /** Ends the glide under way, if any, and says how. */
  private endGlide(how: "complete" | "cancel"): void {
    const current = this.current;
    if (!current) return;
    this.current = null;
    current.handle.active = false;
    const camera = this.viewer.camera as unknown as { _currentFlight?: unknown };
    if (camera._currentFlight === current.sentinel) camera._currentFlight = undefined;
    if (how === "complete") {
      current.options.onComplete?.();
      void this.surfaceIfUnderground();
    } else current.options.onCancel?.();
  }

  /**
   * The last guard against a black arrival: once a glide has landed, the terrain under the
   * camera is sampled at full detail, and a camera that ended up underground -- whatever aimed
   * it there -- glides up to stand over it, if nothing has taken the camera meanwhile. The
   * terrain the globe has loaded is not asked: on landing it can still be a coarse tile,
   * hundreds of metres off in the mountains.
   */
  private async surfaceIfUnderground(): Promise<void> {
    const provider = this.viewer.terrainProvider as { availability?: unknown };
    if (!provider.availability) return;
    const camera = this.viewer.camera;
    const at = Cartographic.clone(camera.positionCartographic);
    const position = Cartesian3.clone(camera.positionWC);
    const { heading, pitch } = camera;
    let terrain: number | undefined;
    try {
      const [sample] = await sampleTerrainMostDetailed(this.viewer.terrainProvider, [
        Cartographic.clone(at),
      ]);
      terrain = sample?.height;
    } catch {
      return;
    }
    const height = surfacedHeight(at.height, terrain);
    // A hand on the map since the landing has the camera, moved or not.
    if (height === null || this.current || !this.ownership.mayCorrect) return;
    const untouched =
      Cartesian3.distance(position, camera.positionWC) < 0.01 && camera.heading === heading;
    if (!untouched) return;
    this.glide(
      {
        longitude: CesiumMath.toDegrees(at.longitude),
        latitude: CesiumMath.toDegrees(at.latitude),
        height,
        heading: CesiumMath.toDegrees(heading),
        pitch: CesiumMath.toDegrees(pitch),
        ground: { height: height - SURFACED_M, measured: true },
      },
      { durationS: SURFACE_S },
    );
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
    this.ownership.appFlew();
    this.flightCount += 1;
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

  /** How far from a sphere's centre `flyToBoundingSphere` arrives. */
  private arrivalRange(sphere: BoundingSphere, rangeMultiplier = 3.2): number {
    // Whole sites never arrive closer than 30 m; a hand-sized object arrives at a few
    // times its own radius so it fills the view.
    const floor = sphere.radius < OBJECT_ARRIVAL_RADIUS_M ? 0.3 : 30;
    return Math.max(sphere.radius * rangeMultiplier, floor);
  }

  /**
   * Where `flyToBoundingSphere` would put the camera, as a pose `flyTo` can fly to: for a
   * flight that may be re-pointed on the way (SiteManager.flyTo), which needs to know where
   * it is going and to compare one destination with the next.
   */
  sphereArrival(
    sphere: BoundingSphere,
    options: { heading?: number; pitch?: number; rangeMultiplier?: number } = {},
  ): ArrivalPose {
    const range = this.arrivalRange(sphere, options.rangeMultiplier);
    const heading = options.heading ?? 100;
    const pitch = options.pitch ?? DEFAULT_ARRIVAL_PITCH;
    const h = CesiumMath.toRadians(heading);
    const p = CesiumMath.toRadians(pitch);
    // Looking along (east sin h cos p, north cos h cos p, up sin p) at the centre, from
    // `range` back along that line, in the centre's east-north-up frame.
    const local = new Cartesian3(
      -Math.sin(h) * Math.cos(p) * range,
      -Math.cos(h) * Math.cos(p) * range,
      -Math.sin(p) * range,
    );
    const frame = Transforms.eastNorthUpToFixedFrame(sphere.center);
    const position = Cartographic.fromCartesian(
      Matrix4.multiplyByPoint(frame, local, new Cartesian3()),
    );
    return {
      longitude: CesiumMath.toDegrees(position.longitude),
      latitude: CesiumMath.toDegrees(position.latitude),
      height: position.height,
      heading,
      pitch,
    };
  }

  /** Steep approach to a sphere, looking down at it: the standard "arrive at a site" move. */
  flyToBoundingSphere(
    sphere: BoundingSphere,
    options: FlyOptions & { rangeMultiplier?: number } = {},
  ): void {
    this.ownership.appFlew();
    this.flightCount += 1;
    const range = this.arrivalRange(sphere, options.rangeMultiplier);
    const pitch = CesiumMath.toRadians(options.pitch ?? DEFAULT_ARRIVAL_PITCH);
    const arrivalHeight =
      Cartographic.fromCartesian(sphere.center).height + range * Math.sin(-pitch);
    this.viewer.camera.flyToBoundingSphere(sphere, {
      offset: new HeadingPitchRange(CesiumMath.toRadians(options.heading ?? 100), pitch, range),
      duration: options.durationS ?? this.durationFor(sphere.center),
      easingFunction: EasingFunction.QUADRATIC_IN_OUT,
      pitchAdjustHeight: this.pitchAdjustHeight(arrivalHeight),
      complete: options.onComplete,
      cancel: options.onCancel,
    });
  }

  /**
   * Flies to an object of a scan (scene selection's Fly to, an object picked in the objects
   * panel): looks down at it from `pitch`, keeping the heading the camera has, at the range,
   * pace and pitch curve of every fly-to here. A scan drawn by a dedicated renderer fetches
   * what the destination shows during the flight, as a site fly-to does; the fetch ends on
   * arrival by itself, or when another flight cancels this one. A site flight still settling
   * leaves the camera to it (`flights`).
   */
  flyToObject(sphere: BoundingSphere, options: { pitch?: number } = {}): void {
    const heading = CesiumMath.toDegrees(this.viewer.camera.heading);
    const pitch = options.pitch ?? OBJECT_ARRIVAL_PITCH;
    const cancelPrefetch = this.prefetch?.({
      boundingSphere: sphere,
      offset: new HeadingPitchRange(
        CesiumMath.toRadians(heading),
        CesiumMath.toRadians(pitch),
        this.arrivalRange(sphere),
      ),
    });
    this.flyToBoundingSphere(sphere, { heading, pitch, onCancel: cancelPrefetch });
  }

  /** Rotates the view so north is up while keeping the position and tilt. */
  resetNorth(): void {
    this.ownership.appFlew();
    this.flightCount += 1;
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
    this.ownership.appFlew();
    this.flightCount += 1;
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
    this.ownership.input(performance.now());
    const distance = this.distanceToSurfaceAtCenter() ?? this.pose().altitude;
    const amount = distance * factor;
    if (amount > 0) this.viewer.camera.zoomIn(amount);
    else this.viewer.camera.zoomOut(-amount);
  }

  /** Home: the whole planet, oriented towards the given longitude. */
  flyHome(longitude = -110, latitude = 30): void {
    this.ownership.appFlew();
    this.flightCount += 1;
    this.viewer.camera.flyTo({
      destination: Cartesian3.fromDegrees(longitude, latitude, 18_000_000),
      orientation: { heading: 0, pitch: -CesiumMath.PI_OVER_TWO, roll: 0 },
      duration: 2.5,
      easingFunction: EasingFunction.QUADRATIC_IN_OUT,
    });
  }

  cancelFlight(): void {
    this.endGlide("cancel");
    this.viewer.camera.cancelFlight();
  }

  /** How long a flight from here to `destination` takes (s): every fly-to's own pace. */
  durationFor(destination: Cartesian3): number {
    const distance = Cartesian3.distance(this.viewer.camera.positionWC, destination);
    // ~1.5 s for a local hop, ~5 s from orbit; never sluggish.
    return CesiumMath.clamp(1.2 + Math.log10(Math.max(distance, 10)) * 0.55, 1.2, 5.5);
  }

  destroy(): void {
    this.endGlide("cancel");
    this.reportPose.cancel();
    for (const timer of this.settleTimers) clearTimeout(timer);
    clearInterval(this.idleRefresh);
    this.setObjectScale(false);
    for (const off of this.unsubscribe) off();
  }
}
