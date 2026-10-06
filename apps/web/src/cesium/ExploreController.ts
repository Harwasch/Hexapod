import {
  Cartesian3,
  Cartographic,
  Math as CesiumMath,
  Matrix4,
  Ray,
  Transforms,
  type Scene,
  type CesiumWidget,
} from "cesium";

import type { Emitter } from "@/lib/emitter";
import {
  WALK,
  step,
  type MoveInput,
  type MoveMode,
  type MoveState,
  type MoveWorld,
} from "@/lib/firstPerson";
import { isTyping } from "@/lib/hotkeys";
import type { Vec3 } from "@/lib/occupancy";

import type { CameraController } from "./CameraController";
import type { SplatCollider } from "./SplatCollider";
import type { SceneEvents } from "./types";

/** Degrees of turn per pixel of mouse movement at a 120 degree field of view, scaled by the
 *  field of view (SuperSplat's viewer: 0.15 degrees per pixel times fov / 120). */
const LOOK_DEG_PER_PX_AT_120 = 0.15;
const MAX_PITCH = CesiumMath.toRadians(89);
/** How far down a ground is searched for under the walker (metres). */
const GROUND_SEARCH_M = 200;
/** Where the ray straight down finds a gap in a scan's ground, rings of `FOOT_RAYS` rays
 *  this far out (m) are tried in turn: a foot, not a point, stands on sparse ground. */
const FOOT_RINGS_M = [0.25, 0.5];
const FOOT_RAYS = 8;
/** Falling longer than this (seconds) with nothing below turns walking into hovering. */
const MAX_FALL_S = 2.5;
/** Moves that change the camera less than this (metres) are not rendered. */
const STILL_M = 1e-5;

const MOVE_KEYS = new Set([
  "KeyW",
  "KeyA",
  "KeyS",
  "KeyD",
  "ArrowUp",
  "ArrowDown",
  "ArrowLeft",
  "ArrowRight",
  "ShiftLeft",
  "ShiftRight",
  "Space",
  "KeyE",
  "KeyQ",
  "KeyC",
  "ControlLeft",
  "ControlRight",
]);

export interface ExploreStatus {
  mode: MoveMode;
  /** The mouse is captured for looking (pointer lock). */
  looking: boolean;
  onGround: boolean;
}

/**
 * First-person exploring: walk the scan at eye height with gravity, or fly. Same scene, same
 * camera; the map controller is paused, not replaced.
 *
 * - **Look**: click the view to capture the mouse (pointer lock); moving it turns the view,
 *   Esc gives it back. Dragging turns the view too, for trackpads and when the capture is
 *   refused.
 * - **Move**: W A S D (or the arrows) relative to where you look -- level when walking --
 *   Shift to sprint, Ctrl to go slow, Space to jump; F (or the HUD button) switches
 *   between walking and flying, where E / Space rise and Q / C sink.
 * - **Collide**: the ground is the scan's own splats under you (SplatCollider), else the
 *   terrain; walls and hedges stop you and you slide along them (the camera controller's
 *   per-frame check covers every move, this one's included).
 *
 * The movement itself (speeds, easing, gravity, steps) is `lib/firstPerson.ts`, in an
 * east/north/up frame fixed where exploring began.
 */
export class ExploreController {
  private readonly scene: Scene;
  private active = false;
  private mode: MoveMode = "walk";
  private readonly keys = new Set<string>();
  private jumpQueued = false;
  private pace = 1;
  private lastTick = 0;
  private dragging: { x: number; y: number } | null = null;
  private removeTick: (() => void) | null = null;
  private collider: SplatCollider | null = null;
  private cameraController: CameraController | null = null;
  private state: MoveState = { position: [0, 0, 0], velocity: [0, 0, 0], onGround: false };
  private heading = 0;
  private pitch = 0;
  private airborneS = 0;
  /** Local east/north/up metres to world, and back, fixed where exploring began. */
  private readonly toWorld = new Matrix4();
  private readonly toLocal = new Matrix4();

  constructor(
    private readonly viewer: CesiumWidget,
    private readonly events: Emitter<SceneEvents>,
  ) {
    this.scene = viewer.scene;
  }

  /** Scanned surfaces to walk on and bump into (SplatCollider). */
  setCollider(collider: SplatCollider | null): void {
    this.collider = collider;
  }

  /** The map camera, whose Space-to-pass-through gives way to jumping while exploring. */
  setCameraController(camera: CameraController): void {
    this.cameraController = camera;
  }

  get isActive(): boolean {
    return this.active;
  }

  /** Walking speed in m/s (sprint and flying scale from it). */
  get currentSpeed(): number {
    return WALK.speed * this.pace;
  }

  setSpeed(metersPerSecond: number): void {
    this.pace = CesiumMath.clamp(metersPerSecond, 0.1, 200) / WALK.speed;
  }

  get status(): ExploreStatus {
    return {
      mode: this.mode,
      looking: document.pointerLockElement === this.viewer.canvas,
      onGround: this.state.onGround,
    };
  }

  setMode(mode: MoveMode): void {
    if (this.mode === mode) return;
    this.mode = mode;
    // Taking off keeps the height; landing lets gravity bring you down.
    this.state = { ...this.state, onGround: false };
    this.announce();
  }

  toggleMode(): void {
    this.setMode(this.mode === "walk" ? "fly" : "walk");
  }

  enter(speed?: number): void {
    if (this.active) return;
    if (speed) this.setSpeed(speed);
    this.active = true;
    this.scene.screenSpaceCameraController.enableInputs = false;
    this.cameraController?.setPassKeyEnabled(false);
    // The person has the camera: nothing else moves it now (cameraOwnership.ts), the
    // terrain collision included, which would lift a walker its own ground has put under it.
    this.cameraController?.takenByUser();
    const camera = this.viewer.camera;
    // The walk has the camera now: a flight under way (a site's glide included) stops here.
    camera.cancelFlight();
    // The frame the walk is measured in: east/north/up at the camera's own spot.
    Transforms.eastNorthUpToFixedFrame(camera.positionWC, undefined, this.toWorld);
    Matrix4.inverseTransformation(this.toWorld, this.toLocal);
    this.heading = camera.heading;
    this.pitch = CesiumMath.clamp(camera.pitch, -MAX_PITCH, MAX_PITCH);
    this.state = { position: [0, 0, 0], velocity: [0, 0, 0], onGround: false };
    // Start on the ground when it is close below; from high up, fly down to it.
    const ground = this.world().groundBelow(0, 0, 0, GROUND_SEARCH_M);
    this.mode = ground !== null && -ground < WALK.eyeHeight * 6 ? "walk" : "fly";
    if (this.mode === "walk" && ground !== null) {
      this.state.position = [0, 0, ground + WALK.eyeHeight];
      this.state.onGround = true;
    }
    this.apply();
    this.viewer.canvas.style.cursor = "crosshair";
    this.lastTick = performance.now();
    // preUpdate fires every widget tick even in request-render mode; preRender would not.
    this.removeTick = this.scene.preUpdate.addEventListener(() => this.tick());
    window.addEventListener("keydown", this.onKeyDown, { capture: true });
    window.addEventListener("keyup", this.onKeyUp, { capture: true });
    window.addEventListener("blur", this.onBlur);
    document.addEventListener("pointerlockchange", this.onLockChange);
    const canvas = this.viewer.canvas;
    canvas.addEventListener("pointerdown", this.onPointerDown);
    window.addEventListener("pointermove", this.onPointerMove);
    window.addEventListener("pointerup", this.onPointerUp);
    this.events.emit("explore", true);
  }

  exit(): void {
    if (!this.active) return;
    this.active = false;
    this.keys.clear();
    this.scene.screenSpaceCameraController.enableInputs = true;
    this.cameraController?.setPassKeyEnabled(true);
    this.viewer.canvas.style.cursor = "";
    if (document.pointerLockElement === this.viewer.canvas) document.exitPointerLock();
    this.removeTick?.();
    this.removeTick = null;
    window.removeEventListener("keydown", this.onKeyDown, { capture: true });
    window.removeEventListener("keyup", this.onKeyUp, { capture: true });
    window.removeEventListener("blur", this.onBlur);
    document.removeEventListener("pointerlockchange", this.onLockChange);
    const canvas = this.viewer.canvas;
    canvas.removeEventListener("pointerdown", this.onPointerDown);
    window.removeEventListener("pointermove", this.onPointerMove);
    window.removeEventListener("pointerup", this.onPointerUp);
    this.events.emit("explore", false);
  }

  private announce(): void {
    this.events.emit("explore", true);
  }

  private readonly onKeyDown = (event: KeyboardEvent): void => {
    // Ctrl is the slow pace here, so only Ctrl with another key (a shortcut) is let through.
    if (isTyping(event.target) || event.metaKey || event.altKey) return;
    if (event.ctrlKey && !event.code.startsWith("Control")) return;
    if (event.code === "Escape") {
      // The first Esc gives the mouse back (the browser does that itself); the next leaves.
      if (document.pointerLockElement === this.viewer.canvas) return;
      event.preventDefault();
      this.exit();
      return;
    }
    if (event.code === "KeyF" && !event.repeat) {
      event.preventDefault();
      this.toggleMode();
      return;
    }
    if (!MOVE_KEYS.has(event.code)) return;
    event.preventDefault();
    event.stopPropagation();
    if (event.code === "Space" && !event.repeat) this.jumpQueued = true;
    this.keys.add(event.code);
    this.scene.requestRender();
  };

  private readonly onKeyUp = (event: KeyboardEvent): void => {
    this.keys.delete(event.code);
  };

  private readonly onBlur = (): void => {
    this.keys.clear();
  };

  private readonly onLockChange = (): void => {
    this.announce();
  };

  private readonly onPointerDown = (event: PointerEvent): void => {
    if (event.button !== 0) return;
    this.dragging = { x: event.clientX, y: event.clientY };
    if (document.pointerLockElement !== this.viewer.canvas) {
      // Capture the mouse for looking; a refusal (an embedded frame, a browser setting)
      // leaves drag-to-look, which works the same.
      const request = this.viewer.canvas.requestPointerLock() as unknown;
      if (request instanceof Promise) request.catch(() => undefined);
    }
  };

  private readonly onPointerMove = (event: PointerEvent): void => {
    const locked = document.pointerLockElement === this.viewer.canvas;
    let dx: number;
    let dy: number;
    if (locked) {
      dx = event.movementX;
      dy = event.movementY;
    } else if (this.dragging) {
      dx = event.clientX - this.dragging.x;
      dy = event.clientY - this.dragging.y;
      this.dragging = { x: event.clientX, y: event.clientY };
    } else return;
    const frustum = this.viewer.camera.frustum as { fov?: number };
    const fovDeg = CesiumMath.toDegrees(frustum.fov ?? CesiumMath.PI_OVER_THREE);
    const rate = CesiumMath.toRadians(LOOK_DEG_PER_PX_AT_120 * (fovDeg / 120));
    this.heading = CesiumMath.zeroToTwoPi(this.heading + dx * rate);
    this.pitch = CesiumMath.clamp(this.pitch - dy * rate, -MAX_PITCH, MAX_PITCH);
    this.apply();
    this.scene.requestRender();
  };

  private readonly onPointerUp = (): void => {
    this.dragging = null;
  };

  private input(): MoveInput {
    const held = (...codes: string[]): boolean => codes.some((code) => this.keys.has(code));
    const axis = (plus: boolean, minus: boolean): number => (plus ? 1 : 0) - (minus ? 1 : 0);
    const input: MoveInput = {
      forward: axis(held("KeyW", "ArrowUp"), held("KeyS", "ArrowDown")),
      right: axis(held("KeyD", "ArrowRight"), held("KeyA", "ArrowLeft")),
      up: axis(held("Space", "KeyE"), held("KeyC", "KeyQ")),
      sprint: held("ShiftLeft", "ShiftRight"),
      slow: held("ControlLeft", "ControlRight"),
      jump: this.jumpQueued,
    };
    this.jumpQueued = false;
    return input;
  }

  /** The ground and walls the walker meets, in the local frame. */
  /** The scan's ground below a point: the ray straight down, else the highest of rings of
   *  rays around it (`FOOT_RINGS_M`); null when none meets a splat within `depth`. */
  private scanGroundBelow(
    from: Cartesian3,
    down: Cartesian3,
    depth: number,
    local: (x: number, y: number, z: number) => Cartesian3,
    x: number,
    y: number,
    fromZ: number,
  ): number | null {
    const collider = this.collider;
    if (!collider) return null;
    const centre = collider.raycast(new Ray(from, down), depth);
    if (centre) return fromZ - centre.distance;
    for (const radius of FOOT_RINGS_M) {
      let best: number | null = null;
      for (let k = 0; k < FOOT_RAYS; k++) {
        const angle = (k / FOOT_RAYS) * Math.PI * 2;
        const start = local(x + Math.cos(angle) * radius, y + Math.sin(angle) * radius, fromZ);
        const hit = collider.raycast(new Ray(start, down), depth);
        if (hit) best = Math.max(best ?? Number.NEGATIVE_INFINITY, fromZ - hit.distance);
      }
      if (best !== null) return best;
    }
    return null;
  }

  private world(): MoveWorld {
    const local = (x: number, y: number, z: number): Cartesian3 =>
      Matrix4.multiplyByPoint(this.toWorld, new Cartesian3(x, y, z), new Cartesian3());
    return {
      groundBelow: (x, y, fromZ, depth) => {
        const from = local(x, y, fromZ);
        const down = Matrix4.multiplyByPointAsVector(
          this.toWorld,
          new Cartesian3(0, 0, -1),
          new Cartesian3(),
        );
        // The scan's own ground first: it is what you see, wherever it was placed. Straight
        // down, then -- where that ray finds a gap (a scan's ground is sparse in places, and
        // its voxels are centimetres) -- rings a foot's width out, the highest they meet.
        const ground = this.scanGroundBelow(from, down, depth, local, x, y, fromZ);
        if (ground !== null) return ground;
        if (!this.scene.globe.show) return null;
        const carto = Cartographic.fromCartesian(from);
        const terrain = this.scene.globe.getHeight(carto);
        if (terrain === undefined) return null;
        const below = carto.height - terrain;
        return below >= 0 && below <= depth ? fromZ - below : null;
      },
      sweep: (from, to) => {
        if (!this.collider?.active || this.cameraController?.passingThrough) return to;
        const resolved = this.collider.resolve(local(...from), local(...to)).position;
        const back = Matrix4.multiplyByPoint(this.toLocal, resolved, new Cartesian3());
        return [back.x, back.y, back.z] as Vec3;
      },
    };
  }

  private tick(): void {
    const now = performance.now();
    const dt = (now - this.lastTick) / 1000;
    this.lastTick = now;
    const before = this.state.position;
    const small = this.cameraController?.isObjectScale ? 0.05 : 1;
    this.state = step(
      this.state,
      this.input(),
      { heading: this.heading, pitch: this.pitch },
      this.mode,
      this.world(),
      dt,
      small,
      this.pace,
    );
    // Fallen for a while with nothing underneath (walked off the edge of the scan): hover.
    this.airborneS = this.state.onGround || this.mode === "fly" ? 0 : this.airborneS + dt;
    if (this.airborneS > MAX_FALL_S) this.setMode("fly");
    const [x, y, z] = this.state.position;
    if (Math.hypot(x - before[0], y - before[1], z - before[2]) < STILL_M) return;
    this.apply();
    this.scene.requestRender();
  }

  /** Puts the camera where the walker is, looking where it looks. */
  private apply(): void {
    const [x, y, z] = this.state.position;
    const destination = Matrix4.multiplyByPoint(
      this.toWorld,
      new Cartesian3(x, y, z),
      new Cartesian3(),
    );
    this.viewer.camera.setView({
      destination,
      orientation: { heading: this.heading, pitch: this.pitch, roll: 0 },
    });
  }

  destroy(): void {
    this.exit();
  }
}
