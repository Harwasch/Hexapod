import { Cartographic, Math as CesiumMath, type Scene, type Viewer } from "cesium";

import type { Emitter } from "@/lib/emitter";

import type { SceneEvents } from "./types";

const KEYS = new Set([
  "w",
  "a",
  "s",
  "d",
  "q",
  "e",
  "shift",
  "arrowup",
  "arrowdown",
  "arrowleft",
  "arrowright",
]);
const MIN_CLEARANCE_M = 0.3;

/**
 * Close-range free-flight ("Explore mode"): WASD/QE motion, drag-to-look,
 * scroll-to-change-speed. Same scene, same camera; the default controller is
 * paused, not replaced.
 */
export class ExploreController {
  private readonly scene: Scene;
  private active = false;
  private readonly pressed = new Set<string>();
  private speed = 4;
  private lastTick = 0;
  private looking = false;
  private lastPointer: { x: number; y: number } | null = null;
  private removeTick: (() => void) | null = null;

  constructor(
    private readonly viewer: Viewer,
    private readonly events: Emitter<SceneEvents>,
  ) {
    this.scene = viewer.scene;
  }

  get isActive(): boolean {
    return this.active;
  }

  get currentSpeed(): number {
    return this.speed;
  }

  setSpeed(metersPerSecond: number): void {
    this.speed = CesiumMath.clamp(metersPerSecond, 0.1, 200);
  }

  enter(speed?: number): void {
    if (this.active) return;
    if (speed) this.setSpeed(speed);
    this.active = true;
    this.scene.screenSpaceCameraController.enableInputs = false;
    this.viewer.canvas.style.cursor = "move";
    this.lastTick = performance.now();
    // preUpdate fires every widget tick even in request-render mode; preRender would not.
    this.removeTick = this.scene.preUpdate.addEventListener(() => this.tick());
    window.addEventListener("keydown", this.onKeyDown, { capture: true });
    window.addEventListener("keyup", this.onKeyUp, { capture: true });
    window.addEventListener("blur", this.onBlur);
    const canvas = this.viewer.canvas;
    canvas.addEventListener("pointerdown", this.onPointerDown);
    window.addEventListener("pointermove", this.onPointerMove);
    window.addEventListener("pointerup", this.onPointerUp);
    canvas.addEventListener("wheel", this.onWheel, { passive: false });
    this.events.emit("explore", true);
  }

  exit(): void {
    if (!this.active) return;
    this.active = false;
    this.pressed.clear();
    this.scene.screenSpaceCameraController.enableInputs = true;
    this.viewer.canvas.style.cursor = "";
    this.removeTick?.();
    this.removeTick = null;
    window.removeEventListener("keydown", this.onKeyDown, { capture: true });
    window.removeEventListener("keyup", this.onKeyUp, { capture: true });
    window.removeEventListener("blur", this.onBlur);
    const canvas = this.viewer.canvas;
    canvas.removeEventListener("pointerdown", this.onPointerDown);
    window.removeEventListener("pointermove", this.onPointerMove);
    window.removeEventListener("pointerup", this.onPointerUp);
    canvas.removeEventListener("wheel", this.onWheel);
    this.events.emit("explore", false);
  }

  private readonly onKeyDown = (event: KeyboardEvent) => {
    if (isTypingTarget(event.target)) return;
    const key = event.key.toLowerCase();
    if (key === "escape") {
      event.preventDefault();
      this.exit();
      return;
    }
    if (KEYS.has(key)) {
      event.preventDefault();
      this.pressed.add(key);
      this.scene.requestRender();
    }
  };

  private readonly onKeyUp = (event: KeyboardEvent) => {
    this.pressed.delete(event.key.toLowerCase());
  };

  private readonly onBlur = () => this.pressed.clear();

  private readonly onPointerDown = (event: PointerEvent) => {
    if (event.button !== 0) return;
    this.looking = true;
    this.lastPointer = { x: event.clientX, y: event.clientY };
  };

  private readonly onPointerMove = (event: PointerEvent) => {
    if (!this.looking || !this.lastPointer) return;
    const dx = event.clientX - this.lastPointer.x;
    const dy = event.clientY - this.lastPointer.y;
    this.lastPointer = { x: event.clientX, y: event.clientY };
    const camera = this.viewer.camera;
    const sensitivity = 0.0032;
    camera.setView({
      orientation: {
        heading: camera.heading + dx * sensitivity,
        pitch: CesiumMath.clamp(
          camera.pitch - dy * sensitivity,
          -CesiumMath.PI_OVER_TWO + 0.02,
          CesiumMath.PI_OVER_TWO - 0.02,
        ),
        roll: 0,
      },
    });
  };

  private readonly onPointerUp = () => {
    this.looking = false;
    this.lastPointer = null;
  };

  /** Scroll moves along the view direction like a zoom; Shift+scroll changes the speed. */
  private readonly onWheel = (event: WheelEvent) => {
    event.preventDefault();
    if (event.shiftKey) {
      const factor = event.deltaY > 0 ? 0.85 : 1.18;
      this.setSpeed(this.speed * factor);
      this.events.emit("explore", true);
      return;
    }
    const step = this.speed * 0.5 * Math.sign(-event.deltaY);
    if (step > 0) this.viewer.camera.moveForward(step);
    else this.viewer.camera.moveBackward(-step);
    this.keepAboveGround();
    this.scene.requestRender();
  };

  private tick(): void {
    const now = performance.now();
    const dt = Math.min(0.1, (now - this.lastTick) / 1000);
    this.lastTick = now;
    if (this.pressed.size === 0) return;
    const camera = this.viewer.camera;
    const boost = this.pressed.has("shift") ? 3 : 1;
    const step = this.speed * boost * dt;
    if (this.pressed.has("w") || this.pressed.has("arrowup")) camera.moveForward(step);
    if (this.pressed.has("s") || this.pressed.has("arrowdown")) camera.moveBackward(step);
    if (this.pressed.has("a") || this.pressed.has("arrowleft")) camera.moveLeft(step);
    if (this.pressed.has("d") || this.pressed.has("arrowright")) camera.moveRight(step);
    if (this.pressed.has("e")) camera.moveUp(step);
    if (this.pressed.has("q")) camera.moveDown(step);
    this.keepAboveGround();
    this.scene.requestRender();
  }

  private keepAboveGround(): void {
    const camera = this.viewer.camera;
    const carto = camera.positionCartographic;
    const ground = this.scene.globe.show ? this.scene.globe.getHeight(carto) : undefined;
    if (ground !== undefined && carto.height < ground + MIN_CLEARANCE_M) {
      const lifted = Cartographic.clone(carto);
      lifted.height = ground + MIN_CLEARANCE_M;
      camera.setView({
        destination: Cartographic.toCartesian(lifted),
        orientation: { heading: camera.heading, pitch: camera.pitch, roll: 0 },
      });
    }
  }

  destroy(): void {
    this.exit();
  }
}

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}
