import type { Cartesian3, Scene, Viewer } from "cesium";

import { isTyping } from "@/lib/hotkeys";

import type { CameraController } from "./CameraController";

/** Pixels per second the arrow keys pan by (screen space, so it feels the same at any height). */
const PAN_PX_PER_S = 700;
/** Shift+arrows: degrees per second of heading and tilt around the view centre. */
const ORBIT_DEG_PER_S = 70;
const TILT_DEG_PER_S = 45;
/** Plus and minus halve or double the distance to the view centre in this many seconds. */
const ZOOM_HALVING_S = 0.9;

const PAN_KEYS = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"]);
const ZOOM_IN_KEYS = new Set(["+", "=", "Add"]);
const ZOOM_OUT_KEYS = new Set(["-", "_", "Subtract"]);

/**
 * Google Maps keyboard navigation: arrows pan, Shift+arrows orbit the view centre (left and
 * right turn, up and down tilt), plus and minus zoom towards it. Keys held down move
 * continuously, ticked on `scene.preUpdate` so it works in request-render mode. Only active
 * while the map itself has focus (nothing typed into an input, no panel list focused) and
 * while the default camera inputs are on (explore mode has its own keys).
 */
export class KeyboardNavigator {
  private readonly scene: Scene;
  private readonly pressed = new Set<string>();
  private shift = false;
  private lastTick: number | null = null;
  private pivot: Cartesian3 | null = null;
  private readonly removeTick: () => void;

  constructor(
    private readonly viewer: Viewer,
    private readonly camera: CameraController,
  ) {
    this.scene = viewer.scene;
    this.removeTick = this.scene.preUpdate.addEventListener(() => this.tick());
    window.addEventListener("keydown", this.onKeyDown);
    window.addEventListener("keyup", this.onKeyUp);
    window.addEventListener("blur", this.release);
  }

  private readonly onKeyDown = (event: KeyboardEvent): void => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    if (!this.mapHasFocus(event.target)) return;
    if (!this.scene.screenSpaceCameraController.enableInputs) return;
    const key = normalise(event.key);
    if (!key) return;
    event.preventDefault();
    if (event.repeat) return;
    this.shift = event.shiftKey;
    if (this.pressed.size === 0) {
      this.lastTick = null;
      this.pivot = this.camera.pivotAtCenter();
    }
    this.pressed.add(key);
  };

  private readonly onKeyUp = (event: KeyboardEvent): void => {
    const key = normalise(event.key);
    if (key) this.pressed.delete(key);
    this.shift = event.shiftKey;
    if (this.pressed.size === 0) this.pivot = null;
  };

  private readonly release = (): void => {
    this.pressed.clear();
    this.pivot = null;
  };

  /** Focus on the page body or the canvas counts as "the map"; a focused list keeps its arrows. */
  private mapHasFocus(target: EventTarget | null): boolean {
    if (isTyping(target)) return false;
    const active = document.activeElement;
    return active === null || active === document.body || active === this.viewer.canvas;
  }

  private tick(): void {
    if (this.pressed.size === 0) return;
    const now = performance.now();
    const dt = this.lastTick === null ? 1 / 60 : Math.min(0.1, (now - this.lastTick) / 1000);
    this.lastTick = now;
    this.pivot ??= this.camera.pivotAtCenter();

    let dx = 0;
    let dy = 0;
    if (this.pressed.has("ArrowLeft")) dx -= 1;
    if (this.pressed.has("ArrowRight")) dx += 1;
    if (this.pressed.has("ArrowUp")) dy -= 1;
    if (this.pressed.has("ArrowDown")) dy += 1;
    if (dx !== 0 || dy !== 0) {
      if (this.shift && this.pivot) {
        const toRadians = Math.PI / 180;
        // Same directions as a Ctrl+drag: right turns the foreground right, up tips towards
        // the horizon.
        this.camera.orbit(
          this.pivot,
          dx * ORBIT_DEG_PER_S * toRadians * dt,
          -dy * TILT_DEG_PER_S * toRadians * dt,
        );
      } else {
        this.camera.pan(-dx * PAN_PX_PER_S * dt, -dy * PAN_PX_PER_S * dt);
      }
    }
    const zoom = (this.pressed.has("+") ? -1 : 0) + (this.pressed.has("-") ? 1 : 0);
    if (zoom !== 0 && this.pivot) {
      this.camera.zoomToward(this.pivot, Math.pow(2, (zoom * dt) / ZOOM_HALVING_S));
    }
  }

  destroy(): void {
    this.removeTick();
    window.removeEventListener("keydown", this.onKeyDown);
    window.removeEventListener("keyup", this.onKeyUp);
    window.removeEventListener("blur", this.release);
  }
}

function normalise(key: string): string | null {
  if (PAN_KEYS.has(key)) return key;
  if (ZOOM_IN_KEYS.has(key)) return "+";
  if (ZOOM_OUT_KEYS.has(key)) return "-";
  return null;
}
