/**
 * The scan viewer's camera against the scan itself: the splats as surfaces the camera turns
 * around, zooms to, and cannot pass through -- unless Space is held.
 *
 * OrbitControls knows nothing of what it looks at: it orbits a target point that starts at
 * the middle of the scan and dollies towards it, so in a site rather than an object the
 * camera passes through whatever is between, and a close look at anything off-centre means
 * orbiting the empty middle. A splat writes no depth, so there is nothing to pick either.
 * The scan's collision -- packaged with it (lib/collision.ts), or for older scans built from
 * the tiles on screen (lib/occupancy.ts, spz.ts) -- gives:
 *
 * - **Orbit around what is in the middle of the view.** When a drag or pinch starts, the
 *   target moves along the view line to the first surface there (Google Maps' rule: the
 *   view turns about what you are looking at). Moving it along the view line changes nothing
 *   on screen, so there is no jump; and a pinch then stops `clearance` short of that surface.
 * - **Wheel towards the cursor, stopping at the surface.** Along the ray through the cursor
 *   (so that point stays under it), each notch closing a share of the distance left, never
 *   nearer than `clearance`. With nothing under the cursor, OrbitControls' zoom-to-cursor.
 * - **No moving into a surface.** Every frame, after the controls move the camera, the move
 *   is swept against the grid; a camera that would enter a surface stops at it and slides
 *   along it (OccupancyGrid.sweep).
 * - **Space to pass through,** held, for what is on the other side of a wall or a hedge.
 *
 * The console opts into cursorNavigation: an off-axis orbit pivot that preserves the view,
 * double-click recenter, focus-scoped flight and cursor/pinch zoom with a solve-scale clearance.
 * The published scan viewer retains the default interaction policy above.
 */

import * as THREE from "three";
import type { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import { PrecomputedSolids, type BrickGrid, type Solids } from "@/lib/collision";
import { SplatOccupancy } from "@/lib/occupancy";

import type { SpzPoints } from "./spz";

/** Each wheel notch (100 px of delta) closes this share of the distance left to a surface. */
const WHEEL_STEP = 0.75;

/** Console-only interaction policy. Omit to retain the scan viewer's original controls. */
export interface NavigationOptions {
  cursorNavigation?: boolean;
  clearance?: () => number;
  flySpeed?: () => number;
  onInteraction?: () => void;
}

/** One centimetre in the solve frame, or 0.1% of its radius when scale is unknown. */
export function consoleClearance(metresPerUnit: number | null | undefined, radius: number): number {
  return metresPerUnit != null && Number.isFinite(metresPerUnit) && metresPerUnit > 0
    ? 0.01 / metresPerUnit
    : Math.max(radius * 0.001, 1e-6);
}

/** Camera-local right, up, forward. Shift changes speed, not direction. */
export function flightAxes(keys: ReadonlySet<string>): [number, number, number] {
  return [
    Number(keys.has("KeyD")) - Number(keys.has("KeyA")),
    Number(keys.has("KeyE")) - Number(keys.has("KeyQ")),
    Number(keys.has("KeyW")) - Number(keys.has("KeyS")),
  ];
}

/** Positive is towards the hit. Zoom remains proportional right down to clearance. */
export function surfaceZoomStep(
  hit: number,
  notches: number,
  clearance: number,
  through = false,
): number {
  let step = hit * (1 - Math.pow(WHEEL_STEP, -Math.max(-3, Math.min(3, notches))));
  if (through && notches < 0 && step < clearance * 4) return hit + clearance;
  if (step > 0 && !through) step = Math.min(step, Math.max(0, hit - clearance));
  return step;
}

export class ScanNavigation {
  /** Built from the tiles on screen, until (or unless) the packaged grid arrives. */
  private readonly runtime = new SplatOccupancy();
  private solids: Solids = this.runtime;
  private passThrough = false;
  private readonly lastGood = new THREE.Vector3();
  private hasGood = false;
  private readonly toLocal = new THREE.Matrix4();
  private readonly raycaster = new THREE.Raycaster();
  private readonly orientationOffset = new THREE.Quaternion();
  private readonly keys = new Set<string>();
  private readonly touches = new Map<number, { clientX: number; clientY: number }>();
  private pinchSpan: number | null = null;
  private zoomBeforePinch: boolean | null = null;
  private lastFrame: number | null = null;
  private readonly off: (() => void)[] = [];

  constructor(
    private readonly scan: THREE.Object3D,
    private readonly camera: THREE.PerspectiveCamera,
    private readonly controls: OrbitControls,
    private readonly element: HTMLElement,
    private readonly options: NavigationOptions = {},
  ) {
    const host = element.parentElement ?? element;
    const onKey = (event: KeyboardEvent): void => {
      if (options.cursorNavigation) {
        if (event.type === "keyup") {
          this.keys.delete(event.code);
          if (event.code === "Space") this.passThrough = false;
          return;
        }
        if (
          document.activeElement !== element ||
          event.ctrlKey ||
          event.metaKey ||
          event.altKey ||
          (event.target instanceof HTMLElement &&
            event.target.closest("input,textarea,select,[contenteditable]"))
        )
          return;
        if (
          ![
            "Space",
            "KeyW",
            "KeyA",
            "KeyS",
            "KeyD",
            "KeyQ",
            "KeyE",
            "ShiftLeft",
            "ShiftRight",
          ].includes(event.code)
        )
          return;
        event.preventDefault();
        this.keys.add(event.code);
        this.interacting();
        if (event.code === "Space") this.passThrough = true;
        return;
      }
      if (event.code !== "Space") return;
      if (event.type === "keydown") {
        event.preventDefault();
        this.passThrough = true;
      } else this.passThrough = false;
    };
    const onBlur = (): void => {
      this.passThrough = false;
      this.keys.clear();
      this.touches.clear();
      this.endPinch();
    };
    if (options.cursorNavigation) {
      // OrbitControls always looks at its target during update, including its own event
      // handlers. Preserve the clicked point's original off-axis view through that update;
      // rotating then carries both eye and view around the pivot. Double-click clears it.
      // A source replacement may inherit an off-axis pivot from the previous navigation.
      const look = new THREE.Matrix4().lookAt(camera.position, controls.target, camera.up);
      this.orientationOffset.setFromRotationMatrix(look).invert().multiply(camera.quaternion);
      // Preserve method identity for disposal; invoke it with controls as its receiver below.
      // eslint-disable-next-line @typescript-eslint/unbound-method
      const update = controls.update;
      controls.update = (delta?: number): boolean => {
        const changed = update.call(controls, delta);
        camera.quaternion.multiply(this.orientationOffset).normalize();
        camera.updateMatrixWorld(true);
        return changed;
      };
      const rightButton = controls.mouseButtons.RIGHT;
      element.addEventListener("dblclick", this.onDoubleClick);
      element.addEventListener("blur", onBlur);
      element.ownerDocument.addEventListener("pointermove", this.onPointerMove, true);
      element.ownerDocument.addEventListener("pointerup", this.onPointerUp, true);
      element.ownerDocument.addEventListener("pointercancel", this.onPointerUp, true);
      this.off.push(() => {
        onBlur();
        controls.update = update;
        controls.mouseButtons.RIGHT = rightButton;
        element.removeEventListener("dblclick", this.onDoubleClick);
        element.removeEventListener("blur", onBlur);
        element.ownerDocument.removeEventListener("pointermove", this.onPointerMove, true);
        element.ownerDocument.removeEventListener("pointerup", this.onPointerUp, true);
        element.ownerDocument.removeEventListener("pointercancel", this.onPointerUp, true);
      });
    }
    // Capture on the canvas's parent: seen before OrbitControls' own listeners on the canvas.
    host.addEventListener("wheel", this.onWheel, { capture: true, passive: false });
    element.addEventListener("pointerdown", this.onPointerDown, { capture: true });
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKey);
    window.addEventListener("blur", onBlur);
    this.off.push(() => {
      host.removeEventListener("wheel", this.onWheel, { capture: true });
      element.removeEventListener("pointerdown", this.onPointerDown, { capture: true });
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keyup", onKey);
      window.removeEventListener("blur", onBlur);
    });
  }

  /**
   * The scan's packaged collision (hexapod.collision): exact at full resolution, and no
   * building here. From then on the tiles' own splats are not needed for collision.
   */
  usePackaged(grid: BrickGrid): void {
    this.solids = new PrecomputedSolids(grid);
    this.runtime.clear();
  }

  /** How close the camera may come to the finest surfaces on screen (metres). */
  get clearance(): number {
    return this.options.clearance?.() ?? this.solids.clearance;
  }

  /** A tile went on screen: its splats become surfaces. */
  show(key: unknown, points: SpzPoints | undefined): void {
    if (!points) return;
    if (this.solids !== this.runtime) return;
    this.runtime.add(key, points.positions, 0, points.count, (i) => points.alphas[i] ?? 0);
  }

  /** A tile came off screen. */
  hide(key: unknown): void {
    this.runtime.remove(key);
  }

  /** The first surface along a world-space ray, as a distance, or null. */
  raycast(origin: THREE.Vector3, direction: THREE.Vector3, far = 1e4): number | null {
    this.scan.updateMatrixWorld(true);
    this.toLocal.copy(this.scan.matrixWorld).invert();
    const o = origin.clone().applyMatrix4(this.toLocal);
    const d = direction.clone().transformDirection(this.toLocal);
    return this.solids.raycast([o.x, o.y, o.z], [d.x, d.y, d.z], far);
  }

  /** Once a frame, after the controls moved the camera: undo any move into a surface. */
  frame(now = performance.now()): void {
    if (this.options.cursorNavigation) {
      // Camera presets and new bounds can change the near plane after navigation starts.
      const near = Math.min(this.camera.near, this.clearance * 0.1);
      if (near !== this.camera.near) {
        this.camera.near = near;
        this.camera.updateProjectionMatrix();
      }
      const dt =
        this.lastFrame === null ? 0 : Math.min(0.05, Math.max(0, (now - this.lastFrame) / 1000));
      this.lastFrame = now;
      const [right, up, forward] = flightAxes(this.keys);
      if (right || up || forward) {
        const fast = this.keys.has("ShiftLeft") || this.keys.has("ShiftRight") ? 4 : 1;
        const move = new THREE.Vector3(right, up, -forward)
          .normalize()
          .applyQuaternion(this.camera.quaternion)
          .multiplyScalar((this.options.flySpeed?.() ?? 1) * fast * dt);
        this.move(move);
      }
    }
    // A pinch stops at the finest surfaces' clearance, which follows what is drawn.
    this.controls.minDistance = this.clearance;
    const position = this.camera.position;
    if (!this.hasGood || this.passThrough || this.solids.empty) {
      this.lastGood.copy(position);
      this.hasGood = true;
      return;
    }
    if (position.distanceToSquared(this.lastGood) < 1e-14) return;
    this.toLocal.copy(this.scan.matrixWorld).invert();
    const from = this.lastGood.clone().applyMatrix4(this.toLocal);
    const to = position.clone().applyMatrix4(this.toLocal);
    const swept = this.solids.sweep(
      [from.x, from.y, from.z],
      [to.x, to.y, to.z],
      this.options.clearance?.(),
    );
    if (swept.blocked) {
      position.fromArray(swept.position).applyMatrix4(this.scan.matrixWorld);
    }
    this.lastGood.copy(position);
  }

  /** A preset jump, optionally retaining its off-axis view (returning from Top down). */
  jumped(preserveView = false): void {
    this.hasGood = false;
    this.orientationOffset.identity();
    if (preserveView) {
      const look = new THREE.Matrix4().lookAt(
        this.camera.position,
        this.controls.target,
        this.camera.up,
      );
      this.orientationOffset.setFromRotationMatrix(look).invert().multiply(this.camera.quaternion);
    }
  }

  /** Drop a replaced source's solids while retaining the user's pivot and view. */
  clear(): void {
    this.runtime.clear();
    this.solids = this.runtime;
    this.hasGood = false;
  }

  dispose(): void {
    for (const off of this.off.splice(0)) off();
    this.clear();
  }

  private ndc(event: { clientX: number; clientY: number }): THREE.Vector2 {
    const rect = this.element.getBoundingClientRect();
    return new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
  }

  private readonly onPointerDown = (event: PointerEvent): void => {
    if (this.options.cursorNavigation) {
      this.element.focus({ preventScroll: true });
      this.interacting();
      if (event.pointerType === "touch") {
        this.touches.set(event.pointerId, { clientX: event.clientX, clientY: event.clientY });
        if (this.touches.size === 2) {
          this.pinchSpan = this.pinch()?.span ?? null;
          this.zoomBeforePinch = this.controls.enableZoom;
          this.controls.enableZoom = false; // Our cursor ray zoom replaces native dolly; native pan stays.
        }
        if (this.touches.size !== 1) return;
      } else {
        this.controls.mouseButtons.RIGHT = event.shiftKey ? THREE.MOUSE.ROTATE : THREE.MOUSE.PAN;
        if (event.button !== 0 || event.shiftKey || event.ctrlKey || event.metaKey) return;
      }
      this.pickPivot(event, false);
      return;
    }
    // The target to the first surface along the view line, when there is one ahead of it
    // or just behind it: the orbit turns about what is in the middle of the view.
    const direction = new THREE.Vector3();
    this.camera.getWorldDirection(direction);
    const hit = this.raycast(this.camera.position, direction);
    if (hit === null) return;
    this.controls.target.copy(this.camera.position).addScaledVector(direction, hit);
  };

  private readonly onWheel = (event: WheelEvent): void => {
    if (event.target !== this.element) return;
    if (this.options.cursorNavigation) {
      const pixels =
        event.deltaMode === 1
          ? event.deltaY * 33
          : event.deltaMode === 2
            ? event.deltaY * this.element.clientHeight
            : event.deltaY;
      if (this.zoomAt(event, pixels / 100)) {
        event.preventDefault();
        event.stopPropagation();
      }
      return;
    }
    this.raycaster.setFromCamera(this.ndc(event), this.camera);
    const { origin, direction } = this.raycaster.ray;
    const hit = this.raycast(origin, direction);
    if (hit === null) return;
    event.preventDefault();
    event.stopPropagation();
    this.controls.autoRotate = false;
    const pixels = event.deltaMode === 1 ? event.deltaY * 33 : event.deltaY;
    const notches = Math.min(3, Math.max(-3, pixels / 100));
    const next = hit * Math.pow(WHEEL_STEP, -notches);
    let step = hit - next;
    if (!this.passThrough) step = Math.min(step, hit - this.clearance);
    // Passing through: close to the surface already, a notch carries the camera past it.
    else if (notches < 0 && hit - next < this.clearance * 4) step = hit + this.clearance;
    if (Math.abs(step) < 1e-9) return;
    const move = direction.clone().multiplyScalar(step);
    this.camera.position.add(move);
    this.controls.target.add(move);
    // A deliberate move toward a surface is not a collision to undo.
    this.lastGood.copy(this.camera.position);
    this.controls.update();
  };

  private interacting(): void {
    this.controls.autoRotate = false;
    this.options.onInteraction?.();
  }

  /** The first solid under a CSS-pixel cursor, in the display's world frame. */
  hitAt(cursor: { clientX: number; clientY: number }): THREE.Vector3 | null {
    this.camera.updateMatrixWorld(true);
    this.raycaster.setFromCamera(this.ndc(cursor), this.camera);
    const { origin, direction } = this.raycaster.ray;
    const hit = this.raycast(origin, direction, this.camera.far);
    return hit === null ? null : origin.clone().addScaledVector(direction, hit);
  }

  private pickPivot(cursor: { clientX: number; clientY: number }, recenter: boolean): void {
    const hit = this.hitAt(cursor);
    if (!hit || hit.distanceTo(this.camera.position) < this.clearance) return;
    const position = this.camera.position.clone();
    const orientation = this.camera.quaternion.clone();
    // Finish any residual damping without carrying a previous drag into this pivot.
    const damping = this.controls.enableDamping;
    this.controls.enableDamping = false;
    this.controls.update();
    this.controls.enableDamping = damping;
    this.camera.position.copy(position);
    this.controls.target.copy(hit);
    this.orientationOffset.identity();
    if (!recenter) {
      const look = new THREE.Matrix4().lookAt(position, hit, this.camera.up);
      this.orientationOffset.setFromRotationMatrix(look).invert().multiply(orientation);
    }
    this.controls.update();
    this.lastGood.copy(position);
    this.hasGood = true;
  }

  private readonly onDoubleClick = (event: MouseEvent): void => {
    this.interacting();
    this.pickPivot(event, true);
  };

  private zoomAt(cursor: { clientX: number; clientY: number }, notches: number): boolean {
    const hit =
      this.hitAt(cursor) ??
      this.raycaster.ray.at(
        Math.max(this.clearance, this.camera.position.distanceTo(this.controls.target)),
        new THREE.Vector3(),
      );
    const ray = hit.sub(this.camera.position);
    const distance = ray.length();
    if (distance === 0) return true;
    this.interacting();
    this.move(
      ray
        .normalize()
        .multiplyScalar(surfaceZoomStep(distance, notches, this.clearance, this.passThrough)),
    );
    this.controls.update();
    return true;
  }

  /** Translate eye and pivot together, checking the entire path (also for flight). */
  private move(delta: THREE.Vector3): void {
    const before = this.camera.position.clone();
    const after = before.clone().add(delta);
    if (!this.passThrough && !this.solids.empty) {
      this.scan.updateMatrixWorld(true);
      this.toLocal.copy(this.scan.matrixWorld).invert();
      const from = before.clone().applyMatrix4(this.toLocal).toArray();
      const to = after.clone().applyMatrix4(this.toLocal).toArray();
      after
        .fromArray(this.solids.sweep(from, to, this.clearance).position)
        .applyMatrix4(this.scan.matrixWorld);
    }
    this.camera.position.copy(after);
    this.controls.target.add(after.sub(before));
    this.lastGood.copy(this.camera.position);
    this.hasGood = true;
  }

  private pinch(): { span: number; clientX: number; clientY: number } | null {
    const [a, b] = [...this.touches.values()];
    return !a || !b
      ? null
      : {
          span: Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY),
          clientX: (a.clientX + b.clientX) / 2,
          clientY: (a.clientY + b.clientY) / 2,
        };
  }

  private readonly onPointerMove = (event: PointerEvent): void => {
    if (!this.touches.has(event.pointerId)) return;
    this.touches.set(event.pointerId, { clientX: event.clientX, clientY: event.clientY });
    const pinch = this.pinch();
    if (pinch && this.pinchSpan && pinch.span > 0) {
      this.zoomAt(pinch, Math.log(pinch.span / this.pinchSpan) / Math.log(WHEEL_STEP));
      this.pinchSpan = pinch.span;
    }
  };

  private endPinch(): void {
    this.pinchSpan = null;
    if (this.zoomBeforePinch !== null) this.controls.enableZoom = this.zoomBeforePinch;
    this.zoomBeforePinch = null;
  }

  private readonly onPointerUp = (event: PointerEvent): void => {
    this.touches.delete(event.pointerId);
    if (this.touches.size < 2) this.endPinch();
  };
}
