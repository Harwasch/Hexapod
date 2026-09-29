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
 */

import * as THREE from "three";
import type { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import { PrecomputedSolids, type BrickGrid, type Solids } from "@/lib/collision";
import { SplatOccupancy } from "@/lib/occupancy";

import type { SpzPoints } from "./spz";

/** Each wheel notch (100 px of delta) closes this share of the distance left to a surface. */
const WHEEL_STEP = 0.75;

export class ScanNavigation {
  /** Built from the tiles on screen, until (or unless) the packaged grid arrives. */
  private readonly runtime = new SplatOccupancy();
  private solids: Solids = this.runtime;
  private passThrough = false;
  private readonly lastGood = new THREE.Vector3();
  private hasGood = false;
  private readonly toLocal = new THREE.Matrix4();
  private readonly raycaster = new THREE.Raycaster();
  private readonly off: (() => void)[] = [];

  constructor(
    private readonly scan: THREE.Object3D,
    private readonly camera: THREE.PerspectiveCamera,
    private readonly controls: OrbitControls,
    private readonly element: HTMLElement,
  ) {
    const host = element.parentElement ?? element;
    const onKey = (event: KeyboardEvent): void => {
      if (event.code !== "Space") return;
      if (event.type === "keydown") {
        event.preventDefault();
        this.passThrough = true;
      } else this.passThrough = false;
    };
    const onBlur = (): void => {
      this.passThrough = false;
    };
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
    return this.solids.clearance;
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
    this.toLocal.copy(this.scan.matrixWorld).invert();
    const o = origin.clone().applyMatrix4(this.toLocal);
    const d = direction.clone().transformDirection(this.toLocal);
    return this.solids.raycast([o.x, o.y, o.z], [d.x, d.y, d.z], far);
  }

  /** Once a frame, after the controls moved the camera: undo any move into a surface. */
  frame(): void {
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
    const swept = this.solids.sweep([from.x, from.y, from.z], [to.x, to.y, to.z]);
    if (swept.blocked) {
      position.fromArray(swept.position).applyMatrix4(this.scan.matrixWorld);
    }
    this.lastGood.copy(position);
  }

  /** Forgets where the camera was: the next position is a jump, not a move to check. */
  jumped(): void {
    this.hasGood = false;
  }

  dispose(): void {
    for (const off of this.off) off();
  }

  private ndc(event: { clientX: number; clientY: number }): THREE.Vector2 {
    const rect = this.element.getBoundingClientRect();
    return new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
  }

  private readonly onPointerDown = (): void => {
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
}
