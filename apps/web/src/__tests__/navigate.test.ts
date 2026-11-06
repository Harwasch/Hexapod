import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BrickGrid } from "@/lib/collision";
import { ScanNavigation, consoleClearance, flightAxes, surfaceZoomStep } from "@/view/navigate";

const cleanup: (() => void)[] = [];
afterEach(() => {
  cleanup.splice(0).forEach((fn) => fn());
});

function setup(consoleMode = true) {
  const parent = document.createElement("div");
  const canvas = document.createElement("canvas");
  canvas.tabIndex = 0;
  canvas.setPointerCapture = vi.fn();
  canvas.releasePointerCapture = vi.fn();
  Object.defineProperties(canvas, { clientWidth: { value: 400 }, clientHeight: { value: 400 } });
  canvas.getBoundingClientRect = () => ({ left: 0, top: 0, width: 400, height: 400 }) as DOMRect;
  parent.append(canvas);
  document.body.append(parent);
  const camera = new THREE.PerspectiveCamera(50, 1, 0.001, 100);
  camera.position.set(1, 1, 4);
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(1, 1, 0);
  controls.update();
  const scan = new THREE.Group();
  scan.updateMatrixWorld();
  const nav = new ScanNavigation(
    scan,
    camera,
    controls,
    canvas,
    consoleMode
      ? {
          cursorNavigation: true,
          clearance: () => 0.01,
          flySpeed: () => 2,
        }
      : {},
  );
  nav.usePackaged(
    new BrickGrid(
      0.25,
      [0, 0, 0],
      new Map([[0, new Uint8Array(64).fill(255)]]),
      { lo: [0, 0, 0], hi: [0, 0, 0] },
      512,
    ),
  );
  cleanup.push(() => {
    nav.dispose();
    controls.dispose();
    parent.remove();
  });
  const pointer = (type: string, x: number, y = 200, options: PointerEventInit = {}) => {
    canvas.dispatchEvent(
      new PointerEvent(type, {
        bubbles: true,
        clientX: x,
        clientY: y,
        pointerId: 1,
        pointerType: "mouse",
        button: 0,
        ...options,
      }),
    );
  };
  const wheel = (deltaY: number) =>
    canvas.dispatchEvent(
      new WheelEvent("wheel", {
        bubbles: true,
        cancelable: true,
        clientX: 200,
        clientY: 200,
        deltaY,
      }),
    );
  return { canvas, camera, controls, scan, nav, pointer, wheel };
}

describe("console surface navigation", () => {
  it("picks the cursor hit without moving the eye or changing the view, then orbits that pivot", () => {
    const { camera, controls, nav, pointer } = setup();
    const hit = nav.hitAt({ clientX: 250, clientY: 200 })!;
    const eye = camera.position.clone(),
      orientation = camera.quaternion.clone();
    const screen = hit.clone().project(camera);
    pointer("pointerdown", 250);
    controls.update();
    expect(controls.target.distanceTo(hit)).toBeCloseTo(0, 10);
    camera.position.toArray().forEach((v, i) => expect(v).toBeCloseTo(eye.toArray()[i]!, 12));
    expect(Math.abs(camera.quaternion.dot(orientation))).toBeCloseTo(1, 12);
    pointer("pointermove", 280);
    controls.update();
    expect(camera.position.distanceTo(eye)).toBeGreaterThan(0.01);
    expect(hit.clone().project(camera).x).toBeCloseTo(screen.x, 10);
    expect(hit.clone().project(camera).y).toBeCloseTo(screen.y, 10);
    pointer("pointerup", 280);
  });
  it("keeps the target on a miss and double-click recentres a hit", () => {
    const { canvas, camera, controls, nav, pointer } = setup();
    const before = controls.target.clone();
    pointer("pointerdown", 2000);
    expect(controls.target.toArray()).toEqual(before.toArray());
    pointer("pointerup", 2000);
    const hit = nav.hitAt({ clientX: 250, clientY: 200 })!;
    const eye = camera.position.clone();
    canvas.dispatchEvent(new MouseEvent("dblclick", { clientX: 250, clientY: 200 }));
    expect(controls.target.distanceTo(hit)).toBeCloseTo(0, 10);
    expect(camera.position.distanceTo(eye)).toBeCloseTo(0, 12);
    expect(
      camera.getWorldDirection(new THREE.Vector3()).dot(hit.clone().sub(eye).normalize()),
    ).toBeCloseTo(1, 12);
  });
  it("recentres on the delivered mouse pixel without drift, including with damping enabled", () => {
    const { canvas, camera, controls, nav } = setup();
    controls.enableDamping = true;
    // The browser trace delivered an integer dblclick for a fractional Playwright point.
    // Probe that delivered ray; keep the same precision assertion, not a wider tolerance.
    const fraction = { clientX: 250.28, clientY: 200.289375 };
    const delivered = {
      clientX: Math.floor(fraction.clientX),
      clientY: Math.floor(fraction.clientY),
    };
    const hit = nav.hitAt(delivered)!;
    expect(nav.hitAt(fraction)!.distanceTo(hit)).toBeGreaterThan(1e-5);
    const eye = camera.position.clone();
    canvas.dispatchEvent(new MouseEvent("dblclick", delivered));
    for (let i = 0; i < 100; i++) controls.update();
    expect(controls.target.distanceTo(hit)).toBeLessThan(1e-5);
    expect(camera.position.distanceTo(eye)).toBeLessThan(1e-5);
  });
  it("wheels down to one centimetre without entering the solid; Space passes through", () => {
    const { canvas, camera, nav, wheel } = setup();
    canvas.focus();
    nav.frame(0);
    for (let i = 0; i < 30; i++) {
      wheel(-100);
      nav.frame(i * 16);
    }
    expect(camera.position.z).toBeGreaterThanOrEqual(2.01 - 1e-12);
    expect(camera.position.z).toBeLessThan(2.011);
    canvas.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, code: "Space" }));
    wheel(-100);
    expect(camera.position.z).toBeLessThan(2);
  });
  it.each([{ button: 2 }, { button: 0, shiftKey: true }])(
    "pans for %j without changing pivot to a surface",
    (buttons) => {
      const { controls, camera, pointer } = setup();
      const eye = camera.position.clone(),
        target = controls.target.clone();
      pointer("pointerdown", 200, 200, buttons);
      pointer("pointermove", 230, 200, buttons);
      const moved = camera.position.clone().sub(eye);
      expect(moved.length()).toBeGreaterThan(0);
      expect(controls.target.clone().sub(target).distanceTo(moved)).toBeCloseTo(0, 12);
      pointer("pointerup", 230, 200, buttons);
    },
  );
  it("flies only with canvas focus, stops on blur and ignores editable controls", () => {
    const { canvas, camera, nav } = setup();
    nav.frame(0);
    window.dispatchEvent(new KeyboardEvent("keydown", { code: "KeyW" }));
    nav.frame(50);
    expect(camera.position.z).toBeCloseTo(4, 10);
    canvas.focus();
    canvas.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, code: "KeyW" }));
    nav.frame(100);
    expect(camera.position.z).toBeCloseTo(3.9, 12);
    canvas.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, code: "ShiftLeft" }));
    nav.frame(150);
    expect(camera.position.z).toBeCloseTo(3.5, 12);
    const input = document.createElement("input");
    document.body.append(input);
    input.focus();
    input.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, code: "KeyW" }));
    nav.frame(200);
    expect(camera.position.z).toBeCloseTo(3.5, 12);
    input.remove();
  });
  it("pinches towards the midpoint and restores OrbitControls zoom after the gesture", () => {
    const { camera, controls, pointer } = setup();
    pointer("pointerdown", 180, 200, { pointerType: "touch", pointerId: 1 });
    pointer("pointerdown", 220, 200, { pointerType: "touch", pointerId: 2 });
    expect(controls.enableZoom).toBe(false);
    pointer("pointermove", 150, 200, { pointerType: "touch", pointerId: 1 });
    pointer("pointermove", 250, 200, { pointerType: "touch", pointerId: 2 });
    expect(camera.position.z).toBeLessThan(4);
    expect(camera.position.z).toBeGreaterThanOrEqual(2.01);
    pointer("pointerup", 150, 200, { pointerType: "touch", pointerId: 1 });
    pointer("pointerup", 250, 200, { pointerType: "touch", pointerId: 2 });
    expect(controls.enableZoom).toBe(true);
  });
  it("leaves the scan viewer on centre-line pivots and its original clearance", () => {
    const { nav, controls, pointer } = setup(false);
    pointer("pointerdown", 250);
    expect(controls.target.distanceTo(new THREE.Vector3(1, 1, 2))).toBeCloseTo(0, 10);
    expect(nav.clearance).toBe(0.375);
    pointer("pointerup", 250);
  });
  it("clears a scrubbed source's collision without changing the pivot or view", () => {
    const { nav, controls, camera, pointer } = setup();
    pointer("pointerdown", 250);
    pointer("pointerup", 250);
    const target = controls.target.clone(),
      orientation = camera.quaternion.clone();
    nav.clear();
    controls.update();
    expect(nav.hitAt({ clientX: 250, clientY: 200 })).toBeNull();
    expect(controls.target.distanceTo(target)).toBeCloseTo(0, 10);
    expect(Math.abs(camera.quaternion.dot(orientation))).toBeCloseTo(1, 10);
    // Returning from a saved camera preset must retain that same off-axis view too.
    nav.jumped(true);
    controls.update();
    expect(Math.abs(camera.quaternion.dot(orientation))).toBeCloseTo(1, 10);
  });
  it("replaces navigation without losing an off-axis view and removes its handlers on disposal", () => {
    const { nav, camera, controls, scan, canvas, pointer } = setup();
    pointer("pointerdown", 250);
    pointer("pointerup", 250);
    const orientation = camera.quaternion.clone();
    nav.dispose();
    // Compare identity after disposal; never call this detached reference.
    // eslint-disable-next-line @typescript-eslint/unbound-method
    const originalUpdate = controls.update;
    const replacement = new ScanNavigation(scan, camera, controls, canvas, {
      cursorNavigation: true,
      clearance: () => 0.01,
    });
    controls.update();
    expect(Math.abs(camera.quaternion.dot(orientation))).toBeCloseTo(1, 10);
    camera.near = 1;
    replacement.frame(0);
    expect(camera.near).toBe(0.001);
    replacement.dispose();
    expect(controls).toHaveProperty("update", originalUpdate);
    const target = controls.target.clone();
    canvas.dispatchEvent(new MouseEvent("dblclick", { clientX: 250, clientY: 200 }));
    expect(controls.target.toArray()).toEqual(target.toArray());
  });
  it("picks packaged solids through the solve's translated and rotated frame", () => {
    const { nav, scan, camera, controls } = setup();
    scan.position.set(20, -10, 5);
    // Avoid OrbitControls’ polar singularity; test the frame transform independently.
    scan.rotation.set(0.3, 0.2, 0.4);
    scan.updateMatrixWorld(true);
    camera.position.set(1, 1, 4).applyMatrix4(scan.matrixWorld);
    camera.up.set(0, 1, 0).transformDirection(scan.matrixWorld);
    controls.target.set(1, 1, 0).applyMatrix4(scan.matrixWorld);
    nav.jumped();
    controls.update();
    const expected = new THREE.Vector3(1, 1, 2).applyMatrix4(scan.matrixWorld);
    expect(nav.hitAt({ clientX: 200, clientY: 200 })!.distanceTo(expected)).toBeCloseTo(0, 10);
  });
});

describe("navigation distances and keys", () => {
  it("chooses scene clearance and an asymptotic step down to one centimetre", () => {
    expect(consoleClearance(2, 100)).toBe(0.005);
    expect(consoleClearance(null, 10)).toBe(0.01);
    let distance = 0.02;
    for (let i = 0; i < 20; i++) {
      const step = surfaceZoomStep(distance, -1, 0.01);
      expect(step).toBeGreaterThanOrEqual(0);
      distance -= step;
      expect(distance).toBeGreaterThanOrEqual(0.01);
    }
    expect(distance).toBeCloseTo(0.01, 12);
    expect(surfaceZoomStep(0.01, -1, 0.01)).toBe(0);
    expect(surfaceZoomStep(0.01, 1, 0.01)).toBeLessThan(0);
  });
  it("maps WASD and Q/E to camera axes, with opposite keys cancelling", () => {
    for (const [key, axis] of Object.entries({
      KeyW: [0, 0, 1],
      KeyS: [0, 0, -1],
      KeyA: [-1, 0, 0],
      KeyD: [1, 0, 0],
      KeyQ: [0, -1, 0],
      KeyE: [0, 1, 0],
    })) {
      expect(flightAxes(new Set([key, "ShiftLeft"]))).toEqual(axis);
    }
    expect(flightAxes(new Set(["KeyW", "KeyS", "KeyA", "KeyD", "KeyQ", "KeyE"]))).toEqual([
      0, 0, 0,
    ]);
  });
});
