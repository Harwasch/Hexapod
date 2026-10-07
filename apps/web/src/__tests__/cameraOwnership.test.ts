/**
 * The camera rule (cameraOwnership.ts, CameraController): once the person has touched the
 * camera, nothing but they move it. Before, a close-up the wheel had just made was undone a
 * moment later -- lifted by CesiumJS's terrain collision when a finer terrain tile landed under
 * it, or glided up by the floor check onto whatever the pick pass found above it. A floor may
 * still stop the person's own motion; it never moves a camera they left at rest.
 */
import { Cartesian3, Cartographic, Ellipsoid, Event, Math as CesiumMath, Matrix4 } from "cesium";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CameraController } from "@/cesium/CameraController";
import { CameraOwnership, INPUT_SETTLE_MS } from "@/cesium/cameraOwnership";
import type { SceneEvents } from "@/cesium/types";
import { Emitter } from "@/lib/emitter";

describe("CameraOwnership", () => {
  it("lets the app correct a camera nobody has touched, and never one somebody has", () => {
    const rule = new CameraOwnership();
    expect(rule.userHasCamera).toBe(false);
    expect(rule.mayCorrect).toBe(true);
    rule.input(1_000);
    expect(rule.userHasCamera).toBe(true);
    expect(rule.mayCorrect).toBe(false);
    // However long it rests.
    rule.controllerMoved(false);
    expect(rule.mayCorrect).toBe(false);
    expect(rule.collisionAllowed(1_000_000)).toBe(false);
    // Until the app flies it again, at the person's asking.
    rule.appFlew();
    expect(rule.mayCorrect).toBe(true);
  });

  it("lets the terrain stop the person's own motion and its inertia, never move it at rest", () => {
    const rule = new CameraOwnership();
    // Untouched: collision as CesiumJS has it.
    expect(rule.collisionAllowed(0)).toBe(true);
    rule.input(10_000);
    expect(rule.collisionAllowed(10_000)).toBe(true);
    expect(rule.collisionAllowed(10_000 + INPUT_SETTLE_MS - 1)).toBe(true);
    // The wheel stopped and nothing coasts: the camera is the person's, at rest.
    expect(rule.collisionAllowed(10_000 + INPUT_SETTLE_MS + 1)).toBe(false);
    // Still coasting on the gesture's inertia: still moving.
    rule.controllerMoved(true);
    expect(rule.collisionAllowed(60_000)).toBe(true);
    rule.controllerMoved(false);
    expect(rule.collisionAllowed(60_000)).toBe(false);
  });

  it("counts a pointer held on the map as moving, however long it is held", () => {
    const rule = new CameraOwnership();
    rule.hold(true, 0);
    expect(rule.userHasCamera).toBe(true);
    expect(rule.moving(1_000_000)).toBe(true);
    rule.hold(false, 1_000_000);
    expect(rule.moving(1_000_000 + INPUT_SETTLE_MS - 1)).toBe(true);
    expect(rule.moving(1_000_000 + INPUT_SETTLE_MS + 1)).toBe(false);
  });
});

/** A viewer with what CameraController reads, and a clock the test moves. */
function stubViewer() {
  const clock = { now: 10_000 };
  vi.spyOn(performance, "now").mockImplementation(() => clock.now);
  const container = document.createElement("div");
  const canvas = document.createElement("canvas");
  container.appendChild(canvas);
  document.body.appendChild(container);
  /** Whether collision was on, each time CesiumJS's controller updated. */
  const collisions: boolean[] = [];
  const controller = {
    enableCollisionDetection: true,
    enableInputs: true,
    minimumZoomDistance: 0.6,
    update(this: { enableCollisionDetection: boolean }) {
      collisions.push(this.enableCollisionDetection);
    },
  };
  // A camera a metre over the ground at 47.6° N, looking down at 45°.
  const position = Cartesian3.fromDegrees(-122.13, 47.64, 101);
  const camera = {
    percentageChanged: 0.01,
    changed: new Event(),
    moveEnd: new Event(),
    transform: Matrix4.clone(Matrix4.IDENTITY),
    positionWC: position,
    positionCartographic: Cartographic.fromCartesian(position),
    directionWC: new Cartesian3(0, 0, -1),
    heading: 0,
    pitch: CesiumMath.toRadians(-45),
    roll: 0,
    frustum: { fovy: CesiumMath.PI_OVER_THREE, near: 1 },
    cancelFlight: vi.fn(),
    setView: vi.fn(),
    flyTo: vi.fn(),
    getPickRay: () => undefined,
  };
  /** The drawn surface under the camera: 2 m above it, as a coarse world tile can be. */
  const surface = { height: 103 };
  const scene = {
    screenSpaceCameraController: controller,
    preRender: new Event(),
    preUpdate: new Event(),
    sampleHeightSupported: true,
    sampleHeight: vi.fn(() => surface.height),
    pickPositionSupported: false,
    requestRender: vi.fn(),
    globe: { ellipsoid: Ellipsoid.WGS84, getHeight: () => 100, pick: () => undefined },
  };
  const viewer = { scene, camera, canvas, container, terrainProvider: {} };
  const controllerUnderTest = new CameraController(viewer as never, new Emitter<SceneEvents>());
  return { controllerUnderTest, controller, collisions, clock, canvas, container, camera, scene };
}

describe("CameraController and the camera rule", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    document.body.innerHTML = "";
  });

  it("keeps CesiumJS's terrain collision on a moving camera, and off one the person left at rest", () => {
    const { controllerUnderTest, controller, collisions, clock, container } = stubViewer();
    // Untouched (the app's flight has just landed): collision as before.
    controller.update();
    expect(collisions.at(-1)).toBe(true);
    // A wheel notch over the map: the zoom, and its inertia, are stopped at the ground.
    container.dispatchEvent(new WheelEvent("wheel", { deltaY: -100 }));
    expect(controllerUnderTest.userHasCamera).toBe(true);
    clock.now += 500;
    controller.update();
    expect(collisions.at(-1)).toBe(true);
    // The wheel has stopped: a finer terrain tile landing now must not lift the close-up.
    clock.now += INPUT_SETTLE_MS;
    controller.update();
    expect(collisions.at(-1)).toBe(false);
    // The setting itself is left as it was, for whoever reads it.
    expect(controller.enableCollisionDetection).toBe(true);
    controllerUnderTest.destroy();
  });

  it("lifts a resting camera the app put under the drawn surface, never one the person put there", () => {
    const untouched = stubViewer();
    const lifted = vi.spyOn(untouched.controllerUnderTest, "glide");
    untouched.camera.moveEnd.raiseEvent();
    untouched.clock.now += 300;
    vi.advanceTimersByTime(300);
    // The app's own landing: eased up off the surface, as before.
    expect(lifted).toHaveBeenCalledTimes(1);
    untouched.controllerUnderTest.destroy();
    vi.restoreAllMocks();
    document.body.innerHTML = "";

    const touched = stubViewer();
    const glide = vi.spyOn(touched.controllerUnderTest, "glide");
    // The person zooms to a metre over the ground, under the coarse tile's surface, and stops.
    touched.container.dispatchEvent(new WheelEvent("wheel", { deltaY: -100 }));
    touched.camera.moveEnd.raiseEvent();
    for (let i = 0; i < 20; i++) {
      touched.clock.now += 1_000;
      vi.advanceTimersByTime(1_000);
      touched.camera.moveEnd.raiseEvent();
    }
    expect(glide).not.toHaveBeenCalled();
    touched.controllerUnderTest.destroy();
  });

  it("gives the camera back to the app only when the app flies it", () => {
    const { controllerUnderTest, canvas } = stubViewer();
    canvas.dispatchEvent(new PointerEvent("pointerdown", { button: 0 }));
    window.dispatchEvent(new PointerEvent("pointerup", { button: 0 }));
    expect(controllerUnderTest.userHasCamera).toBe(true);
    controllerUnderTest.flyHome();
    expect(controllerUnderTest.userHasCamera).toBe(false);
    controllerUnderTest.takenByUser();
    expect(controllerUnderTest.userHasCamera).toBe(true);
    controllerUnderTest.destroy();
  });
});
