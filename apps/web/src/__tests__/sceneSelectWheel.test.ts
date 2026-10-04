/**
 * The scene-selection controller listens to the wheel only while the wheel means something to
 * it (cesium/sceneSelect/SceneSelectController.ts): a brush to size, or candidates to cycle.
 * Its listener is a window one, in the capture phase and allowed to cancel, so while it is
 * attached every wheel tick -- the globe's own zoom included -- waits for script first.
 */

import { afterEach, describe, expect, it, vi } from "vitest";

import { SceneSelectController } from "@/cesium/sceneSelect/SceneSelectController";
import { useSceneSelect } from "@/state/sceneSelect";

function viewer() {
  return {
    scene: {
      screenSpaceCameraController: { enableInputs: true },
      requestRender: () => undefined,
    },
    camera: {},
    canvas: document.createElement("canvas"),
  };
}

/** How many of the calls recorded on `spy` were for the wheel. */
function wheelListeners(spy: { mock: { calls: unknown[][] } }): number {
  return spy.mock.calls.filter(([type]) => type === "wheel").length;
}

describe("the selection's wheel listener", () => {
  afterEach(() => {
    useSceneSelect.getState().clear();
    useSceneSelect.getState().setMode("pick");
    vi.restoreAllMocks();
  });

  it("is attached only while there are candidates to cycle or a brush to size", () => {
    const added = vi.spyOn(window, "addEventListener");
    const removed = vi.spyOn(window, "removeEventListener");
    const controller = new SceneSelectController(viewer() as never);
    expect(controller.listensToWheel).toBe(false);
    expect(wheelListeners(added)).toBe(0);

    // One candidate: nothing to cycle.
    useSceneSelect.getState().select("scan", [7], 1, 0, null);
    expect(controller.listensToWheel).toBe(false);

    useSceneSelect.getState().select("scan", [7, 3, 1], 3, 0, null);
    expect(controller.listensToWheel).toBe(true);
    expect(wheelListeners(added)).toBe(1);
    // Alt and the wheel cycle, as before.
    window.dispatchEvent(new WheelEvent("wheel", { altKey: true, deltaY: 1, cancelable: true }));
    expect(useSceneSelect.getState().index).toBe(1);
    // A plain wheel is the globe's.
    const plain = new WheelEvent("wheel", { deltaY: 1, cancelable: true });
    window.dispatchEvent(plain);
    expect(plain.defaultPrevented).toBe(false);
    expect(useSceneSelect.getState().index).toBe(1);

    useSceneSelect.getState().clear();
    expect(controller.listensToWheel).toBe(false);
    expect(wheelListeners(removed)).toBe(1);

    // The brush sizes with Alt and the wheel.
    useSceneSelect.getState().setMode("paint");
    expect(controller.listensToWheel).toBe(true);
    const brush = useSceneSelect.getState().brush;
    window.dispatchEvent(new WheelEvent("wheel", { altKey: true, deltaY: -1, cancelable: true }));
    expect(useSceneSelect.getState().brush).toBeGreaterThan(brush);

    controller.destroy();
    expect(controller.listensToWheel).toBe(false);
    expect(wheelListeners(removed)).toBe(2);
  });
});
