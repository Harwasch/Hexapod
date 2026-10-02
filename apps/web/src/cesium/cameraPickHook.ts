/**
 * Cesium's camera control (ScreenSpaceCameraController) reads the depth buffer back from the
 * GPU to find the point under the cursor -- at each gesture start, and on every frame of a
 * zoom near the ground. A read-back waits for the GPU to finish its frame, which with a
 * splat scan on screen is 100 to 300 ms of the main thread, so a click or a key press
 * behind it waits too. The engine patch asks `pickHook` first; here it answers from the
 * splats' solids (SplatCollider, CPU), and the depth pick runs only where no splat is.
 */

import * as CesiumBarrel from "cesium";
import type { Cartesian2, Cartesian3, Ray } from "cesium";

import type { SplatCollider } from "./SplatCollider";

type PickHook = (windowPosition: Cartesian2, ray: Ray) => Cartesian3 | undefined;

interface ControllerModule {
  pickHook?: PickHook;
}

function controllerModule(): ControllerModule | undefined {
  const candidate = (CesiumBarrel as unknown as Record<string, unknown>)
    .ScreenSpaceCameraController;
  return typeof candidate === "function" ? (candidate as unknown as ControllerModule) : undefined;
}

export function installCameraPickHook(collider: SplatCollider): () => void {
  const module = controllerModule();
  if (!module) return () => undefined;
  const hook: PickHook = (_window, ray) => collider.raycast(ray)?.point;
  module.pickHook = hook;
  return () => {
    if (module.pickHook === hook) module.pickHook = undefined;
  };
}
