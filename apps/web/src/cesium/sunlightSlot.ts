/**
 * Where a splat model's sunlight relight hooks into the scene (SiteManager `attachTileset`): a
 * few lines on the globe's first load; the relight itself (`sunlight.ts`, its shader part)
 * loaded the first time the sun is turned on. Off -- and always while As captured is on --
 * nothing is fetched and nothing joins the shader.
 */

import type { Cesium3DTileset, Scene } from "cesium";

import { useAsCaptured } from "@/state/asCaptured";
import { useLighting } from "@/state/lighting";

import type { startSunlight } from "./sunlight";

export function attachSunlight(
  tileset: Cesium3DTileset,
  scene: Scene,
  assetId: string,
  load: () => Promise<{ startSunlight: typeof startSunlight }> = () => import("./sunlight"),
): () => void {
  let handle: ReturnType<typeof startSunlight> | undefined;
  let loading = false;
  let disposed = false;
  const on = (): boolean => useLighting.getState().sun !== "off" && !useAsCaptured.getState().on;
  const apply = (): void => {
    if (handle) return handle.sync(on());
    if (!on() || loading) return;
    loading = true;
    load()
      .then((module) => {
        if (disposed) return;
        handle = module.startSunlight(tileset, scene, assetId);
        handle.sync(on());
      })
      // Not relit, and so not reported: the badge says nothing it did not draw. A later
      // switch tries again.
      .catch(() => {
        loading = false;
      });
  };
  const unsubscribe = [useLighting.subscribe(apply), useAsCaptured.subscribe(apply)];
  apply();
  return () => {
    disposed = true;
    for (const off of unsubscribe) off();
    handle?.dispose();
  };
}
