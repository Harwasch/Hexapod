/**
 * Draws a measured splat's inferred layers (lib/inferred.ts) beside it: each its own
 * tileset, placed where the measured one is, shown only while it is shown and while
 * inferred layers are wanted (state/inferred.ts), and faded by its own view cones -- so a
 * fill made for the outside of a scan is never drawn from the inside.
 */
import { Cesium3DTileset, Matrix4, type Scene } from "cesium";

import { createLogger } from "@/lib/log";
import { evidenceOf, inferredLayersOf, resolveLayerUrl } from "@/lib/inferred";
import { useInferred } from "@/state/inferred";

import { keepOffscreenSplats } from "./splatInternals";
import { attachViewCones } from "./splatViewCones";

const log = createLogger("inferred");

export type LoadLayer = (url: string, parent: Cesium3DTileset) => Promise<Cesium3DTileset>;

async function loadLayer(url: string, parent: Cesium3DTileset): Promise<Cesium3DTileset> {
  const tileset = await Cesium3DTileset.fromUrl(url, {
    maximumScreenSpaceError: parent.maximumScreenSpaceError,
    show: false,
    enableCollision: false,
  });
  keepOffscreenSplats(tileset);
  tileset.cullRequestsWhileMoving = false;
  return tileset;
}

/** Loads `parent`'s inferred layers into `scene`; returns the disposer. */
export function attachInferredLayers(
  parent: Cesium3DTileset,
  scene: Pick<Scene, "primitives" | "preUpdate" | "requestRender">,
  assetId: string,
  load: LoadLayer = loadLayer,
): () => void {
  const refs = inferredLayersOf((parent.root as { extras?: unknown } | undefined)?.extras);
  const url = (parent as unknown as { resource?: { url?: string } }).resource?.url;
  if (!refs.length || !url) return () => undefined;
  const layers: { tileset: Cesium3DTileset; off: () => void }[] = [];
  let disposed = false;
  const sync = (): void => {
    const wanted = parent.show && useInferred.getState().show;
    for (const { tileset } of layers) {
      if (!Matrix4.equals(tileset.modelMatrix, parent.modelMatrix))
        tileset.modelMatrix = Matrix4.clone(parent.modelMatrix);
      if (tileset.show !== wanted) tileset.show = wanted;
    }
  };
  const offUpdate = scene.preUpdate.addEventListener(sync);
  const offStore = useInferred.subscribe(() => scene.requestRender());
  void Promise.allSettled(
    refs.map(async (ref) => {
      const tileset = await load(resolveLayerUrl(url, ref.uri), parent);
      if (disposed) {
        tileset.destroy();
        return;
      }
      // The layer's own root has the final say on what it is.
      const evidence =
        evidenceOf((tileset.root as { extras?: unknown } | undefined)?.extras) ?? ref.evidence;
      scene.primitives.add(tileset);
      layers.push({ tileset, off: attachViewCones(tileset) });
      useInferred
        .getState()
        .setLayers(assetId, [...(useInferred.getState().layers[assetId] ?? []), evidence]);
      sync();
      scene.requestRender();
    }),
  ).then((results) => {
    for (const result of results) {
      if (result.status === "rejected")
        log.warn("inferred layer did not load", { asset: assetId, error: String(result.reason) });
    }
  });
  return () => {
    disposed = true;
    offUpdate();
    offStore();
    for (const { tileset, off } of layers) {
      off();
      scene.primitives.remove(tileset);
    }
    layers.length = 0;
    useInferred.getState().setLayers(assetId, []);
  };
}
