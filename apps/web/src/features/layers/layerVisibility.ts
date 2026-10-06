import type { Layer } from "@twin/contracts";

import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { recordAction } from "@/state/history";
import { useLayers } from "@/state/layers";

type LayerRef = Pick<Layer, "id" | "render">;

/**
 * Shows or hides a catalog layer as one undoable step (`state/history.ts`): what the layer
 * switches, the command box and the layer pills all go through here. Turning on a layer of
 * an exclusive group (a basemap) switches the group's other one off in the scene
 * (`LayerManager.setVisible`); its undo turns that one back on too.
 */
export function setLayerVisible(
  scene: Pick<CesiumSceneManager, "layers"> | null,
  layer: LayerRef & Pick<Layer, "name">,
  visible: boolean,
  catalog: readonly LayerRef[],
): void {
  if (!scene) return;
  const runtime = useLayers.getState().runtime;
  if (Boolean(runtime[layer.id]?.visible) === visible) {
    void scene.layers.setVisible(layer.id, visible);
    return;
  }
  const group = layer.render.exclusiveGroup;
  const displaced =
    visible && group
      ? catalog
          .filter((l) => l.id !== layer.id && l.render.exclusiveGroup === group)
          .filter((l) => runtime[l.id]?.visible)
          .map((l) => l.id)
      : [];
  recordAction(
    `${visible ? "Show" : "Hide"} ${layer.name}`,
    () => void scene.layers.setVisible(layer.id, visible),
    () => {
      void scene.layers.setVisible(layer.id, !visible);
      for (const id of displaced) void scene.layers.setVisible(id, true);
    },
  );
}
