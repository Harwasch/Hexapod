import { useLayers as useLayerCatalog } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { recordAction } from "@/state/history";
import { useLayers } from "@/state/layers";
import { useMission } from "@/state/mission";

import { setLayerVisible } from "../layers/layerVisibility";

export interface QuickLayer {
  id: "imagery" | "vegetation" | "zones" | "tracks";
  label: string;
  on: boolean;
  disabled: boolean;
  toggle: () => void;
}

/**
 * The four layers an operator flips most: the basemap imagery, vegetation (land cover), and
 * the mission's zones and tracks. The layer pills draw them and the command box lists them,
 * from this one definition. Each flip is one undoable step (`state/history.ts`).
 */
export function useQuickLayers(): QuickLayer[] {
  const scene = useScene();
  const catalog = useLayerCatalog();
  const runtime = useLayers((s) => s.runtime);
  const mission = useMission((s) => s.layers);
  const project = useMission((s) => s.project);
  const catalogLayers = catalog.data ?? [];
  /** The mission's zones or tracks on or off, in the store and on the map. */
  const setMissionLayer = (key: "zones" | "tracks", on: boolean): void => {
    if (useMission.getState().layers[key] !== on) useMission.getState().toggleLayer(key);
    scene?.mission.setLayer(key, on);
  };
  const flipMission = (key: "zones" | "tracks", label: string): void => {
    const on = mission[key];
    recordAction(
      `${on ? "Hide" : "Show"} ${label}`,
      () => setMissionLayer(key, !on),
      () => setMissionLayer(key, on),
    );
  };

  const basemap =
    (catalog.data ?? []).find(
      (l) => l.render.exclusiveGroup === "basemap" && runtime[l.id]?.visible,
    ) ?? (catalog.data ?? []).find((l) => l.slug === "bing-maps-aerial");
  const vegetation =
    (catalog.data ?? []).find((l) => l.slug === "esa-worldcover-2021") ??
    (catalog.data ?? []).find((l) => l.category === "land-cover");
  const imageryOn = Boolean(basemap && runtime[basemap.id]?.visible);
  const vegetationOn = Boolean(vegetation && runtime[vegetation.id]?.visible);

  return [
    {
      id: "imagery",
      label: "Imagery",
      on: imageryOn,
      disabled: !basemap,
      toggle: () => basemap && setLayerVisible(scene, basemap, !imageryOn, catalogLayers),
    },
    {
      id: "vegetation",
      label: "Vegetation",
      on: vegetationOn,
      disabled: !vegetation,
      toggle: () => vegetation && setLayerVisible(scene, vegetation, !vegetationOn, catalogLayers),
    },
    {
      id: "zones",
      label: "Zones",
      on: mission.zones,
      disabled: !project,
      toggle: () => flipMission("zones", "zones"),
    },
    {
      id: "tracks",
      label: "Tracks",
      on: mission.tracks,
      disabled: !project,
      toggle: () => flipMission("tracks", "tracks"),
    },
  ];
}
