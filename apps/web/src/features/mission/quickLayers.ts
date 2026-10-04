import { useLayers as useLayerCatalog } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { useLayers } from "@/state/layers";
import { useMission } from "@/state/mission";

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
 * from this one definition.
 */
export function useQuickLayers(): QuickLayer[] {
  const scene = useScene();
  const catalog = useLayerCatalog();
  const runtime = useLayers((s) => s.runtime);
  const mission = useMission((s) => s.layers);
  const toggle = useMission((s) => s.toggleLayer);
  const project = useMission((s) => s.project);

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
      toggle: () => basemap && void scene?.layers.setVisible(basemap.id, !imageryOn),
    },
    {
      id: "vegetation",
      label: "Vegetation",
      on: vegetationOn,
      disabled: !vegetation,
      toggle: () => vegetation && void scene?.layers.setVisible(vegetation.id, !vegetationOn),
    },
    {
      id: "zones",
      label: "Zones",
      on: mission.zones,
      disabled: !project,
      toggle: () => {
        toggle("zones");
        scene?.mission.setLayer("zones", !mission.zones);
      },
    },
    {
      id: "tracks",
      label: "Tracks",
      on: mission.tracks,
      disabled: !project,
      toggle: () => {
        toggle("tracks");
        scene?.mission.setLayer("tracks", !mission.tracks);
      },
    },
  ];
}
