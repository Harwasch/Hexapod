import { useLayers as useLayerCatalog } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { useLayers } from "@/state/layers";
import { useMission } from "@/state/mission";

/** Quick layer toggles on the right edge (design: LAYER TOGGLES). */
export function LayerPills() {
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

  const pills = [
    {
      id: "imagery",
      label: "Imagery",
      on: imageryOn,
      disabled: !basemap,
      onClick: () => basemap && void scene?.layers.setVisible(basemap.id, !imageryOn),
    },
    {
      id: "vegetation",
      label: "Vegetation",
      on: vegetationOn,
      disabled: !vegetation,
      onClick: () => vegetation && void scene?.layers.setVisible(vegetation.id, !vegetationOn),
    },
    {
      id: "zones",
      label: "Zones",
      on: mission.zones,
      disabled: !project,
      onClick: () => {
        toggle("zones");
        scene?.mission.setLayer("zones", !mission.zones);
      },
    },
    {
      id: "tracks",
      label: "Tracks",
      on: mission.tracks,
      disabled: !project,
      onClick: () => {
        toggle("tracks");
        scene?.mission.setLayer("tracks", !mission.tracks);
      },
    },
  ];

  return (
    <div className="mc-pills" role="group" aria-label="Quick layers" data-testid="layer-pills">
      {pills.map((pill) => (
        <button
          key={pill.id}
          type="button"
          className={`glass mc-pill ${pill.on ? "is-on" : ""}`}
          aria-pressed={pill.on}
          disabled={pill.disabled}
          onClick={pill.onClick}
          data-testid={`pill-${pill.id}`}
        >
          {pill.label}
        </button>
      ))}
    </div>
  );
}
