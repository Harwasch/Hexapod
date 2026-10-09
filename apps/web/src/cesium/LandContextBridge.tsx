import { useEffect } from "react";
import { Cartesian3, Color, HeightReference, PolygonHierarchy, type Entity } from "cesium";
import { polygonsOf } from "@twin/geo";
import { useLandContext, type LandContextLayer } from "@/state/landContext";
import { useLand } from "@/state/land";
import { useUi } from "@/state/ui";
import { useLandIdentity } from "@/state/landIdentity";
import { useScene } from "./SceneContext";

/** Separate, disposable analytical/candidate overlays; never change the land scope. */
export function LandContextBridge() {
  const host = useScene();
  useEffect(() => {
    if (!host) return;
    const rendered = new Map<string, { layer: LandContextLayer; entities: Entity[] }>();
    const positions = (points: number[][]) =>
      Cartesian3.fromDegreesArray(points.flatMap((point) => [point[0] ?? 0, point[1] ?? 0]));
    const sync = () => {
      if (host.isDestroyed) return;
      const { layers } = useLandContext.getState();
      host.viewer.entities.suspendEvents();
      try {
        for (const [id, current] of rendered) {
          if (current.layer === layers[id]) continue;
          current.entities.forEach((entity) => host.viewer.entities.remove(entity));
          rendered.delete(id);
        }
        for (const layer of Object.values(layers)) {
          if (rendered.has(layer.id)) continue;
          const entities: Entity[] = [];
          for (const feature of layer.features) {
            const color = Color.fromCssColorString(
              layer.selectedIds?.includes(feature.id) ? "#f1cb7a" : "#88c8e5",
            );
            const id = `land-context:${layer.id}/${feature.id}`;
            const add = (options: Entity.ConstructorOptions) =>
              entities.push(host.viewer.entities.add(options));
            const geometry = feature.geometry;
            if (geometry.type === "Point") {
              add({
                id,
                name: feature.label,
                position: Cartesian3.fromDegrees(
                  geometry.coordinates[0] ?? 0,
                  geometry.coordinates[1] ?? 0,
                ),
                point: {
                  pixelSize: 12,
                  color,
                  outlineColor: Color.WHITE,
                  outlineWidth: 1,
                  heightReference: HeightReference.CLAMP_TO_GROUND,
                  disableDepthTestDistance: Number.POSITIVE_INFINITY,
                },
              });
            } else if (geometry.type === "LineString") {
              add({
                id,
                name: feature.label,
                polyline: {
                  positions: positions(geometry.coordinates),
                  width: layer.selectedIds?.includes(feature.id) ? 6 : 3,
                  material: color,
                  clampToGround: true,
                },
              });
            } else {
              polygonsOf(geometry).forEach((polygon, index) => {
                const outer = polygon[0];
                if (!outer) return;
                add({
                  id: `${id}#${index}`,
                  name: feature.label,
                  polygon: {
                    hierarchy: new PolygonHierarchy(
                      positions(outer),
                      polygon.slice(1).map((ring) => new PolygonHierarchy(positions(ring))),
                    ),
                    material: color.withAlpha(0.18),
                  },
                  polyline: {
                    positions: positions(outer),
                    width: 2,
                    material: color,
                    clampToGround: true,
                  },
                });
              });
            }
          }
          rendered.set(layer.id, { layer, entities });
        }
      } finally {
        host.viewer.entities.resumeEvents();
      }
      host.scene.requestRender();
    };
    const offInventoryPick = host.events.on("land-feature-select", ({ id }) => {
      useLandContext.getState().selectInventory(id);
      useUi.getState().setPanel("land");
    });
    const offContext = useLandContext.subscribe(sync);
    const offLand = useLand.subscribe((next, previous) => {
      if (next.active?.id !== previous.active?.id) useLandContext.getState().clear();
      else if (previous.mode === "candidates" && next.mode !== "candidates")
        useLandContext.getState().removeLayer("candidates");
    });
    const offIdentity = useLandIdentity.subscribe((next, previous) => {
      if (next.principalId !== previous.principalId || next.workspaceId !== previous.workspaceId)
        useLandContext.getState().clear();
    });
    sync();
    return () => {
      offInventoryPick();
      offContext();
      offLand();
      offIdentity();
      if (!host.isDestroyed)
        for (const current of rendered.values())
          current.entities.forEach((entity) => host.viewer.entities.remove(entity));
    };
  }, [host]);
  return null;
}
