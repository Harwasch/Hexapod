import { useEffect } from "react";
import {
  Cartesian3,
  Color,
  ColorMaterialProperty,
  ConstantProperty,
  HeightReference,
  PolygonHierarchy,
  type Entity,
} from "cesium";
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
    const rendered = new Map<
      string,
      { layer: LandContextLayer; entities: { entity: Entity; featureId: string }[] }
    >();
    const positions = (points: number[][]) =>
      Cartesian3.fromDegreesArray(points.flatMap((point) => [point[0] ?? 0, point[1] ?? 0]));
    const sync = () => {
      if (host.isDestroyed) return;
      const { layers } = useLandContext.getState();
      host.viewer.entities.suspendEvents();
      try {
        for (const [id, current] of rendered) {
          if (current.layer === layers[id]) continue;
          const next = layers[id];
          if (
            next?.features === current.layer.features &&
            next.researchArtifactId === current.layer.researchArtifactId &&
            next.unit === current.layer.unit
          ) {
            for (const { entity, featureId } of current.entities) {
              const selected = Boolean(next.selectedIds?.includes(featureId));
              if (selected === Boolean(current.layer.selectedIds?.includes(featureId))) continue;
              const color = Color.fromCssColorString(selected ? "#f1cb7a" : "#88c8e5");
              if (entity.point) {
                entity.point.color = new ConstantProperty(color);
                entity.point.pixelSize = new ConstantProperty(selected ? 16 : 12);
              }
              if (entity.polyline) {
                entity.polyline.material = new ColorMaterialProperty(color);
                entity.polyline.width = new ConstantProperty(
                  entity.polygon ? (selected ? 4 : 2) : selected ? 6 : 3,
                );
              }
              if (entity.polygon)
                entity.polygon.material = new ColorMaterialProperty(
                  color.withAlpha(selected ? 0.28 : 0.18),
                );
            }
            current.layer = next;
            continue;
          }
          current.entities.forEach(({ entity }) => host.viewer.entities.remove(entity));
          rendered.delete(id);
        }
        for (const layer of Object.values(layers)) {
          if (rendered.has(layer.id)) continue;
          const entities: { entity: Entity; featureId: string }[] = [];
          for (const feature of layer.features) {
            const color = Color.fromCssColorString(
              layer.selectedIds?.includes(feature.id) ? "#f1cb7a" : "#88c8e5",
            );
            const id = `land-context:${layer.id}/${feature.id}`;
            const add = (options: Entity.ConstructorOptions) =>
              entities.push({
                entity: host.viewer.entities.add({
                  ...options,
                  ...(layer.researchArtifactId
                    ? {
                        properties: {
                          landResearchArtifactId: layer.researchArtifactId,
                          landResearchFeatureId: feature.id,
                        },
                      }
                    : {}),
                }),
                featureId: feature.id,
              });
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
                  pixelSize: layer.selectedIds?.includes(feature.id) ? 16 : 12,
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
                    material: color.withAlpha(
                      layer.selectedIds?.includes(feature.id) ? 0.28 : 0.18,
                    ),
                  },
                  polyline: {
                    positions: positions(outer),
                    width: layer.selectedIds?.includes(feature.id) ? 4 : 2,
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
    const offResearchPick = host.events.on(
      "land-research-feature-select",
      ({ layerId, featureId }) => {
        const state = useLandContext.getState();
        const layer = state.layers[layerId];
        if (
          !layer?.researchArtifactId ||
          !layer.features.some((feature) => feature.id === featureId)
        )
          return;
        state.selectMapFeature(layerId, featureId, "map");
        useUi.getState().setPanel("land");
      },
    );
    const offContext = useLandContext.subscribe((next, previous) => {
      if (next.layers !== previous.layers) sync();
    });
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
      offResearchPick();
      offContext();
      offLand();
      offIdentity();
      if (!host.isDestroyed)
        for (const current of rendered.values())
          current.entities.forEach(({ entity }) => host.viewer.entities.remove(entity));
    };
  }, [host]);
  return null;
}
