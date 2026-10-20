import { useEffect } from "react";
import { Cartesian3, Color, PolygonHierarchy, type Entity } from "cesium";

import { polygonsOf } from "@twin/geo";

import { useAsCaptured } from "@/state/asCaptured";
import { useLandMap, type MapTone } from "@/state/landMap";

import { repaintUntilDrawn } from "./repaintUntilDrawn";
import { useScene } from "./SceneContext";

const TONES: Record<MapTone, string> = {
  accent: "#7fd8c0",
  agent: "#9db8f0",
  warn: "#f0b254",
  neutral: "#f2f6ef",
};

/**
 * Outlines the place a card is pointing at; the leader line ends inside it. A card in the
 * agent's tone points at the agent's work, which As captured does not draw (ADR 0019).
 */
export function LandFocusBridge() {
  const host = useScene();
  useEffect(() => {
    if (!host) return;
    let entities: Entity[] = [];
    let stopRepaint: () => void = () => undefined;
    const positions = (ring: number[][]) =>
      Cartesian3.fromDegreesArray(ring.flatMap((p) => [p[0] ?? 0, p[1] ?? 0]));
    const show = () => {
      if (host.isDestroyed) return;
      entities.forEach((entity) => host.viewer.entities.remove(entity));
      entities = [];
      const focus = useLandMap.getState().focus;
      if (focus && !(focus.tone === "agent" && useAsCaptured.getState().on)) {
        const color = Color.fromCssColorString(TONES[focus.tone]);
        const add = (options: Entity.ConstructorOptions) =>
          entities.push(
            host.viewer.entities.add({ id: `land-focus:${entities.length}`, ...options }),
          );
        const geometry = focus.geometry;
        if (geometry.type === "LineString")
          add({
            polyline: {
              positions: positions(geometry.coordinates),
              width: 7,
              material: color.withAlpha(0.85),
              clampToGround: true,
            },
          });
        else if (geometry.type === "Polygon" || geometry.type === "MultiPolygon")
          polygonsOf(geometry).forEach((polygon) => {
            const outer = polygon[0];
            if (!outer) return;
            add({
              polygon: {
                hierarchy: new PolygonHierarchy(
                  positions(outer),
                  polygon.slice(1).map((ring) => new PolygonHierarchy(positions(ring))),
                ),
                material: color.withAlpha(0.2),
              },
            });
            polygon.forEach((ring) =>
              add({
                polyline: {
                  positions: positions(ring),
                  width: 3.5,
                  material: color,
                  clampToGround: true,
                },
              }),
            );
          });
      }
      stopRepaint();
      stopRepaint = repaintUntilDrawn(host);
    };
    const off = useLandMap.subscribe((next, previous) => {
      if (next.focus !== previous.focus) show();
    });
    const offAsCaptured = useAsCaptured.subscribe((next, previous) => {
      if (next.on !== previous.on) show();
    });
    show();
    return () => {
      off();
      offAsCaptured();
      stopRepaint();
      if (!host.isDestroyed) entities.forEach((entity) => host.viewer.entities.remove(entity));
    };
  }, [host]);
  return null;
}
