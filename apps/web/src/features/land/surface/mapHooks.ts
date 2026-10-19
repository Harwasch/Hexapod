import { useEffect, useId, type RefObject } from "react";

import type { Footprint, LandMapGeometry } from "@twin/contracts";

import { useScene } from "@/cesium/SceneContext";
import type { LandPoint } from "@/state/land";
import { registerLeader, type MapTone } from "@/state/landMap";

/** A place on the map a card can point at. */
export interface MapTarget {
  key: string;
  geometry: LandMapGeometry | Footprint;
  anchor?: LandPoint;
}

/** Draws a line from the element to the anchor while `active`; nothing re-renders per frame. */
export function useLeader(
  ref: RefObject<HTMLElement | null>,
  anchor: LandPoint | null | undefined,
  tone: MapTone = "accent",
  active = true,
): void {
  const id = useId();
  const lon = anchor?.[0];
  const lat = anchor?.[1];
  useEffect(() => {
    const element = ref.current;
    if (!element || lon === undefined || lat === undefined || !active) return;
    return registerLeader({ id, element, anchor: [lon, lat], tone });
  }, [ref, id, lon, lat, tone, active]);
}

/** Frames a shape with room around it; a point gets a neighbourhood a few hundred metres wide. */
export function useFlyTo(): (geometry: LandMapGeometry | Footprint) => void {
  const scene = useScene();
  return (geometry) => {
    if (!scene) return;
    const points: number[][] =
      geometry.type === "Point"
        ? [geometry.coordinates]
        : geometry.type === "LineString"
          ? geometry.coordinates
          : geometry.type === "Polygon"
            ? geometry.coordinates.flat()
            : geometry.coordinates.flat(2);
    let west = Infinity;
    let south = Infinity;
    let east = -Infinity;
    let north = -Infinity;
    for (const [lon = 0, lat = 0] of points) {
      west = Math.min(west, lon);
      east = Math.max(east, lon);
      south = Math.min(south, lat);
      north = Math.max(north, lat);
    }
    const dx = Math.max(0.002, east - west);
    const dy = Math.max(0.002, north - south);
    scene.camera.flyToRectangle(
      Math.max(-180, west - dx * 0.6),
      Math.max(-90, south - dy * 0.5),
      Math.min(180, east + dx * 0.6),
      Math.min(90, north + dy * 0.5),
    );
  };
}
