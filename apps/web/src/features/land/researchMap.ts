import { boundsOf } from "@twin/geo";
import type { LandMapGeometry } from "@twin/contracts";

/** Bounded iteration also works for large line samples without spreading them onto the stack. */
export function researchMapBounds(features: { geometry: LandMapGeometry }[]) {
  let west = Infinity,
    south = Infinity,
    east = -Infinity,
    north = -Infinity;
  const point = (coordinates: number[]) => {
    const [x, y] = coordinates;
    if (x === undefined || y === undefined || !Number.isFinite(x) || !Number.isFinite(y)) return;
    west = Math.min(west, x);
    east = Math.max(east, x);
    south = Math.min(south, y);
    north = Math.max(north, y);
  };
  for (const { geometry } of features) {
    if (geometry.type === "Point") point(geometry.coordinates);
    else if (geometry.type === "LineString") geometry.coordinates.forEach(point);
    else {
      const bounds = boundsOf(geometry);
      point([bounds.west, bounds.south]);
      point([bounds.east, bounds.north]);
    }
  }
  if (!Number.isFinite(west)) return null;
  const dx = Math.max(0.0005, (east - west) * 0.1),
    dy = Math.max(0.0005, (north - south) * 0.1);
  return {
    west: Math.max(-180, west - dx),
    south: Math.max(-90, south - dy),
    east: Math.min(180, east + dx),
    north: Math.min(90, north + dy),
  };
}
export function mapValue(value: number | null | undefined, unit?: string | null) {
  return value == null || !Number.isFinite(value)
    ? "No value"
    : `${value.toLocaleString(undefined, { maximumSignificantDigits: 8 })}${unit ? ` ${unit}` : ""}`;
}
