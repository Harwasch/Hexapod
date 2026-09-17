/**
 * Areas drawn in the console: zones the operator makes on the spot, anywhere on Earth, so a
 * plan never needs a fleet backend to name its ground. They are stored on the plans that
 * cover them and kept in the browser until then.
 */

import type { Footprint } from "@twin/contracts";
import { centerOf, destination, footprintAreaM2, toAcres, type LonLat } from "@twin/geo";

import type { Measurement } from "@/state/measurements";

import type { ViewAreaOrigin, Zone } from "./types";

export const AREA_PREFIX = "A-";

export function isAreaZone(zone: Pick<Zone, "id">): boolean {
  return zone.id.startsWith(AREA_PREFIX);
}

/** A-01, A-02 … past the highest number in use. */
export function nextAreaId(existing: readonly Pick<Zone, "id">[]): string {
  let max = 0;
  for (const zone of existing) {
    const match = /^A-(\d+)$/.exec(zone.id);
    if (match) max = Math.max(max, Number(match[1]));
  }
  return `${AREA_PREFIX}${String(max + 1).padStart(2, "0")}`;
}

export function zoneFromFootprint(id: string, name: string, footprint: Footprint): Zone {
  const acres = Math.max(1, Math.round(toAcres(footprintAreaM2(footprint))));
  return {
    id,
    name,
    short: "area",
    footprint,
    tone: "blue",
    treated: false,
    acres,
    task: "Treatment",
    machines: "unassigned",
    progressPct: 0,
    note: "Drawn in the console.",
    anchor: centerOf(footprint),
  };
}

/** An axis-aligned rectangle around a point, in metres each way. */
export function rectangleFootprint(
  center: LonLat,
  halfWidthM: number,
  halfHeightM: number,
): Footprint {
  const north = destination(center, 0, halfHeightM).latitude;
  const south = destination(center, 180, halfHeightM).latitude;
  const east = destination(center, 90, halfWidthM).longitude;
  const west = destination(center, 270, halfWidthM).longitude;
  return {
    type: "Polygon",
    coordinates: [
      [
        [west, south],
        [east, south],
        [east, north],
        [west, north],
        [west, south],
      ],
    ],
  };
}

/** The ground the view is looking at: half the viewport around its centre, 60 m to 5 km each way. */
export function viewFootprint(
  center: LonLat,
  metersPerPixel: number,
  viewport: { width: number; height: number },
  fraction = 0.5,
  offset: { dx: number; dy: number } = { dx: 0, dy: 0 },
): Footprint {
  const clamp = (m: number) => Math.min(5000, Math.max(60, m));
  const halfW = clamp(metersPerPixel * viewport.width * fraction * 0.5);
  const halfH = clamp(metersPerPixel * viewport.height * fraction * 0.5);
  // Screen-right is east and screen-up is north for a north-up view; a heading would rotate
  // this, which the view rectangle ignores (it is a starting point, not a survey).
  const shifted = destination(
    destination(center, 90, metersPerPixel * viewport.width * offset.dx),
    0,
    -metersPerPixel * viewport.height * offset.dy,
  );
  return rectangleFootprint(shifted, halfW, halfH);
}

/** A zone from the view with the origin kept, so its size and position can be adjusted later. */
export function viewAreaZone(id: string, name: string, origin: ViewAreaOrigin): Zone {
  const zone = zoneFromFootprint(
    id,
    name,
    viewFootprint(
      origin.center,
      origin.metersPerPixel,
      { width: origin.width, height: origin.height },
      origin.fraction,
      { dx: origin.dx, dy: origin.dy },
    ),
  );
  return { ...zone, view: origin, note: "Made from the view; adjust its size and position." };
}

/** A finished area measurement as a zone. */
export function areaFromMeasurement(
  measurement: Measurement,
  id: string,
  name: string,
): Zone | null {
  if (measurement.mode !== "area") return null;
  // A double-click to finish repeats the last corner; repeated corners make no polygon.
  const ring: [number, number][] = [];
  for (const p of measurement.points) {
    const last = ring[ring.length - 1];
    if (last && Math.abs(last[0] - p.longitude) < 1e-7 && Math.abs(last[1] - p.latitude) < 1e-7)
      continue;
    ring.push([p.longitude, p.latitude]);
  }
  const first = ring[0];
  const last = ring[ring.length - 1];
  if (first && last && ring.length > 1 && first[0] === last[0] && first[1] === last[1]) ring.pop();
  if (ring.length < 3 || !first) return null;
  ring.push([first[0], first[1]]);
  const footprint: Footprint = { type: "Polygon", coordinates: [ring] };
  return footprintAreaM2(footprint) > 1 ? zoneFromFootprint(id, name, footprint) : null;
}
