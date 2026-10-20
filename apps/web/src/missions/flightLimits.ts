/**
 * What the DJI export refuses before it offers a download. Drone missions are a stub until
 * there is a drone (`VITE_ENABLE_DRONE_EXPORT`, off; docs/DRONES.md), so these keep a future
 * enablement from inheriting a dangerous default: a route is offered whole or not at all.
 *
 * Loaded only with the drone card (`DroneScoutExport`), never with the globe.
 */

import type { Footprint } from "@twin/contracts";

import { laneSets, orbitRing, scoutGeometry, type Scout } from "./scout";

/** A scout area as the export flies it, with the name the card calls it by. */
export interface FlownArea {
  name: string;
  footprint: Footprint;
  scout: Scout;
}

/**
 * Why a route cannot be offered because a pass of it could not be planned, naming each area
 * and what is missing; null when every area's passes are planned. A route with an area left
 * out, or an oblique with one of its crossing passes, would fly less than the plan holds.
 */
export function routeRefusal(areas: readonly FlownArea[]): string | null {
  if (areas.length === 0) return "Nothing to fly: there are no scout areas in this route.";
  const gaps = areas.flatMap(({ name, footprint, scout }): string[] => {
    if (scout.pattern === "orbit")
      return orbitRing(footprint, scout).length < 2
        ? [`${name}: no orbit can be drawn round it`]
        : [];
    const [first, crossing] = laneSets(footprint, scout);
    const sets = scout.pattern === "oblique" ? [first, crossing] : [first];
    const planned = sets.filter((lanes) => lanes.length > 0).length;
    if (planned === sets.length) return [];
    if (planned === 0) {
      const { lineSpacingM, altitudeM } = scoutGeometry(scout);
      return [
        `${name}: no lane fits (lines ${Math.round(lineSpacingM)} m apart at ${Math.round(altitudeM)} m), so nothing would be flown over it`,
      ];
    }
    return [`${name}: only one of its two crossing passes fits; the other would have no lane`];
  });
  return gaps.length > 0
    ? `Not every pass could be planned, so no route is offered. ${gaps.join(". ")}.`
    : null;
}
