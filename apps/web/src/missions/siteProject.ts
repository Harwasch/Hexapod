/**
 * A mission project for a catalog site that has no fleet backend yet: one zone (the site
 * boundary), no machines, and plans from the API. Planning works from day one; machines
 * arrive when a robot bridge registers them through the MissionProvider seam.
 */

import type { Site } from "@twin/contracts";
import { centerOf, toAcres } from "@twin/geo";

import type { Project } from "./types";

export const SITE_ZONE_ID = "SITE";

export function siteProjectId(siteId: string): string {
  return `site-${siteId}`;
}

export function siteProject(site: Site): Project {
  const acres = Math.round(toAcres(site.areaM2));
  // The catalog computes the centroid; a fallback record may omit it.
  const centroid = (site.centroid as Site["centroid"] | undefined) ?? centerOf(site.boundary);
  return {
    id: siteProjectId(site.id),
    siteId: site.id,
    name: site.name,
    meta: `${acres.toLocaleString()} acres · no fleet registered · plans only`,
    simulated: false,
    machines: [],
    zones: [
      {
        id: SITE_ZONE_ID,
        name: `${site.name} boundary`,
        short: "site",
        footprint: site.boundary,
        tone: "teal",
        treated: false,
        acres,
        task: "Treatment",
        machines: "unassigned",
        progressPct: 0,
        note: "The whole site. Zones arrive with the fleet backend; until then plans cover the boundary.",
        anchor: { longitude: centroid.longitude, latitude: centroid.latitude },
      },
    ],
    plans: [],
    agent: {
      headline: "Agent idle",
      summary: "Nothing running",
      footer:
        "Plans draft from the goal you write; a robot bridge will bring machines and telemetry.",
      actions: [],
    },
    fleetStats: [
      { value: "0", label: "Machines" },
      { value: "1", label: "Zone" },
      { value: String(acres), label: "Acres" },
    ],
    fleetNote:
      "No machines are registered for this site. Register them through the robot bridge to assign work.",
    workLog: [],
    feeds: [],
  };
}
