/**
 * "Make this the job": the lit ground, protected patches left out, becomes the areas of a new
 * plan on the saved Land.
 *
 * The API outlines the alike cells on the signature's analysis grid (`POST /similar/zones`; 10 m,
 * coarser on a large land): the largest patches each, the rest as one area, specks under half an
 * acre (and the rare patch too thin to store as an outline) left out and counted.
 * Each becomes an area zone that is private and marked as derived from the Land, so the plan
 * goes to the workspace and never to browser storage, with its provenance as the zone's
 * attribution. A fresh plan composer takes them (`adoptAreas`) and the bar asks what should
 * happen there; like-here areas of an earlier job that no plan uses are replaced.
 */

import type { LandArea, SignatureScene, SignatureZone, SignatureZones } from "@twin/contracts";

import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { LAND_AREA_PREFIX, nextAreaId, zoneFromFootprint } from "@/missions/areas";
import type { Zone } from "@/missions/types";
import { missionLandId, useMission } from "@/state/mission";
import { useShowOnce } from "@/state/showOnce";

import { adoptAreas } from "../mission/planFlow";
import { engageLand } from "../mission/landPlanning";
import { requestSignatureZones } from "./api";
import { formatAcres, protectedRectangles, spacingLabel } from "./grid";

export const LIKE_HERE_SHORT = "like here";
/**
 * Every like-here area's attribution starts so, whatever the grid's spacing (which follows it):
 * how an earlier job's areas are found again.
 */
export const LIKE_HERE_ATTRIBUTION = "Similar Sentinel-2 signature";
const ACRE_M2 = 4046.8564224;
const MAX_RECTANGLES = 5000;

export interface JobSummary {
  areas: number;
  acres: number;
  protectedAcres: number;
  dropped: { patches: number; acres: number };
  /** Patches big enough but too thin to store as an outline (rare), left out too. */
  unoutlined: { patches: number; acres: number };
}

function day(scene: SignatureScene): string {
  return scene.acquiredAt?.slice(0, 10) ?? `${scene.period.startDate} to ${scene.period.endDate}`;
}

export function isLikeHereZone(zone: Zone, landId: string): boolean {
  return (
    zone.land?.kind === "derived" &&
    zone.land.landId === landId &&
    (zone.attribution?.startsWith(LIKE_HERE_ATTRIBUTION) ?? false)
  );
}

/** The areas as zones on the current project, after the ids it already uses. */
export function likeHereZones(
  read: SignatureZones,
  land: Pick<LandArea, "id" | "revision">,
  existing: readonly Pick<Zone, "id">[],
): Zone[] {
  const pct = Math.round(read.threshold * 100);
  const { cellLatitude, cellLongitude } = read.seed;
  const where = `${cellLatitude.toFixed(5)}, ${cellLongitude.toFixed(5)}`;
  const attribution =
    `${LIKE_HERE_ATTRIBUTION} at ${spacingLabel(read.nativeResolutionM)} to the cell at ` +
    `${where}, at least ${pct}% alike; scenes ${read.scenes.map(day).join(", ")} · ` +
    read.attribution;
  const zones: Zone[] = [];
  const add = (area: SignatureZone, name: string) => {
    const id = nextAreaId([...existing, ...zones], LAND_AREA_PREFIX);
    zones.push({
      ...zoneFromFootprint(id, name, area.footprint),
      tone: "amber",
      short: LIKE_HERE_SHORT,
      note:
        `Similar seasonal satellite signature to the place you tapped (${where}): at least ` +
        `${pct}% alike, ${Math.round(area.meanSimilarity * 100)}% on average. Not a species ID.`,
      attribution: attribution.slice(0, 300),
      private: true,
      land: { kind: "derived", landId: land.id, featureId: null, boundaryRevision: land.revision },
    });
  };
  read.zones.forEach((area, index) =>
    add(area, `Like-here patch ${index + 1} (${formatAcres(area.areaM2 / ACRE_M2)} ac)`),
  );
  if (read.remainder) {
    const { patches, areaM2 } = read.remainder;
    add(
      read.remainder,
      `Like-here: ${patches} smaller patch${patches === 1 ? "" : "es"} (${formatAcres(areaM2 / ACRE_M2)} ac)`,
    );
  }
  return zones;
}

/**
 * Outlines the lit ground and opens a plan composer on the Land with it. Returns what became
 * of it (no areas: nothing big enough at this setting), or throws with what went wrong.
 */
export async function makeJob(
  land: LandArea,
  scene: CesiumSceneManager | null,
): Promise<JobSummary> {
  const { session, result, threshold, guarded } = useShowOnce.getState();
  if (!session || !result) throw new Error("Compare a place first.");
  const protectedGround = protectedRectangles(result.grid, guarded);
  if (protectedGround.length > MAX_RECTANGLES)
    throw new Error(
      "The protected ground is too scattered to send. Free a few patches and try again.",
    );
  const read = await requestSignatureZones(session.rasterId, {
    longitude: result.read.seed.longitude,
    latitude: result.read.seed.latitude,
    threshold,
    protected: protectedGround,
  });
  const summary: JobSummary = {
    areas: read.zones.length + (read.remainder ? 1 : 0),
    acres:
      (read.zones.reduce((sum, zone) => sum + zone.areaM2, 0) + (read.remainder?.areaM2 ?? 0)) /
      ACRE_M2,
    protectedAcres: (read.protectedCells * read.cellAreaM2) / ACRE_M2,
    dropped: { patches: read.dropped.patches, acres: read.dropped.areaM2 / ACRE_M2 },
    unoutlined: { patches: read.unoutlined.patches, acres: read.unoutlined.areaM2 / ACRE_M2 },
  };
  if (summary.areas === 0) return summary;
  if (missionLandId() !== land.id) {
    // A card open on other ground would hold the switch back: this is a new plan anyway.
    useMission.getState().closeComposer();
    await engageLand(land);
  }
  const project = useMission.getState().project;
  if (!project || missionLandId() !== land.id)
    throw new Error(`${land.name} could not be opened as mission ground. Try again.`);
  // A job made again replaces the like-here areas of the last one, unless a plan keeps them.
  for (const zone of project.zones) {
    if (isLikeHereZone(zone, land.id) && !project.plans.some((p) => p.zoneIds.includes(zone.id)))
      useMission.getState().removeArea(project.id, zone.id);
  }
  const zones = likeHereZones(read, land, useMission.getState().project?.zones ?? []);
  const pct = Math.round(threshold * 100);
  const parts = [
    `${zones.length} like-here area${zones.length === 1 ? "" : "s"} on ${land.name}, ` +
      `${formatAcres(summary.acres)} ac: a similar Sentinel-2 signature at ` +
      `${spacingLabel(read.nativeResolutionM)} to the place you tapped, at least ${pct}% alike`,
  ];
  if (summary.protectedAcres > 0)
    parts.push(`${formatAcres(summary.protectedAcres)} ac you protected left out`);
  if (summary.dropped.patches > 0)
    parts.push(
      `${summary.dropped.patches} speck${summary.dropped.patches === 1 ? "" : "s"} under half an ` +
        `acre (${formatAcres(summary.dropped.acres)} ac) left out`,
    );
  if (summary.unoutlined.patches > 0)
    parts.push(
      `${summary.unoutlined.patches} patch${summary.unoutlined.patches === 1 ? "" : "es"} too ` +
        `thin to outline (${formatAcres(summary.unoutlined.acres)} ac) left out`,
    );
  const note = `${parts.join("; ")}.`;
  // A new plan, from a fresh composer, as "Plan work on this land" starts one.
  useMission.getState().openComposer();
  adoptAreas(zones, { source: "signature", note });
  useMission
    .getState()
    .appendLog(
      "agent",
      `I outlined ${zones.length} area${zones.length === 1 ? "" : "s"} (${formatAcres(summary.acres)} ac) that look like the spot you tapped. What should happen there?`,
    );
  scene?.mission.flyToZones(zones.map((z) => z.id));
  window.dispatchEvent(new CustomEvent("twin:bar", { detail: "" }));
  return summary;
}
