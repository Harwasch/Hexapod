/**
 * "Plan a re-scan of weak areas": the weak parts of a dated capture become a scout plan.
 *
 * The API outlines the epoch's weak cells (weak-v1: fewer than half of a cell's splats pass
 * the capture checks, or the cell was never seen; ADR 0016) as at most three smoothed patches
 * (within one battery's lanes, with drone missions on), nearest the view first, each with the pattern its reasons ask
 * for (nadir lines, crossing obliques or an orbit) and a hint. Each patch becomes an area zone
 * on the site's project carrying its scout capture (`missions/scout.ts`): flown by the drone
 * class, at the height that gives the detail the patch asks for, with the pattern and camera
 * pitch it names, and an informational `rescanOf` back to the epoch. Sites are public, so the
 * zones are too: no `private`, no `land` (modelled on `features/knowledge/liftFog.ts`, whose
 * patches are a private Land's). A fresh plan composer then drafts a survey over them through
 * the usual flow (`adoptAreas`, `startPlanDraft`); the planner sizes each pattern.
 *
 * It reads mission zones and the API only, so it works under every splat renderer. Nothing
 * here changes the confidence map: only a new capture of the site does (C7: a re-scan is any
 * later epoch whose grid covers the cells).
 */

import type { Site, WeakArea, WeakAreas } from "@twin/contracts";
import { footprintAreaM2, toAcres } from "@twin/geo";

import { requestWeakAreas } from "@/api/queries";
import { ApiError } from "@/api/error";
import { env } from "@/app/env";
import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { describeError } from "@/lib/log";
import { nextAreaId, zoneFromFootprint } from "@/missions/areas";
import { DemoMissionProvider } from "@/missions/demo";
import { droneFootprintWidthM } from "@/missions/machineClasses";
import {
  DEFAULT_SCOUT,
  MAX_ALTITUDE_M,
  MIN_ALTITUDE_M,
  SCOUT_CAMERAS,
  scoutBudgetM2,
  scoutGeometry,
  scoutOf,
  scoutResolution,
  type Scout,
} from "@/missions/scout";
import { siteProject } from "@/missions/siteProject";
import type { Zone } from "@/missions/types";
import { useMission } from "@/state/mission";
import { useToasts } from "@/state/toasts";

import { adoptAreas } from "../mission/planFlow";
import { startPlanDraft } from "../mission/planDrafting";

export const RESCAN_ATTRIBUTION = "Confidence map weak-v1";
/** What a re-scan area's chip on the map says: its passes are a forecast, not evidence. */
export const RESCAN_SHORT = "would re-scan";
/** Patches asked for, as "Lift the fog" asks. */
export const RESCAN_MAX_AREAS = 3;

type RescanReason = NonNullable<NonNullable<Scout["rescanOf"]>["reasons"]>[number];
const REASONS: readonly RescanReason[] = [
  "views",
  "spread",
  "gsd",
  "accuracy",
  "unseen",
  "alignment",
];

/**
 * The height above take-off that flies `gsdM` (metres per pixel) over flat ground: the inverse
 * of `scoutGeometry`'s detail, rounded down to whole metres so the detail flown is never
 * coarser than asked, within 20 to 120 m (Part 107's 400 ft). The drone class's height when
 * the patch asks for no detail (its grid was not metric).
 */
export function altitudeForGsd(gsdM: number | null): number {
  if (gsdM === null || !Number.isFinite(gsdM) || gsdM <= 0) return DEFAULT_SCOUT.altitudeM;
  const camera = SCOUT_CAMERAS[DEFAULT_SCOUT.camera];
  const altitudeM = Math.floor((gsdM * camera.imageWidthPx) / droneFootprintWidthM(1));
  return Math.min(MAX_ALTITUDE_M, Math.max(MIN_ALTITUDE_M, altitudeM));
}

/** A weak patch's scout capture: the drone class's flight, flown as the patch asks. */
export function rescanScout(area: WeakArea, read: WeakAreas, siteId: string): Scout {
  return {
    ...DEFAULT_SCOUT,
    purpose: "re-scan-weak",
    pattern: area.pattern,
    gimbalPitchDeg: area.gimbalPitchDeg,
    altitudeM: altitudeForGsd(area.targetGsdM),
    rescanOf: {
      siteId,
      epochId: read.epochId,
      policy: read.policy,
      cells: area.cells,
      reasons: REASONS.filter((reason) => (area.reasons[reason] ?? 0) > 0),
    },
  };
}

/** The patches as zones on the current project, after the ids it already uses. */
export function rescanZones(
  read: WeakAreas,
  siteId: string,
  existing: readonly Pick<Zone, "id">[],
): Zone[] {
  const zones: Zone[] = [];
  read.areas.forEach((area, index) => {
    const id = nextAreaId([...existing, ...zones]);
    zones.push({
      ...zoneFromFootprint(id, `Weak area ${index + 1}`, area.footprint),
      short: RESCAN_SHORT,
      task: "Re-scan",
      note: area.hint,
      attribution: RESCAN_ATTRIBUTION,
      scout: rescanScout(area, read, siteId),
    });
  });
  return zones;
}

/**
 * The re-scan goal: a survey at the detail flown, in the words `goalAtDetail` restates for a
 * new height. A survey for the planner, never a "scan" (that is a 3D reconstruction job, whose
 * questions and capture table would override the areas' own captures) nor a treatment.
 */
export function rescanGoal(gsdM: number): string {
  return `Survey the weak ground again from the air at ${Math.round(gsdM * 1000) / 10} cm per pixel so it can be measured again`;
}

/** What a 409 from the weak-areas route means, by `problem.code`. */
const CONFLICTS: Record<string, { title: string; body: string; tone: "info" | "error" }> = {
  "confidence-not-measured": {
    tone: "info",
    title: "Not measured",
    body: "This date has no confidence map (a splat upload carries no capture evidence), so nothing here can be called weak.",
  },
  "no-rescan-yet": {
    tone: "info",
    title: "No re-scan yet",
    body: "No later capture of this site has been measured yet.",
  },
  "confidence-unreadable": {
    tone: "error",
    title: "Could not read this date's confidence map",
    body: "The map beside its tiles is missing or damaged. Re-running the capture measures it again.",
  },
  "no-site-frame": {
    tone: "info",
    title: "This site has no frame to place weak cells by",
    body: "Its capture was not registered in a site frame, so its cells cannot be put on the globe.",
  },
};

/** The site's mission project, as the scene would make it (`SceneBridge`). */
function projectOf(site: Site) {
  return new DemoMissionProvider().projectForSite(site.id, site.slug) ?? siteProject(site);
}

/** Asks for the weak areas of `epochId` near the view and drafts a re-scan plan over them. */
export async function rescanWeak(
  scene: CesiumSceneManager | null,
  site: Site,
  epochId: string,
): Promise<void> {
  const before = useMission.getState().project;
  if (!before) {
    useToasts.getState().push({ tone: "info", title: "The map is still starting; try again." });
    return;
  }
  const center = scene?.camera.viewCenter() ?? null;
  // One battery's lanes bound a drone's patches; walked or filmed (no drone yet, drone
  // missions off) a battery is nobody's limit, so none is asked for or claimed. The budget is
  // the drone class's 60 m nadir lanes though a patch may fly lower (docs/DRONES.md).
  const drone = env.droneExportEnabled;
  let read: WeakAreas;
  try {
    read = await requestWeakAreas(site.id, epochId, {
      budgetM2: drone ? scoutBudgetM2(DEFAULT_SCOUT) : null,
      focus: center ? [center.longitude, center.latitude] : null,
      maxAreas: RESCAN_MAX_AREAS,
    });
  } catch (error) {
    const known =
      error instanceof ApiError && error.status === 409
        ? CONFLICTS[error.problem?.code ?? ""]
        : undefined;
    useToasts.getState().push(
      known ?? {
        tone: "error",
        title: "Could not find the weak areas",
        body: `${describeError(error)} Try again in a moment.`,
      },
    );
    return;
  }
  // The view went to another project while the API answered: the patches are not for it. By
  // id: the same project is composed afresh when its plans arrive or its site is renamed.
  if (useMission.getState().project?.id !== before.id) return;
  if (read.areas.length === 0) {
    // No area back is not "nothing weak": the API drops parts under 1 m² (MIN_AREA_M2) and
    // what the site's boundary clips away, so weak cells may remain that no patch holds.
    const left = read.weakCells;
    useToasts.getState().push(
      left > 0
        ? {
            tone: "info",
            title: `No weak patch big enough to plan (${read.policy})`,
            body: `${left.toLocaleString()} weak cell${left === 1 ? " remains" : "s remain"}, but only in scattered parts under 1 m² or outside the site, which are not planned. They stay weak until a new capture covers them.`,
          }
        : {
            tone: "info",
            title: `Nothing here is weak (${read.policy})`,
            body: "Every measured part of this date passes the capture checks. There is nothing to re-scan.",
          },
    );
    return;
  }
  // The patches are this site's ground: plan on its project (a new plan anyway, so a card
  // open on another project closes with it).
  if (before.siteId !== site.id) useMission.getState().setProject(projectOf(site));
  const project = useMission.getState().project;
  if (!project) return;
  const zones = rescanZones(read, site.id, project.zones);
  const acres = zones.reduce((sum, z) => sum + toAcres(footprintAreaM2(z.footprint)), 0);
  // The finest detail any patch asks for names the goal; each patch flies its own height.
  const gsdM = Math.min(...zones.map((z) => scoutGeometry(scoutOf(z.scout)).gsdM));
  const goal = rescanGoal(gsdM);
  const count = `${zones.length} weak area${zones.length === 1 ? "" : "s"}`;
  const note = `${count}, ${acres < 1 ? acres.toFixed(2) : Math.round(acres)} ac, of ${read.weakCells.toLocaleString()} weak cells chosen by ${RESCAN_ATTRIBUTION}${drone && read.truncatedByBudget ? " within one battery" : ""}. ${read.heuristic}`;
  // A fresh composer, as "Lift the fog" starts one: a card left open would hand the re-scan
  // its zones, machines, answers and, mid-revision, the approved plan it replaces.
  useMission.getState().openComposer({ goal });
  adoptAreas(zones, { source: "rescan", note });
  useMission.getState().setView("plan");
  useMission.getState().appendLog("agent", note);
  const zoneIds = zones.map((z) => z.id);
  scene?.mission.flyToZones(zoneIds);
  await startPlanDraft(goal, { zoneIds, answers: { resolution: scoutResolution(gsdM) } });
}
