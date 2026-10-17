/**
 * "Lift the fog": the foggiest ground near the view becomes a scout plan.
 *
 * The API picks the foggiest cells of the selected Land within one battery's lanes (nearest
 * the view when the fog is even), as smoothed patches. Each patch becomes an area zone that
 * carries its scout capture (`missions/scout.ts`: the drone class's flight), private and
 * marked as derived from the Land, so the plan goes to the workspace and never to browser
 * storage. They join the Land's mission ground (`engageLand`: its own project, or a fleet's it
 * is layered on), never whatever project happens to be current. A fresh plan composer then drafts a survey over them through the usual flow
 * (`adoptAreas`, `startPlanDraft`); the map draws their real flight lines dashed, as a
 * forecast, and the draft can be exported as a DJI route (`ScoutExport`). The patches stay
 * editable with the area handles like any drawn area.
 *
 * Nothing here lifts the fog itself: only new stored evidence does.
 */

import type { LandArea } from "@twin/contracts";
import { footprintAreaM2, toAcres } from "@twin/geo";

import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { describeError } from "@/lib/log";
import { nextAreaId, zoneFromFootprint } from "@/missions/areas";
import {
  DEFAULT_SCOUT,
  SCOUT_SHORT,
  scoutBudgetM2,
  scoutGeometry,
  scoutGoal,
  scoutResolution,
} from "@/missions/scout";
import type { Zone } from "@/missions/types";
import { landScope } from "@/state/landIdentity";
import { missionLandId, useMission } from "@/state/mission";
import { useToasts } from "@/state/toasts";

import { engageLand } from "../mission/landPlanning";
import { adoptAreas } from "../mission/planFlow";
import { startPlanDraft } from "../mission/planDrafting";
import { KNOWLEDGE_CELL_M, requestScoutAreas } from "./api";

export const FOGGY_THRESHOLD = 0.5;
export const SCOUT_ATTRIBUTION = "Fog of knowing fog-v1";

/** The patches as zones on the current project, after the ids it already uses. */
export function scoutZones(
  areas: readonly { footprint: Zone["footprint"]; reason: string }[],
  land: Pick<LandArea, "id" | "revision">,
  existing: readonly Pick<Zone, "id">[],
): Zone[] {
  const zones: Zone[] = [];
  areas.forEach((area, index) => {
    const id = nextAreaId([...existing, ...zones]);
    zones.push({
      ...zoneFromFootprint(id, `Fog patch ${index + 1}`, area.footprint),
      short: SCOUT_SHORT,
      task: "Scout",
      note: area.reason,
      attribution: SCOUT_ATTRIBUTION,
      scout: { ...DEFAULT_SCOUT },
      private: true,
      land: { kind: "derived", landId: land.id, featureId: null, boundaryRevision: land.revision },
    });
  });
  return zones;
}

/** Asks for the foggiest patches near the view and drafts a scout plan over them. */
export async function liftFog(scene: CesiumSceneManager | null, land: LandArea): Promise<void> {
  const state = useMission.getState();
  if (!state.project) {
    useToasts.getState().push({ tone: "info", title: "The map is still starting; try again." });
    return;
  }
  const center = scene?.camera.viewCenter() ?? null;
  // The patches are this session's private ground: a sign-out or workspace switch while the
  // API answers has dropped its private plans and ground, and the answer must not come back.
  const scope = landScope();
  let read;
  try {
    read = await requestScoutAreas(land.id, {
      threshold: FOGGY_THRESHOLD,
      budgetM2: scoutBudgetM2(DEFAULT_SCOUT),
      focus: center ? [center.longitude, center.latitude] : null,
      maxAreas: 3,
      cellM: KNOWLEDGE_CELL_M,
    });
  } catch (error) {
    useToasts.getState().push({
      tone: "error",
      title: "Could not find the foggy ground",
      body: `${describeError(error)} Try again in a moment.`,
    });
    return;
  }
  if (landScope() !== scope) return;
  if (read.areas.length === 0) {
    useToasts.getState().push({
      tone: "info",
      title: `Nothing here is foggier than ${FOGGY_THRESHOLD}`,
      body: "Every part of this land has recent enough evidence. There is nothing to scout.",
    });
    return;
  }
  // The fog follows the selected Land, the project need not: a site switch keeps the Land, and
  // a card on another Land holds its switch back. The patches are this Land's ground, so the
  // plan goes on its mission ground; this is a new plan anyway, so an open card closes.
  if (missionLandId() !== land.id) {
    useMission.getState().closeComposer();
    try {
      await engageLand(land);
    } catch (error) {
      if (landScope() !== scope) return;
      useToasts.getState().push({
        tone: "error",
        title: `Could not open ${land.name} for planning`,
        body: `${describeError(error)} Try again in a moment.`,
      });
      return;
    }
  }
  if (landScope() !== scope) return;
  const project = useMission.getState().project;
  if (!project || missionLandId() !== land.id) {
    useToasts.getState().push({
      tone: "error",
      title: `Could not open ${land.name} for planning`,
      body: "Select the land again, then lift the fog.",
    });
    return;
  }
  const zones = scoutZones(read.areas, land, project.zones);
  const acres = zones.reduce((sum, z) => sum + toAcres(footprintAreaM2(z.footprint)), 0);
  const { gsdM } = scoutGeometry(DEFAULT_SCOUT);
  const goal = scoutGoal(gsdM);
  const note = `${zones.length} fog patch${zones.length === 1 ? "" : "es"}, ${Math.round(acres)} ac, chosen by ${SCOUT_ATTRIBUTION}${read.truncatedByBudget ? " within one battery" : ""}. ${read.forecast}`;
  // A new plan, from a fresh composer (as "Plan on this land" starts one): a composer left
  // open would hand the scout its zones, its machines, its answers and, mid-revision, the
  // approved plan it replaces.
  useMission.getState().openComposer({ goal });
  adoptAreas(zones, { source: "fog", note });
  useMission.getState().setView("plan");
  useMission.getState().appendLog("agent", note);
  const zoneIds = zones.map((z) => z.id);
  scene?.mission.flyToZones(zoneIds);
  // The patches are flown at the class's height: say the detail that gives to the planner.
  await startPlanDraft(goal, { zoneIds, answers: { resolution: scoutResolution(gsdM) } });
}
