/**
 * "Lift the fog": the foggiest ground near the view becomes a scout plan.
 *
 * The API picks the foggiest cells of the selected Land within one battery's lanes (nearest
 * the view when the fog is even), as smoothed patches. Each patch becomes an area zone that
 * carries its scout capture (`missions/scout.ts`: the drone class's flight), private and
 * marked as derived from the Land, so the plan goes to the workspace and never to browser
 * storage. The plan composer then drafts a survey over them through the usual flow
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
import { DEFAULT_SCOUT, SCOUT_SHORT, scoutBudgetM2, scoutGeometry } from "@/missions/scout";
import type { Zone } from "@/missions/types";
import { useMission } from "@/state/mission";
import { useToasts } from "@/state/toasts";

import { adoptAreas } from "../mission/planFlow";
import { startPlanDraft } from "../mission/planDrafting";
import { KNOWLEDGE_CELL_M, requestScoutAreas } from "./api";

export const FOGGY_THRESHOLD = 0.5;
export const SCOUT_ATTRIBUTION = "Fog of knowing fog-v1";

/** The scout goal: a survey (it must never say "clear", which the planner reads as treat). */
export function scoutGoal(gsdM = scoutGeometry(DEFAULT_SCOUT).gsdM): string {
  const cm = Math.round(gsdM * 1000) / 10;
  return `Survey the foggy ground from the air at ${cm} cm per pixel so the fog can lift`;
}

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
  if (read.areas.length === 0) {
    useToasts.getState().push({
      tone: "info",
      title: `Nothing here is foggier than ${FOGGY_THRESHOLD}`,
      body: "Every part of this land has recent enough evidence. There is nothing to scout.",
    });
    return;
  }
  const project = useMission.getState().project;
  if (!project) return;
  const zones = scoutZones(read.areas, land, project.zones);
  const acres = zones.reduce((sum, z) => sum + toAcres(footprintAreaM2(z.footprint)), 0);
  const goal = scoutGoal();
  const note = `${zones.length} fog patch${zones.length === 1 ? "" : "es"}, ${Math.round(acres)} ac, chosen by ${SCOUT_ATTRIBUTION}${read.truncatedByBudget ? " within one battery" : ""}. ${read.forecast}`;
  if (!useMission.getState().composer) useMission.getState().openComposer({ goal });
  adoptAreas(zones, { source: "fog", note });
  useMission.getState().setView("plan");
  useMission.getState().appendLog("agent", note);
  const zoneIds = zones.map((z) => z.id);
  scene?.mission.flyToZones(zoneIds);
  // The patches are flown at the class's height, about 2 cm per pixel: say so to the planner.
  await startPlanDraft(goal, { zoneIds, answers: { resolution: "2" } });
}
