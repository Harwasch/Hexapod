/**
 * A saved Land as the ground missions run on.
 *
 * Selecting a Land (`LandMissionBridge`) engages it: its boundary and inventory features
 * (`GET /land/{id}/ground`) become zones. Over a fleet's project whose ground is on the Land
 * they join that project, so the fleet stays; anywhere else the Land becomes a project of its
 * own, `land:<id>`, the id Land actions already schedule under. "Plan work on this land" opens
 * the plan card on it, and "Open in Plan" takes a scheduled action's mission there. Everything
 * on a Land is workspace-private (`planDrafting.ts` routes it; `state/mission.ts` never stores
 * it in the browser).
 */

import type { QueryClient } from "@tanstack/react-query";

import type { LandArea, PlanRecord } from "@twin/contracts";

import { api, unwrap } from "@/api/client";
import { fetchLandGround } from "@/api/queries";
import { describeError } from "@/lib/log";
import {
  boundaryZoneOf,
  groundModeFor,
  landIdOfProject,
  landProject,
  landProjectId,
  landZones,
} from "@/missions/landGround";
import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";
import { useMission } from "@/state/mission";
import { useUi } from "@/state/ui";

/**
 * Loads a Land's ground into the mission store: layered onto the current project when they
 * overlap, else as the Land's own project. With `layerOnly` (the project changed under an
 * engaged Land) it only layers, and never takes the operator off the project they chose.
 * Null when the selection moved on while the ground loaded, or nothing was engaged.
 */
export async function engageLand(
  land: LandArea,
  options: { layerOnly?: boolean } = {},
): Promise<"layer" | "project" | null> {
  const ground = await fetchLandGround(land.id, land.revision);
  if (useLand.getState().active?.id !== land.id) return null;
  const zones = landZones(ground);
  const store = useMission.getState();
  const own = landIdOfProject(store.project?.id) === land.id;
  if (!own && groundModeFor(store.project, land) === "layer") {
    store.engageLand({ landId: land.id, zones });
    return "layer";
  }
  if (options.layerOnly && !own) return null;
  store.engageLand({ landId: land.id, zones }, landProject(land, ground));
  return "project";
}

/**
 * "Plan work on this land": the plan card opens with the Land as its ground, and the bar takes
 * the sentence of work.
 */
export async function planOnLand(land: LandArea): Promise<void> {
  const say = (text: string) => useMission.getState().appendLog("agent", text);
  useMission.getState().closeComposer();
  try {
    if (!(await engageLand(land))) return;
  } catch (error) {
    say(`I couldn't load ${land.name} as ground: ${describeError(error)}`);
    useMission.getState().setStreamOpen(true);
    return;
  }
  const project = useMission.getState().project;
  const zones = project?.zones ?? [];
  const boundary = boundaryZoneOf(zones, land.id);
  if (!boundary) return;
  const features = zones.filter(
    (z) => z.land?.landId === land.id && z.land.kind === "feature",
  ).length;
  const mapped = features ? ` + ${features} mapped feature${features === 1 ? "" : "s"}` : "";
  useMission.getState().openComposer({ zoneIds: [boundary.id] });
  useMission.getState().updateComposer({
    ground: {
      source: "land",
      note: `${land.name}, ${boundary.acres.toLocaleString()} ac, saved boundary revision ${land.revision}${mapped}.`,
    },
  });
  say(`Planning on ${land.name}. Tell me what to do there, e.g. “monitor the fence every week”.`);
  window.dispatchEvent(new CustomEvent("twin:bar", { detail: "" }));
}

/**
 * "Open in Plan" for a mission a Land action scheduled: the plan opens where it is filed. A
 * mission filed under another project than the one in view is only pointed to.
 */
export async function openLandMission(
  land: LandArea,
  record: PlanRecord,
  client: QueryClient,
): Promise<void> {
  const store = useMission.getState();
  if (record.projectId === landProjectId(land.id)) {
    if (store.project?.id !== record.projectId) {
      store.closeComposer();
      try {
        const ground = await fetchLandGround(land.id, land.revision);
        useMission
          .getState()
          .engageLand({ landId: land.id, zones: landZones(ground) }, landProject(land, ground));
      } catch (error) {
        store.appendLog("agent", `I couldn't open ${land.name}'s plans: ${describeError(error)}`);
        store.setStreamOpen(true);
        return;
      }
    }
  } else if (record.projectId !== store.project?.id) {
    store.appendLog(
      "agent",
      `“${record.title}” is filed under the mission project ${record.projectId}. Switch to that project to see it in Plan.`,
    );
    store.setStreamOpen(true);
    return;
  }
  await client.invalidateQueries({ queryKey: ["workspace-plans"] });
  useMission.getState().openPlan(record.id);
}

/**
 * "Revise the action in Land": the Land panel opens on the Land's actions. The plan's Land is
 * selected first if another one is.
 */
export async function openLandActions(landId: string): Promise<void> {
  if (useLand.getState().active?.id !== landId) {
    try {
      const land = await unwrap(
        api.GET("/api/v1/land/{land_id}", { params: { path: { land_id: landId } } }),
      );
      useLand.getState().select(land);
    } catch (error) {
      useMission
        .getState()
        .appendLog("agent", `I couldn't open that land: ${describeError(error)}`);
      useMission.getState().setStreamOpen(true);
      return;
    }
  }
  useLandContext.getState().setSection("actions");
  useUi.getState().setPanel("land");
}
