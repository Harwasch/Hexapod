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

import type { LandArea, PlanRecord, SiteSummary } from "@twin/contracts";

import { api, unwrap } from "@/api/client";
import { fetchLandGround, queryKeys } from "@/api/queries";
import { siteDisplayName } from "@/features/sites/siteNames";
import { describeError } from "@/lib/log";
import { DEMO_PROJECT } from "@/missions/demo";
import {
  boundaryZoneOf,
  groundModeFor,
  landIdOfProject,
  landProject,
  landProjectId,
  landZones,
} from "@/missions/landGround";
import { siteProjectId } from "@/missions/siteProject";
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
  options: {
    layerOnly?: boolean;
    /** Called when a switch that waited for an open plan card happens (to frame the land). */
    onWaitedSwitch?: () => void;
  } = {},
): Promise<"layer" | "project" | null> {
  const ground = await fetchLandGround(land.id, land.revision);
  if (useLand.getState().active?.id !== land.id) return null;
  const zones = landZones(ground);
  const store = useMission.getState();
  const own = landIdOfProject(store.project?.id) === land.id;
  if (!own && groundModeFor(store.project, land) === "layer") {
    store.engageLand({ landId: land.id, zones });
    if (useMission.getState().landGround?.landId === land.id) return "layer";
    // A plan card is open with another Land layered here: its ground leaves with the switch,
    // so the switch waits until the card is approved or closed.
    waitForCard(land, options.onWaitedSwitch);
    return null;
  }
  if (options.layerOnly && !own) return null;
  store.engageLand({ landId: land.id, zones }, landProject(land, ground));
  if (useMission.getState().landGround?.landId === land.id) return "project";
  // A plan card is open on another Land's project: its ground cannot take this Land's, so
  // the switch waits until the card is approved or closed.
  waitForCard(land, options.onWaitedSwitch);
  return null;
}

/** The Land whose switch waits for the open plan card (its latest record), and how to stop. */
let waiting: { land: LandArea; onSwitch?: () => void; stop: () => void } | null = null;

function waitForCard(land: LandArea, onSwitch?: () => void): void {
  if (waiting?.land.id === land.id) {
    // Selected again, or its boundary revised meanwhile: the switch takes the latest record.
    waiting = { ...waiting, land, onSwitch };
    return;
  }
  waiting?.stop();
  const card = useMission.getState().project?.name ?? "the current project";
  useMission
    .getState()
    .appendLog(
      "agent",
      `${land.name} opens in Plan once you approve or close the plan card on ${card}.`,
    );
  const stop = useMission.subscribe((state) => {
    if (state.composer || !waiting) return;
    const { land: latest, onSwitch: then } = waiting;
    stop();
    waiting = null;
    // Only for the Land still selected; a reload elsewhere never moves the operator.
    if (useLand.getState().active?.id !== latest.id) return;
    engageLand(latest, { onWaitedSwitch: then })
      .then((mode) => {
        if (mode === "project") then?.();
      })
      .catch((error: unknown) => {
        console.warn("Land ground could not be loaded", error);
      });
  });
  waiting = { land, onSwitch, stop };
}

/**
 * A mission project's name for the copy, never its id: the project in view, the simulated
 * demo or a catalog site the browser has listed. Null when it is none of those (another
 * Land's plans, or a site not listed here), so the copy says "another mission project".
 */
export function missionProjectName(
  projectId: string,
  sites: readonly SiteSummary[] | undefined,
): string | null {
  const current = useMission.getState().project;
  if (current?.id === projectId) return current.name;
  if (landIdOfProject(projectId) !== null) return null;
  if (projectId === DEMO_PROJECT.id) return DEMO_PROJECT.name;
  const site = sites?.find((candidate) => siteProjectId(candidate.id) === projectId);
  return site ? siteDisplayName(site) : null;
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
    const name = missionProjectName(
      record.projectId,
      client.getQueryData<SiteSummary[]>(queryKeys.sites),
    );
    store.appendLog(
      "agent",
      name
        ? `“${record.title}” is filed under the mission project ${name}. Switch to ${name} to see it in Plan.`
        : `“${record.title}” is filed under another mission project. Switch to that project to see it in Plan.`,
    );
    store.setStreamOpen(true);
    return;
  }
  await client.invalidateQueries({ queryKey: ["workspace-plans"] });
  useMission.getState().openPlan(record.id);
}

/**
 * "Revise the action in Land": Land opens on the Land's Plans lens, where the action is a
 * plan. The plan's Land is selected first if another one is. The lens store is loaded on
 * demand, with the Land surface, so this bundle does not carry it.
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
  const { useLens } = await import("@/features/land/surface/lenses");
  useLens.getState().setLens("plans");
}
