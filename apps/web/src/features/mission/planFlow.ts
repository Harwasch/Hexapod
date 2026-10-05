/**
 * Planning from the bar. One sentence of work ("3D scan this field", "mow the orchard by
 * Friday") becomes a plan on the map: the ground is found first (named zones, the mapped
 * feature the goal talks about, or the spot the operator clicks), then the agent drafts with
 * sensible defaults and the card shows the result to approve, drag or talk to. The map menu's
 * "Plan here" starts from the other end: the spot first, the sentence after.
 */

import { centerOf, destination, type LonLat } from "@twin/geo";

import { outlineGround } from "@/api/queries";
import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { isAreaZone, nextAreaId, viewAreaZone } from "@/missions/areas";
import { resolveGround } from "@/missions/ground";
import { fetchOsmAreas, kindForGoal, OSM_ATTRIBUTION, zoneFromCandidate } from "@/missions/osm";
import type { Zone } from "@/missions/types";
import { useMission } from "@/state/mission";
import { useViewer } from "@/state/viewer";

import { startPlanDraft } from "./planDrafting";

/** Strips "plan:" and "draft a plan to" so the goal reads as the operator said it. */
export function goalFromText(text: string): string {
  return text
    .replace(
      /^(?:(?:draft|create|make|write|start|new|plan)\s+(?:me\s+)?(?:a\s+)?(?:new\s+)?(?:mission\s+)?(?:plan|mission)\b(?:\s+(?:to|for|that|:))?\s*|plan:\s*)/i,
      "",
    )
    .trim();
}

function mentionedZones(text: string, zones: Zone[]): string[] {
  const ids = new Set((text.match(/\b[AZ]-\d{2}\b/gi) ?? []).map((id) => id.toUpperCase()));
  return zones.filter((z) => ids.has(z.id)).map((z) => z.id);
}

function nearestToView(candidates: { footprint: Zone["footprint"] }[], center: LonLat) {
  let best: (typeof candidates)[number] | null = null;
  let bestD = Infinity;
  for (const c of candidates) {
    const p = centerOf(c.footprint);
    const d = (p.longitude - center.longitude) ** 2 + (p.latitude - center.latitude) ** 2;
    if (d < bestD) {
      bestD = d;
      best = c;
    }
  }
  return best;
}

/** The mapped feature the goal names ("the lake"), nearest the view centre, or null. */
async function groundFromWords(
  goal: string,
  scene: CesiumSceneManager,
): Promise<{ zone: Zone; note: string } | null> {
  const kind = kindForGoal(goal);
  const center = scene.camera.viewCenter();
  if (!kind || !center) return null;
  const mpp = useViewer.getState().camera.metersPerPixel;
  const halfW = Math.min(5000, Math.max(60, (mpp * center.width) / 2));
  const halfH = Math.min(5000, Math.max(60, (mpp * center.height) / 2));
  const north = destination(center, 0, halfH).latitude;
  const south = destination(center, 180, halfH).latitude;
  const east = destination(center, 90, halfW).longitude;
  const west = destination(center, 270, halfW).longitude;
  try {
    const candidates = await fetchOsmAreas({ west, south, east, north }, kind);
    const pick = nearestToView(candidates, center);
    if (!pick) return null;
    const id = nextAreaId(useMission.getState().project?.zones ?? []);
    const candidate = candidates.find((c) => c.footprint === pick.footprint);
    if (!candidate) return null;
    return {
      zone: zoneFromCandidate(id, candidate),
      note: `${candidate.name}, ${Math.round(candidate.acres).toLocaleString()} ac, outline from OpenStreetMap.`,
    };
  } catch {
    return null;
  }
}

/** Adds the area to the project and the card's scope. */
function adopt(zone: Zone): void {
  const state = useMission.getState();
  if (!state.project) return;
  state.addArea(state.project.id, zone);
  const composer = useMission.getState().composer;
  if (composer && !composer.zoneIds.includes(zone.id))
    state.updateComposer({ zoneIds: [...composer.zoneIds, zone.id] });
}

/**
 * Plans from one sentence. Returns the agent's first reply for the stream; the rest of the
 * conversation (ground found, draft ready) is logged as it happens.
 */
export async function planFromText(
  text: string,
  scene: CesiumSceneManager | null,
): Promise<string> {
  const state = useMission.getState();
  const project = state.project;
  if (!project) return "The world is still starting; try again in a moment.";
  const goal = goalFromText(text);
  if (goal.length < 3) {
    if (!state.composer) state.openComposer();
    return "Tell me what to do and where, in one sentence.";
  }
  const composer = state.composer;
  // Ground already in scope: zones named in the goal, else what the open card already has.
  let zoneIds = mentionedZones(goal, project.zones);
  if (zoneIds.length === 0 && composer?.zoneIds.length) zoneIds = composer.zoneIds;
  if (!composer) state.openComposer({ goal, zoneIds });
  else if (composer.replacePlanId === null)
    state.updateComposer({ goal, zoneIds, status: "idle", error: null });
  if (zoneIds.length > 0) {
    void startPlanDraft(goal, { zoneIds });
    return "Drafting.";
  }
  // The goal names a kind of ground ("the lake"): take the mapped one in view. "This field"
  // is pointing, not naming, so it waits for the click below.
  const pointing = /\b(this|that|here|there)\b/i.test(goal);
  if (scene && !pointing) {
    const found = await groundFromWords(goal, scene);
    if (found) {
      adopt(found.zone);
      useMission.getState().updateComposer({ ground: { source: "osm", note: found.note } });
      void startPlanDraft(goal, { zoneIds: [found.zone.id] });
      return `I took ${found.note} Drag its corners if it is not quite right.`;
    }
  }
  // A site with its own zones: the planner picks among them.
  if (project.zones.some((z) => !isAreaZone(z))) {
    void startPlanDraft(goal, { zoneIds: [] });
    return "Drafting.";
  }
  if (!scene) return "The world is still starting; try again in a moment.";
  // Otherwise: the operator points at the ground and the agent outlines it.
  useMission.getState().updateComposer({ status: "awaiting-ground" });
  void waitForGround(goal, scene);
  return "Click the ground you mean on the map and I'll outline it. Esc to cancel.";
}

/** Waits for one click, outlines the ground under it, then drafts. */
export async function waitForGround(goal: string, scene: CesiumSceneManager): Promise<void> {
  // Closing the card while the map waits for a click ends the wait.
  const unsubscribe = useMission.subscribe((s) => {
    if (!s.composer) scene.areas.cancelPick();
  });
  const picked = await scene.areas.pickGround();
  unsubscribe();
  const state = useMission.getState();
  if (!picked || !state.composer || !state.project) {
    if (state.composer?.status === "awaiting-ground") state.updateComposer({ status: "idle" });
    return;
  }
  await outlineAt(picked, goal, scene);
}

/** A point on the map: where it is, and where it sits in the view (0–1 across and down). */
export interface GroundPoint {
  longitude: number;
  latitude: number;
  x: number;
  y: number;
}

/**
 * Outlines the ground at `point` (`missions/ground.ts`) into the open card's scope, then
 * drafts `goal`; without a goal yet the card keeps the ground and waits for one in the bar.
 */
export async function outlineAt(
  point: GroundPoint,
  goal: string,
  scene: CesiumSceneManager,
): Promise<void> {
  const state = useMission.getState();
  if (!state.composer || !state.project) return;
  state.updateComposer({ status: "locating" });
  state.appendLog("agent", "Looking at the ground there…");
  const id = nextAreaId(state.project.zones);
  const configured = state.plannerConfigured;
  const found = await resolveGround(
    { longitude: point.longitude, latitude: point.latitude },
    id,
    goal,
    {
      point: { x: point.x, y: point.y },
      snapshot: () => scene.snapshot(),
      unproject: (x, y) => scene.groundAt(x, y),
      outline: outlineGround,
      vision: configured,
    },
  );
  if (!useMission.getState().composer) return;
  adopt(found.zone);
  useMission.getState().updateComposer({ ground: { source: found.source, note: found.note } });
  useMission.getState().appendLog("agent", found.note);
  if (goal.trim().length >= 3) void startPlanDraft(goal, { zoneIds: [found.zone.id] });
  else useMission.getState().updateComposer({ status: "idle" });
}

/**
 * "Plan here" (the map menu): the plan composer, with the ground at the point outlined as its
 * ground; `point.x` and `point.y` are CSS px from the canvas's top left. An open card keeps its
 * goal and trades its ground for this one; a new card has no goal yet, so the draft waits for
 * the sentence in the bar (`planFromText` takes the card's ground).
 */
export function planHere(
  point: { longitude: number; latitude: number; x: number; y: number },
  scene: CesiumSceneManager,
): void {
  const state = useMission.getState();
  if (!state.project) return;
  // A click still awaited from "Click the ground" would outline a second place.
  scene.areas.cancelPick();
  if (state.composer) state.updateComposer({ zoneIds: [], ground: null, error: null });
  else state.openComposer();
  const canvas = scene.viewer.canvas;
  void outlineAt(
    {
      longitude: point.longitude,
      latitude: point.latitude,
      x: point.x / Math.max(1, canvas.clientWidth),
      y: point.y / Math.max(1, canvas.clientHeight),
    },
    useMission.getState().composer?.goal ?? "",
    scene,
  );
}

/** A rectangle of the ground in view, when the operator would rather not click. */
export function groundFromView(scene: CesiumSceneManager): Zone | null {
  const center = scene.camera.viewCenter();
  if (!center) return null;
  const id = nextAreaId(useMission.getState().project?.zones ?? []);
  return viewAreaZone(id, `View area ${id.slice(2)}`, {
    center: { longitude: center.longitude, latitude: center.latitude },
    metersPerPixel: useViewer.getState().camera.metersPerPixel,
    width: center.width,
    height: center.height,
    fraction: 0.5,
    dx: 0,
    dy: 0,
  });
}

export { OSM_ATTRIBUTION };
