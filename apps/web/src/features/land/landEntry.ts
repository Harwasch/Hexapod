import type { LandArea, LandCreate } from "@twin/contracts";

import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { useLand, type LandMode, type LandPoint } from "@/state/land";
import { useLandContext, type LandWorkspaceSection } from "@/state/landContext";
import { useUi } from "@/state/ui";

/**
 * Ways into the Land tool from elsewhere on the map: the inspector's "What's here", the map's
 * right-click menu, and later the Descent arrival. They live here, not in each caller, because
 * entering the tool the wrong way loses work:
 *
 * - `LandPanel.save()` REVISES the active land when one is selected. A new outline started
 *   while a saved land is active would silently replace that land's boundary on save, so a
 *   new outline always clears the selection first (`useLand.clear()`).
 * - An unsaved outline is never replaced: the tool opens on it instead. That covers a proposed
 *   or imported one (`useLand.draft`) and one still being drawn (points placed in draw,
 *   corridor or split mode, with no draft yet). `clear()` and `select()` drop both, are not
 *   undoable, and LandDraftRecovery then deletes the browser copy too.
 *
 * Selecting a land is picked up by the mission side too (LandMissionBridge), which may switch
 * the plan's project to that land. That is intended: a selected land is the ground plans use.
 */

export type OutlineResult = "started" | "has-draft";

/**
 * Whether the Land tool holds an outline the user would lose: a draft, or a sketch with points
 * placed. The sketch test is the one LandDraftRecovery uses to decide what to keep in the
 * browser, so "unsaved" here means the same thing it does there.
 */
export function hasUnsavedOutline(s: {
  draft: LandCreate | null;
  mode: LandMode;
  points: readonly LandPoint[];
}): boolean {
  return (
    s.draft !== null ||
    ((s.mode === "draw" || s.mode === "corridor" || s.mode === "split") && s.points.length > 0)
  );
}

function openTool(): void {
  useUi.getState().setPanel("land");
}

/** Stop whatever the map is doing with the pointer, as LandPanel's own `begin` does. */
function freePointer(scene: CesiumSceneManager | null): void {
  const ui = useUi.getState();
  ui.setMeasureMode(null);
  ui.setExploreMode(false);
  scene?.areas.cancelPick();
  scene?.areas.edit(null);
}

/**
 * Open the Land tool finding mapped parcels, fields and lines at a point, the way "Find
 * parcels, lines or buildings" does after its first click. Returns "has-draft", and only
 * opens the tool, when an unsaved outline (a draft or a sketch in progress) exists.
 */
export function outlineLandAt(
  point: { longitude: number; latitude: number },
  scene: CesiumSceneManager | null,
): OutlineResult {
  if (hasUnsavedOutline(useLand.getState())) {
    openTool();
    return "has-draft";
  }
  freePointer(scene);
  useLand.getState().clear();
  openTool();
  useLand.getState().begin("candidates");
  useLand.getState().addPoint([point.longitude, point.latitude]);
  return "started";
}

/**
 * Propose an outline found elsewhere (a mapped feature, say) as a new, unsaved land. Use this
 * rather than `useLand.propose`, which keeps the active land and so would turn the next save
 * into a revision of it. Refuses, like `outlineLandAt`, to replace an unsaved outline.
 */
export function proposeOutline(draft: LandCreate, scene: CesiumSceneManager | null): OutlineResult {
  if (hasUnsavedOutline(useLand.getState())) {
    openTool();
    return "has-draft";
  }
  freePointer(scene);
  useLand.getState().clear();
  useLand.getState().propose(draft);
  openTool();
  return "started";
}

/**
 * Select a saved land and open its workspace on a section (Discover by default). Selecting
 * drops any unsaved outline, so with one open this only opens the tool on it ("has-draft").
 */
export function openLand(
  land: LandArea,
  section: LandWorkspaceSection = "discover",
): "opened" | "has-draft" {
  if (hasUnsavedOutline(useLand.getState())) {
    openTool();
    return "has-draft";
  }
  useLand.getState().select(land);
  useLandContext.getState().setSection(section);
  openTool();
  return "opened";
}
