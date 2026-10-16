import type { LandArea, LandCreate } from "@twin/contracts";

import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { useLand } from "@/state/land";
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
 * - An unsaved outline (`useLand.draft`) is never replaced: the tool opens on it instead.
 *
 * Selecting a land is picked up by the mission side too (LandMissionBridge), which may switch
 * the plan's project to that land. That is intended: a selected land is the ground plans use.
 */

export type OutlineResult = "started" | "has-draft";

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
 * opens the tool, when an unsaved outline exists.
 */
export function outlineLandAt(
  point: { longitude: number; latitude: number },
  scene: CesiumSceneManager | null,
): OutlineResult {
  if (useLand.getState().draft) {
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
  if (useLand.getState().draft) {
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
  if (useLand.getState().draft) {
    openTool();
    return "has-draft";
  }
  useLand.getState().select(land);
  useLandContext.getState().setSection(section);
  openTool();
  return "opened";
}
