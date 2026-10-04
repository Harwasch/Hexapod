import { useMission } from "@/state/mission";
import { selectedId, useSceneSelect } from "@/state/sceneSelect";
import { useUi } from "@/state/ui";

/** What the scene does for a step back: the parts of `CesiumSceneManager` it touches. */
export interface StepBackScene {
  /** The brush away, else the scan object cleared (`SceneSelectController.escape`). */
  sceneSelect: { escape(): boolean };
  mission: { setSelectedZone(zoneId: string | null): void };
  selection: { clear(): void };
}

/** Which step Escape took. */
export type StepBack =
  | "shortcuts"
  | "more"
  | "activity"
  | "write-token"
  | "brush"
  | "measure"
  | "switcher"
  | "object"
  | "feeds"
  | "composer"
  | "selection"
  | "view"
  | "inspector"
  | "panel";

/**
 * Escape, once: the first of these that applies, and nothing else.
 *
 * 1. What floats over the HUD: the shortcut sheet, the phone's More sheet, the agent's
 *    activity log, the write-token prompt.
 * 2. A mode that holds the pointer: the brush (it holds the camera too), then measuring.
 * 3. The site switcher, a popover too.
 * 4. What is selected, the latest thing first: an object of a scan; a machine's camera feeds,
 *    the plan composer, then the machine or zone.
 * 5. Then the panels: the Plan / Fleet drawer back to the map, the inspector, the tool panel.
 *
 * Dialogs, popovers and tooltips (Radix) close themselves on Escape and mark the key as
 * handled, and `useHotkey` then leaves it alone, so they come before all of this. Returns what
 * it did, or null when there was nothing to step back from.
 */
export function stepBack(scene: StepBackScene | null): StepBack | null {
  const ui = useUi.getState();
  const mission = useMission.getState();
  const objects = useSceneSelect.getState();
  if (ui.shortcutsOpen) {
    ui.setShortcutsOpen(false);
    return "shortcuts";
  }
  if (ui.moreOpen) {
    ui.setMoreOpen(false);
    return "more";
  }
  if (ui.activityOpen) {
    ui.setActivityOpen(false);
    return "activity";
  }
  if (ui.writeTokenPrompt) {
    ui.setWriteTokenPrompt(false);
    return "write-token";
  }
  if (objects.mode === "paint") {
    if (!scene?.sceneSelect.escape()) objects.setMode("pick");
    return "brush";
  }
  if (ui.measureMode) {
    ui.setMeasureMode(null);
    return "measure";
  }
  if (mission.projectsOpen) {
    mission.setProjectsOpen(false);
    return "switcher";
  }
  if (selectedId(objects) !== null) {
    if (!scene?.sceneSelect.escape()) objects.clear();
    return "object";
  }
  if (mission.feedsOpen) {
    mission.setFeedsOpen(false);
    return "feeds";
  }
  if (mission.composer && mission.composer.status !== "drafting") {
    mission.closeComposer();
    return "composer";
  }
  if (mission.selection) {
    mission.select(null);
    scene?.mission.setSelectedZone(null);
    return "selection";
  }
  if (mission.view !== "map") {
    mission.setView("map");
    return "view";
  }
  if (ui.inspectorOpen) {
    ui.setInspectorOpen(false);
    scene?.selection.clear();
    return "inspector";
  }
  if (ui.activePanel) {
    ui.setPanel(null);
    return "panel";
  }
  return null;
}
