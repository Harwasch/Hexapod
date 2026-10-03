import { useMission } from "./mission";
import { objectSelected, useSceneSelect } from "./sceneSelect";
import { useSelection } from "./selection";
import { useUi } from "./ui";

/**
 * What the scene does when one kind of selection gives way to another: the map's own marks
 * of each (a zone's outline, the scan object's highlight and brush, the inspector's marker).
 */
export interface SelectionScene {
  /** Takes the zone's outline off the map (`MissionManager.setSelectedZone(null)`). */
  clearZone(): void;
  /** Puts the brush away and clears the scan object (`SceneSelectController`). */
  clearObject(): void;
  /** Takes the inspector's marker and highlight off the map (`SelectionManager.clear`). */
  clearInspector(): void;
}

/** Without a scene: the stores alone. */
const STORES_ONLY: SelectionScene = {
  clearZone: () => undefined,
  clearObject: () => {
    const store = useSceneSelect.getState();
    store.setMode("pick");
    store.clear();
  },
  clearInspector: () => undefined,
};

/**
 * One selection at a time. The HUD shows what is selected in one card (`SelectionCard`): a
 * machine or a zone (`state/mission.ts`), or an object of a scan picked in the scene, or the
 * brush out to paint one (`state/sceneSelect.ts`); the Location and site card (the
 * inspector) shares the right dock with it. A new selection of one kind replaces the other,
 * so two cards never show for two things picked one after the other:
 *
 * - an object chosen (a click, the brush, the objects panel), or the brush taken up, clears a
 *   machine or zone and closes the inspector;
 * - a machine or zone chosen (a marker, the Fleet table, the command box) clears the object
 *   and puts the brush away;
 * - a new place in the inspector (a click on the ground or a site) clears the object.
 *
 * Wired once by SceneBridge with the scene's side of each; returns the unbinder.
 */
export function bindOneSelection(scene: SelectionScene = STORES_ONLY): () => void {
  const offObject = useSceneSelect.subscribe((state, previous) => {
    const started =
      (state.mode === "paint" && previous.mode !== "paint") ||
      (objectSelected(state) &&
        (state.candidates[state.index] !== previous.candidates[previous.index] ||
          state.assetId !== previous.assetId));
    if (!started) return;
    if (useMission.getState().selection) {
      useMission.getState().select(null);
      scene.clearZone();
    }
    if (useUi.getState().inspectorOpen || useSelection.getState().selection) {
      useUi.getState().setInspectorOpen(false);
      useSelection.getState().setSelection(null);
      scene.clearInspector();
    }
  });
  const offMission = useMission.subscribe((state, previous) => {
    if (!state.selection || state.selection === previous.selection) return;
    if (objectSelected(useSceneSelect.getState())) scene.clearObject();
  });
  const offInspector = useSelection.subscribe((state, previous) => {
    // A new place, not the same one refined (its terrain height arriving).
    if (!state.selection || state.selection.at === previous.selection?.at) return;
    if (objectSelected(useSceneSelect.getState())) scene.clearObject();
  });
  return () => {
    offObject();
    offMission();
    offInspector();
  };
}
