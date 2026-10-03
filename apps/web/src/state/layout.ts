import { create } from "zustand";

import { useMission } from "./mission";
import { objectSelected, useSceneSelect } from "./sceneSelect";
import { useSettings } from "./settings";
import { useUi } from "./ui";

/**
 * The HUD's two docks, and which one was opened last.
 *
 * The screen is split into fixed regions (see `.hud` in `app.css`), so two surfaces can
 * only collide if two of them want the same region at once. Two rules settle that:
 *
 * 1. **One side panel at a time.** A tool panel (Layers, Measure, Add) opens in the left
 *    dock and the Plan / Fleet drawer on the right; opening one closes the other, so the
 *    map between them keeps its width. Picking Plan closes the open tool panel, and opening
 *    a tool panel returns the view to Map. What is selected on the map (the right dock)
 *    stays beside the drawer: picking a Fleet row selects the machine without closing it.
 * 2. **On a phone there is one sheet.** The left dock, the drawer and the right dock all
 *    become the same bottom sheet, so the one touched last wins and the other waits behind
 *    it. `focus` is that choice ("left" is a tool panel or the drawer); the CSS reads it
 *    from `data-sheet` on `.hud` and only acts on it at compact widths.
 */
export type Dock = "left" | "right";

interface LayoutState {
  focus: Dock | null;
  setFocus: (focus: Dock | null) => void;
}

export const useLayout = create<LayoutState>()((set) => ({
  focus: null,
  setFocus: (focus) => set({ focus }),
}));

/**
 * Anything showing in the right dock: a selection (a machine, a zone, a scan object or the
 * brush), the inspector, feeds, dev tools.
 */
export function rightDockBusy(): boolean {
  const mission = useMission.getState();
  return (
    mission.selection !== null ||
    objectSelected(useSceneSelect.getState()) ||
    mission.feedsOpen ||
    useUi.getState().inspectorOpen ||
    useSettings.getState().devToolsOpen
  );
}

/** Anything showing on the left of the sheet rule: a tool panel or the Plan / Fleet drawer. */
export function leftDockBusy(): boolean {
  return useUi.getState().activePanel !== null || useMission.getState().view !== "map";
}

let bound = false;

/** Wires the two rules above. Idempotent; returns an unbinder for tests and HMR. */
export function bindDockRules(): () => void {
  if (bound) return () => undefined;
  bound = true;
  const settle = () => {
    const { focus, setFocus } = useLayout.getState();
    // Closing the focused dock hands the sheet back to the other one if it has content.
    if (focus === "left" && !leftDockBusy()) setFocus(rightDockBusy() ? "right" : null);
    if (focus === "right" && !rightDockBusy()) setFocus(leftDockBusy() ? "left" : null);
  };
  const offUi = useUi.subscribe((state, prev) => {
    if (state.activePanel && state.activePanel !== prev.activePanel) {
      if (useMission.getState().view !== "map") useMission.getState().setView("map");
      useLayout.getState().setFocus("left");
    }
    if (state.inspectorOpen && !prev.inspectorOpen) useLayout.getState().setFocus("right");
    settle();
  });
  const offMission = useMission.subscribe((state, prev) => {
    if (state.view !== "map" && state.view !== prev.view) {
      if (useUi.getState().activePanel) useUi.getState().setPanel(null);
      useLayout.getState().setFocus("left");
    }
    if (
      (state.selection && state.selection !== prev.selection) ||
      (state.feedsOpen && !prev.feedsOpen)
    )
      useLayout.getState().setFocus("right");
    settle();
  });
  const offSettings = useSettings.subscribe((state, prev) => {
    if (state.devToolsOpen && !prev.devToolsOpen) useLayout.getState().setFocus("right");
    settle();
  });
  // An object picked in the scene, or the brush taken up, brings its card to the front.
  const offObject = useSceneSelect.subscribe((state, prev) => {
    const shown = objectSelected(state);
    if (shown && (!objectSelected(prev) || state.candidates !== prev.candidates))
      useLayout.getState().setFocus("right");
    settle();
  });
  return () => {
    offUi();
    offMission();
    offSettings();
    offObject();
    bound = false;
  };
}
