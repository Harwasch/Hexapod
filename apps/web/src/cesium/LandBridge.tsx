import { useEffect } from "react";
import type { Footprint } from "@twin/contracts";

import { useLand } from "@/state/land";
import { useUi } from "@/state/ui";

import { LandMapController } from "./LandMapController";
import { useScene } from "./SceneContext";

export function LandBridge() {
  const scene = useScene();
  useEffect(() => {
    if (!scene) return;
    const controller = new LandMapController(
      scene,
      (point) => useLand.getState().addPoint(point),
      (boundary) => useLand.getState().updateBoundary(boundary),
    );
    let owningPointer = false;
    let shownBoundary: Footprint | null | undefined;
    let shownPoints: ReturnType<typeof useLand.getState>["points"] | undefined;
    let shownEditing = false;
    const sync = () => {
      const state = useLand.getState();
      const ui = useUi.getState();
      const mode =
        ui.activePanel === "land" && !ui.measureMode && !ui.exploreMode ? state.mode : "browse";
      const owns = mode !== "browse";
      if (owns !== owningPointer) {
        owningPointer = owns;
        scene.setInteractionMode(
          owns ? "land" : ui.measureMode ? "measure" : ui.exploreMode ? "explore" : "select",
        );
      }
      controller.setMode(mode);
      controller.setSnapping(state.snapEnabled);
      const boundary = state.draft?.boundary ?? state.active?.boundary ?? null;
      const editing = mode === "edit";
      if (boundary !== shownBoundary || state.points !== shownPoints || editing !== shownEditing) {
        shownBoundary = boundary;
        shownPoints = state.points;
        shownEditing = editing;
        controller.show(boundary, state.points, editing);
      }
    };
    const offLand = useLand.subscribe(sync);
    const offUi = useUi.subscribe(sync);
    sync();
    return () => {
      offLand();
      offUi();
      controller.destroy();
      if (owningPointer && !scene.isDestroyed) scene.setInteractionMode("select");
    };
  }, [scene]);
  return null;
}
