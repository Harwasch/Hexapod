import { useCallback } from "react";

import { useScene } from "@/cesium/SceneContext";
import type { Machine, Plan } from "@/missions/types";
import { useMission } from "@/state/mission";

/** Imperative mission actions shared by markers, cards, tables and the command bar. */
export function useMissionActions() {
  const scene = useScene();

  const selectMachine = useCallback(
    (id: string, options: { fly?: boolean } = {}) => {
      useMission.getState().select({ kind: "machine", id });
      useMission.getState().setView("map");
      scene?.mission.setSelectedZone(null);
      if (options.fly) scene?.mission.flyToMachine(id);
    },
    [scene],
  );

  const selectZone = useCallback(
    (id: string, options: { fly?: boolean } = {}) => {
      useMission.getState().select({ kind: "zone", id });
      useMission.getState().setView("map");
      scene?.mission.setSelectedZone(id);
      if (options.fly) scene?.mission.flyToZone(id);
    },
    [scene],
  );

  const clearSelection = useCallback(() => {
    useMission.getState().select(null);
    scene?.mission.setSelectedZone(null);
  }, [scene]);

  const runMachineAction = useCallback((machine: Machine) => {
    const state = useMission.getState();
    state.appendLog("you", `${machine.primaryAction} ${machine.id}`);
    state.appendLog(
      "agent",
      `${machine.primaryAction} queued for ${machine.name}. This fleet is simulated, so nothing moves yet; a robot bridge will pick this up here.`,
    );
    state.setStreamOpen(true);
  }, []);

  const showPlanOnMap = useCallback(
    (plan: Plan) => {
      const state = useMission.getState();
      state.setView("map");
      const first = plan.zoneIds[0];
      if (first) {
        state.select({ kind: "zone", id: first });
        scene?.mission.setSelectedZone(first);
        scene?.mission.flyToZone(first);
      } else {
        scene?.mission.flyToProject();
      }
    },
    [scene],
  );

  const flyToProject = useCallback(() => scene?.mission.flyToProject(), [scene]);

  return {
    selectMachine,
    selectZone,
    clearSelection,
    runMachineAction,
    showPlanOnMap,
    flyToProject,
  };
}
