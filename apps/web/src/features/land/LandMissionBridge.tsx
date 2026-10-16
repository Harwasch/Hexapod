import { useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";

import { engageLand } from "@/features/mission/landPlanning";
import { useLand } from "@/state/land";
import { useLandScope } from "@/state/landIdentity";
import { useMission } from "@/state/mission";

/**
 * Keeps the selected Land engaged as mission ground (`features/mission/landPlanning.ts`).
 *
 * A saved Land selected, or its boundary revised, loads its ground: layered onto the current
 * project when they overlap, else as the Land's own project. A change of project (flying to a
 * site) only layers the Land again when the new project is on it; it never takes the operator
 * back. An edit to the Land's inventory reloads the ground. With no Land selected the layered
 * zones leave. Renders nothing; failures go to the console, since the Land panel says what
 * went wrong with the Land itself.
 */
export function LandMissionBridge() {
  const active = useLand((s) => s.active);
  const mode = useLand((s) => s.mode);
  const drafting = useLand((s) => s.draft !== null);
  const projectId = useMission((s) => s.project?.id ?? null);
  const scope = useLandScope();
  const client = useQueryClient();
  const landId = active?.id ?? null;
  const revision = active?.revision ?? null;
  const browsing = mode === "browse" && !drafting;

  useEffect(() => {
    if (!landId) {
      useMission.getState().disengageLand();
      return;
    }
    const land = useLand.getState().active;
    if (!land || !browsing) return;
    engageLand(land).catch((error: unknown) => {
      console.warn("Land ground could not be loaded", error);
    });
  }, [landId, revision, browsing]);

  useEffect(() => {
    const land = useLand.getState().active;
    if (!land || !projectId || useMission.getState().landGround?.projectId === projectId) return;
    engageLand(land, { layerOnly: true }).catch((error: unknown) => {
      console.warn("Land ground could not be layered", error);
    });
  }, [projectId]);

  // An inventory edit (a fence drawn, a feature retired) changes the ground: load it again.
  useEffect(() => {
    if (!landId) return;
    return client.getQueryCache().subscribe((event) => {
      const { queryKey: key } = event.query as { queryKey: readonly unknown[] };
      if (
        event.type !== "updated" ||
        event.action.type !== "success" ||
        key[0] !== "land-inventory" ||
        key[1] !== scope ||
        key[2] !== landId
      )
        return;
      // Only where the Land is engaged now: a reload never moves the operator.
      const land = useLand.getState().active;
      const { landGround, project } = useMission.getState();
      if (land?.id !== landId || landGround?.landId !== landId) return;
      if (landGround.projectId !== project?.id) return;
      engageLand(land).catch((error: unknown) =>
        console.warn("Land ground could not be reloaded", error),
      );
    });
  }, [client, landId, scope]);

  return null;
}
