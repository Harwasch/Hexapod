import { useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";

import type { PlanRecord } from "@twin/contracts";

import { usePlans, useWorkspacePlans } from "@/api/queries";
import { landIdOfProject } from "@/missions/landGround";
import { planFromRecord, zonesFromRecord } from "@/missions/planDraft";
import { useMission } from "@/state/mission";

import { plansInvalidate } from "./planDrafting";

/**
 * The project's stored plans into the store: the public ones and, when a workspace is open,
 * its private missions (plans on a saved Land, and missions Land actions scheduled), which the
 * public list never carries. The API is the source of truth whenever it answers; the ground
 * the plans carry joins the project's zones first, so their acres and overlays resolve.
 */
export function usePlansSync(): void {
  const projectId = useMission((s) => s.project?.id ?? null);
  const setRemotePlans = useMission((s) => s.setRemotePlans);
  const queryClient = useQueryClient();
  const remote = usePlans(projectId);
  const workspace = useWorkspacePlans(projectId);
  useEffect(() => {
    // Invalidation, not refetch(): it asks again only for the lists that are switched on, so
    // approving a public plan never calls the workspace route for someone without Land access.
    plansInvalidate.current = () => {
      void queryClient.invalidateQueries({ queryKey: ["plans"] });
      void queryClient.invalidateQueries({ queryKey: ["workspace-plans"] });
    };
    return () => {
      plansInvalidate.current = null;
    };
  }, [queryClient]);
  const publicData = remote.builtin || remote.isLoading ? null : remote.data;
  const privateData = workspace.data;
  // A Land's own project has only workspace plans: wait for them rather than wipe its list
  // with the empty public answer while they load or when the workspace cannot be reached.
  const landScoped = landIdOfProject(projectId) !== null;
  useEffect(() => {
    // The store looks the project up itself; only a change of project or of records matters.
    const current = useMission.getState().project;
    if (!current || !publicData) return;
    if (landScoped && !privateData) return;
    const byId = new Map<string, PlanRecord>();
    for (const record of [...publicData, ...(privateData ?? [])]) byId.set(record.id, record);
    const records = [...byId.values()];
    useMission.getState().mergeAreas(current.id, records.flatMap(zonesFromRecord));
    const composed = useMission.getState().project ?? current;
    setRemotePlans(
      current.id,
      records.map((record) => planFromRecord(record, composed)),
    );
  }, [projectId, publicData, privateData, landScoped, setRemotePlans]);
}
