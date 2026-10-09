import { useQuery } from "@tanstack/react-query";
import type { LandArea } from "@twin/contracts";
import { api, unwrap } from "@/api/client";
import { useLandScope, useLandAccessReady } from "@/state/landIdentity";

export function useInvestigations(landId: string) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: ["land-research", scope, "list", landId],
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/investigations", { params: { path: { land_id: landId } } }),
      ),
    enabled: ready,
    retry: false,
  });
}

export function useInvestigation(id: string | null, offset = 0) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: ["land-research", scope, "investigation", id, offset],
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/research/investigations/{investigation_id}", {
          params: { path: { investigation_id: id ?? "" }, query: { offset, limit: 100 } },
        }),
      ),
    enabled: Boolean(id) && ready,
    retry: false,
    refetchInterval: (query) =>
      query.state.data?.runs.some((run) => run.status === "queued" || run.status === "running")
        ? 1500
        : false,
  });
}

export function useResearchStatus() {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: ["land-research", scope, "status"],
    queryFn: () => unwrap(api.GET("/api/v1/research/status")),
    enabled: ready,
    staleTime: 60_000,
    retry: false,
  });
}

export async function beginInvestigation(land: LandArea, question: string) {
  return unwrap(
    api.POST("/api/v1/land/{land_id}/investigations", {
      params: { path: { land_id: land.id } },
      body: { title: question.slice(0, 300), question, boundaryRevision: land.revision },
    }),
  );
}

export function startResearch(
  id: string,
  question: string,
  kind: "overview" | "investigation",
  requestKey: string,
) {
  return unwrap(
    api.POST("/api/v1/research/investigations/{investigation_id}/runs", {
      params: { path: { investigation_id: id } },
      body: { question, kind, requestKey },
    }),
  );
}
export const cancelResearch = (id: string) =>
  unwrap(api.POST("/api/v1/research/runs/{run_id}/cancel", { params: { path: { run_id: id } } }));
export const setFindingDisposition = (
  id: string,
  disposition: "visible" | "pinned" | "dismissed",
) =>
  unwrap(
    api.PUT("/api/v1/research/findings/{finding_id}", {
      params: { path: { finding_id: id } },
      body: { disposition },
    }),
  );
