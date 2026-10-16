import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { components, LandArea } from "@twin/contracts";
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

export function useInvestigation(id: string | null, offset = 0, limit = 100, pollMs = 1500) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: ["land-research", scope, "investigation", id, offset, limit],
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/research/investigations/{investigation_id}", {
          params: { path: { investigation_id: id ?? "" }, query: { offset, limit } },
        }),
      ),
    enabled: Boolean(id) && ready,
    retry: false,
    refetchInterval: (query) =>
      query.state.data?.runs.some((run) => run.status === "queued" || run.status === "running")
        ? pollMs
        : false,
  });
}

/**
 * What research can do on this server: the AI model, whether a research worker has checked
 * in lately, and what waits in this workspace's queue. `pollWhileWaiting` refreshes it every
 * 10 s while a run sits queued, so "none has checked in for 14 min" keeps counting and the
 * notice clears when a worker starts; one query key, so every caller shares the polling.
 */
export function useResearchStatus(pollWhileWaiting = false) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: ["land-research", scope, "status"],
    queryFn: () => unwrap(api.GET("/api/v1/research/status")),
    enabled: ready,
    staleTime: 60_000,
    retry: false,
    refetchInterval: pollWhileWaiting ? 10_000 : false,
  });
}

export type PlaceFact = components["schemas"]["PlaceFact"];
export type PlaceFactKind = PlaceFact["kind"];

/**
 * One public fact about a point (elevation, soil, land cover, flood zone, ecoregion), looked
 * up live by the API and never saved. Keyed to about a metre, like the server's cache, and
 * kept for the server's 6 h; a failure is shown, not retried, until the person asks.
 */
export function usePlaceFact(
  kind: PlaceFactKind,
  at: { longitude: number; latitude: number } | null,
  enabled: boolean,
) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  const client = useQueryClient();
  const queryKey = ["place-fact", scope, kind, at?.longitude.toFixed(5), at?.latitude.toFixed(5)];
  return useQuery({
    queryKey,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/research/place/{fact}", {
          params: {
            path: { fact: kind },
            query: {
              longitude: at?.longitude ?? 0,
              latitude: at?.latitude ?? 0,
              // A refetch after "unavailable" is the Retry button: past the server's cached
              // failure, or Retry would only repeat it.
              retry: client.getQueryData<PlaceFact>(queryKey)?.status === "unavailable",
            },
          },
        }),
      ),
    enabled: enabled && ready && at !== null,
    staleTime: 6 * 3600_000,
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
  budget?: components["schemas"]["ResearchBudget"],
  focus?: components["schemas"]["ResearchFocusReference"] | null,
) {
  return unwrap(
    api.POST("/api/v1/research/investigations/{investigation_id}/runs", {
      params: { path: { investigation_id: id } },
      body: { question, kind, requestKey, budget, focus },
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
