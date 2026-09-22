/**
 * The console's data layer.
 *
 * Deliberately not `@/api/queries.ts`. That module is the globe app's, and everything in
 * it is wrapped in `withFallback()` — an offline catalog, a built-in demo site, a
 * read-only world that still renders when the API is gone. None of that belongs here: a
 * data console with no API has nothing true to show, and inventing a demo row in a table
 * about what is really in storage would be the one lie this page cannot afford.
 *
 * What it does share is the client — the same `openapi-fetch` instance, the same typed
 * paths, the same write-token middleware — so the token entered here is the token the
 * globe app uses and a contract change breaks both at compile time.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import type {
  ArtifactRow,
  Capture,
  Job,
  JobCreate,
  PipelineCatalogue,
  Site,
  SiteSummary,
  StorageReconciliation,
} from "@twin/contracts";

import { api, unwrap } from "@/api/client";

export const adminKeys = {
  captures: ["admin", "captures"] as const,
  jobs: ["admin", "jobs"] as const,
  artifacts: ["admin", "artifacts"] as const,
  reconciliation: ["admin", "reconciliation"] as const,
  recipes: ["admin", "recipes"] as const,
  sites: ["admin", "sites"] as const,
};

/**
 * How often the tables re-ask while the page is open.
 *
 * Slower than the globe app's 4 s job poll on purpose: that panel is watching one upload
 * it started, this page is a browser over everything, and a full-table refetch every four
 * seconds would fight the person reading it. A run launched here invalidates immediately,
 * so the row appears at once rather than on the next tick.
 */
const POLL_MS = 10_000;

export function useCaptures() {
  return useQuery({
    queryKey: adminKeys.captures,
    queryFn: () =>
      unwrap<Capture[]>(api.GET("/api/v1/captures", { params: { query: { limit: 200 } } })),
    refetchInterval: POLL_MS,
  });
}

export function useJobs() {
  return useQuery({
    queryKey: adminKeys.jobs,
    queryFn: () => unwrap<Job[]>(api.GET("/api/v1/jobs", { params: { query: { limit: 200 } } })),
    refetchInterval: POLL_MS,
  });
}

export function useArtifacts() {
  return useQuery({
    queryKey: adminKeys.artifacts,
    queryFn: () =>
      unwrap<ArtifactRow[]>(api.GET("/api/v1/artifacts", { params: { query: { limit: 500 } } })),
    refetchInterval: POLL_MS,
  });
}

/** Sites, only so a capture row can name the site it became. */
export function useSites() {
  return useQuery({
    queryKey: adminKeys.sites,
    queryFn: () => unwrap<SiteSummary[]>(api.GET("/api/v1/sites")),
    staleTime: 60_000,
  });
}

/**
 * Storage against the database, in both directions.
 *
 * Not polled: it walks a bucket. It is fetched when the Captures view opens and when
 * somebody asks for it again, which is the honest cost of the question.
 */
export function useReconciliation(enabled: boolean) {
  return useQuery({
    queryKey: adminKeys.reconciliation,
    queryFn: () => unwrap<StorageReconciliation>(api.GET("/api/v1/storage/reconciliation", {})),
    enabled,
    staleTime: Infinity,
    retry: false,
  });
}

/**
 * The recipes and providers a run can use.
 *
 * `retry: false` because the honest failure here is a 404 — a deployment whose API image
 * has no `tools/pipeline` (infra/api.Dockerfile copies `apps/api` and nothing else). That
 * is a fact to show, not a request to repeat.
 */
export function useRecipes() {
  return useQuery({
    queryKey: adminKeys.recipes,
    queryFn: () => unwrap<PipelineCatalogue>(api.GET("/api/v1/recipes", {})),
    staleTime: 5 * 60_000,
    retry: false,
  });
}

/** Queue a run. The parameter overrides ride on `params`, keyed by stage id. */
export function useLaunchRun() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ captureId, body }: { captureId: string; body: JobCreate }) =>
      unwrap<Job>(
        api.POST("/api/v1/captures/{capture_id}/process", {
          params: { path: { capture_id: captureId } },
          body,
        }),
      ),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: adminKeys.jobs });
      void client.invalidateQueries({ queryKey: adminKeys.captures });
    },
  });
}

export function useCancelRun() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) =>
      unwrap<Job>(
        api.POST("/api/v1/jobs/{job_id}/cancel", { params: { path: { job_id: jobId } } }),
      ),
    onSuccess: () => void client.invalidateQueries({ queryKey: adminKeys.jobs }),
  });
}

export function useRetryRun() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) =>
      unwrap<Job>(
        api.POST("/api/v1/jobs/{job_id}/retry", {
          params: { path: { job_id: jobId } },
          body: { fromStage: null },
        }),
      ),
    onSuccess: () => void client.invalidateQueries({ queryKey: adminKeys.jobs }),
  });
}

/** One step's log, fetched only when its drawer is open. */
export function useStepLog(jobId: string | null, stepId: string | null) {
  return useQuery({
    queryKey: ["admin", "log", jobId, stepId] as const,
    queryFn: () =>
      unwrap<{ text: string; logKey: string }>(
        api.GET("/api/v1/jobs/{job_id}/steps/{step_id}/log", {
          params: { path: { job_id: jobId ?? "", step_id: stepId ?? "" } },
        }),
      ),
    enabled: jobId !== null && stepId !== null,
    retry: false,
  });
}

/** Sites by id, for "which site did this capture become". */
export function siteIndex(sites: SiteSummary[] | Site[] | undefined): Record<string, string> {
  const index: Record<string, string> = {};
  for (const site of sites ?? []) index[site.id] = site.name;
  return index;
}
