import { useMutation, useQuery, useQueryClient, type UseQueryResult } from "@tanstack/react-query";

import type {
  AssetCreate,
  CameraBookmarkCreate,
  Capture,
  CaptureCreate,
  CaptureDetail,
  GroundOutline,
  HealthStatus,
  IonStatus,
  Job,
  JobCreate,
  JobRetry,
  JobStepLog,
  Layer,
  LayerCreate,
  OutlineRequest,
  PlanDraft,
  PlanDraftRequest,
  PlanRecord,
  PlanRecordCreate,
  PlanRecordRevise,
  PlanRecordStatus,
  PlannerStatus,
  Site,
  SiteCreate,
  SiteSummary,
} from "@twin/contracts";

import { api, isOffline, unwrap } from "./client";
import { builtinDemoSite, builtinLayers, toSummary } from "./fallback";

export const queryKeys = {
  health: ["health"] as const,
  sites: ["sites"] as const,
  site: (id: string) => ["sites", id] as const,
  layers: ["layers"] as const,
  ion: ["ion"] as const,
  captures: ["captures"] as const,
  capture: (id: string) => ["captures", id] as const,
  jobs: ["jobs"] as const,
  stepLog: (jobId: string, stepId: string) => ["jobs", jobId, "steps", stepId, "log"] as const,
};

/** Result decorated with whether the data came from the built-in fallback catalog. */
export interface CatalogResult<T> {
  data: T;
  builtin: boolean;
  isLoading: boolean;
  error: unknown;
  refetch: () => Promise<unknown>;
}

function withFallback<T>(query: UseQueryResult<T>, fallback: () => T): CatalogResult<T> {
  const offline = query.isError && isOffline(query.error);
  return {
    data: query.data ?? (offline ? fallback() : (query.data as T)),
    builtin: offline,
    isLoading: query.isPending,
    error: query.isError && !offline ? query.error : null,
    refetch: query.refetch,
  };
}

const RETRY = { retry: 1, retryDelay: 800, staleTime: 30_000 } as const;

export function useHealth() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: () => unwrap<HealthStatus>(api.GET("/api/v1/health")),
    ...RETRY,
    refetchInterval: 60_000,
  });
}

export function useSites(): CatalogResult<SiteSummary[]> {
  const query = useQuery({
    queryKey: queryKeys.sites,
    queryFn: () => unwrap<SiteSummary[]>(api.GET("/api/v1/sites")),
    ...RETRY,
  });
  return withFallback(query, () => [toSummary(builtinDemoSite())]);
}

export function useSite(id: string | null): CatalogResult<Site | null> {
  const query = useQuery({
    queryKey: queryKeys.site(id ?? "none"),
    queryFn: () =>
      unwrap<Site>(api.GET("/api/v1/sites/{site_id}", { params: { path: { site_id: id ?? "" } } })),
    enabled: Boolean(id) && !id?.startsWith("builtin-"),
    ...RETRY,
  });
  if (id?.startsWith("builtin-")) {
    return {
      data: builtinDemoSite(),
      builtin: true,
      isLoading: false,
      error: null,
      refetch: () => Promise.resolve(null),
    };
  }
  const result = withFallback(query, () => (id ? builtinDemoSite() : null));
  return { ...result, data: id ? result.data : null };
}

export function useLayers(): CatalogResult<Layer[]> {
  const query = useQuery({
    queryKey: queryKeys.layers,
    queryFn: () => unwrap<Layer[]>(api.GET("/api/v1/layers")),
    ...RETRY,
  });
  return withFallback(query, builtinLayers);
}

export function usePlannerStatus() {
  return useQuery({
    queryKey: ["planner-status"],
    queryFn: () => unwrap<PlannerStatus>(api.GET("/api/v1/agent/status")),
    staleTime: 5 * 60_000,
  });
}

/** Asks the mission planner for a structured plan draft. */
export function draftPlan(body: PlanDraftRequest): Promise<PlanDraft> {
  return unwrap<PlanDraft>(api.POST("/api/v1/agent/plan-draft", { body }));
}

/** The model outlines the feature under a clicked point in a picture of the view. */
export function outlineGround(body: OutlineRequest): Promise<GroundOutline> {
  return unwrap<GroundOutline>(api.POST("/api/v1/agent/outline", { body }));
}

/** A project's persisted plans; `builtin` when the API is offline (the browser store then holds them). */
export function usePlans(projectId: string | null): CatalogResult<PlanRecord[]> {
  const query = useQuery({
    queryKey: ["plans", projectId],
    queryFn: () =>
      unwrap<PlanRecord[]>(
        api.GET("/api/v1/plans", { params: { query: { projectId: projectId ?? "" } } }),
      ),
    enabled: projectId !== null,
    staleTime: 15_000,
  });
  return withFallback(query, () => []);
}

export const plansApi = {
  create: (body: PlanRecordCreate) => unwrap<PlanRecord>(api.POST("/api/v1/plans", { body })),
  revise: (planId: string, body: PlanRecordRevise) =>
    unwrap<PlanRecord>(
      api.PUT("/api/v1/plans/{plan_id}", { params: { path: { plan_id: planId } }, body }),
    ),
  setStatus: (planId: string, status: PlanRecordStatus) =>
    unwrap<PlanRecord>(
      api.PATCH("/api/v1/plans/{plan_id}/status", {
        params: { path: { plan_id: planId } },
        body: { status },
      }),
    ),
  remove: (planId: string) =>
    unwrap<unknown>(
      api.DELETE("/api/v1/plans/{plan_id}", { params: { path: { plan_id: planId } } }),
    ),
};

export function useIonStatus() {
  return useQuery({
    queryKey: queryKeys.ion,
    queryFn: () => unwrap<IonStatus>(api.GET("/api/v1/ion/status")),
    ...RETRY,
  });
}

export function useCreateSite() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (body: SiteCreate) => unwrap<Site>(api.POST("/api/v1/sites", { body })),
    onSuccess: () => void client.invalidateQueries({ queryKey: queryKeys.sites }),
  });
}

export function useCreateLayer() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (body: LayerCreate) => unwrap<Layer>(api.POST("/api/v1/layers", { body })),
    onSuccess: () => void client.invalidateQueries({ queryKey: queryKeys.layers }),
  });
}

export function useCreateAsset() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (body: AssetCreate) => unwrap(api.POST("/api/v1/assets", { body })),
    onSuccess: (_data, body) => {
      void client.invalidateQueries({ queryKey: queryKeys.sites });
      if (body.siteId) void client.invalidateQueries({ queryKey: queryKeys.site(body.siteId) });
    },
  });
}

export function useCreateBookmark(siteId: string | null) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (body: CameraBookmarkCreate) =>
      unwrap(
        api.POST("/api/v1/sites/{site_id}/bookmarks", {
          params: { path: { site_id: siteId ?? "" } },
          body,
        }),
      ),
    onSuccess: () => {
      if (siteId) void client.invalidateQueries({ queryKey: queryKeys.site(siteId) });
    },
  });
}

export function useDeleteBookmark(siteId: string | null) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (bookmarkId: string) =>
      unwrap(
        api.DELETE("/api/v1/sites/{site_id}/bookmarks/{bookmark_id}", {
          params: { path: { site_id: siteId ?? "", bookmark_id: bookmarkId } },
        }),
      ),
    onSuccess: () => {
      if (siteId) void client.invalidateQueries({ queryKey: queryKeys.site(siteId) });
    },
  });
}

export function useDeleteLayer() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (layerId: string) =>
      unwrap(api.DELETE("/api/v1/layers/{layer_id}", { params: { path: { layer_id: layerId } } })),
    onSuccess: () => void client.invalidateQueries({ queryKey: queryKeys.layers }),
  });
}

/* ---- Captures and jobs ---------------------------------------------------- */

/**
 * How often the panel asks the API what changed, while it is open.
 *
 * Uploads report themselves (the browser is doing them); this poll is for the half the
 * browser cannot see — a worker picking a job up and moving it through its stages.
 * `useHealth` set the precedent for polling here; four seconds is the same idea at the
 * pace a stage list has to move to look alive.
 */
const JOB_POLL_MS = 4000;

/** Captures, newest first. `builtin` here means "no API", which disables the drop zone. */
export function useCaptures(enabled = true): CatalogResult<Capture[]> {
  const query = useQuery({
    queryKey: queryKeys.captures,
    queryFn: () => unwrap<Capture[]>(api.GET("/api/v1/captures")),
    enabled,
    ...RETRY,
    staleTime: 0,
    refetchInterval: enabled ? JOB_POLL_MS : false,
  });
  return withFallback(query, () => []);
}

/**
 * Every recent job, polled alongside the captures.
 *
 * One list rather than a query per capture: `GET /captures` returns rows without their
 * jobs, and a panel of ten cards should not mean ten polls.
 */
export function useJobs(enabled = true): CatalogResult<Job[]> {
  const query = useQuery({
    queryKey: queryKeys.jobs,
    queryFn: () => unwrap<Job[]>(api.GET("/api/v1/jobs", { params: { query: { limit: 50 } } })),
    enabled,
    ...RETRY,
    staleTime: 0,
    refetchInterval: enabled ? JOB_POLL_MS : false,
  });
  return withFallback(query, () => []);
}

/** The newest job for each capture: a retry supersedes the run it retried. */
export function latestJobByCapture(jobs: Job[]): Record<string, Job> {
  const newest: Record<string, Job> = {};
  for (const job of jobs) {
    const current = newest[job.captureId];
    if (!current || current.createdAt < job.createdAt) newest[job.captureId] = job;
  }
  return newest;
}

export const capturesApi = {
  create: (body: CaptureCreate) => unwrap<Capture>(api.POST("/api/v1/captures", { body })),
  get: (captureId: string) =>
    unwrap<CaptureDetail>(
      api.GET("/api/v1/captures/{capture_id}", { params: { path: { capture_id: captureId } } }),
    ),
  process: (captureId: string, body: JobCreate) =>
    unwrap<Job>(
      api.POST("/api/v1/captures/{capture_id}/process", {
        params: { path: { capture_id: captureId } },
        body,
      }),
    ),
};

export function useProcessCapture() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ captureId, recipe }: { captureId: string; recipe: string }) =>
      capturesApi.process(captureId, { recipe }),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: queryKeys.captures });
      void client.invalidateQueries({ queryKey: queryKeys.jobs });
    },
  });
}

export function useCancelJob() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) =>
      unwrap<Job>(
        api.POST("/api/v1/jobs/{job_id}/cancel", { params: { path: { job_id: jobId } } }),
      ),
    onSuccess: () => void client.invalidateQueries({ queryKey: queryKeys.jobs }),
  });
}

/**
 * Retry a finished run from one of its stages.
 *
 * Deliberately not the same thing as the worker's own retry: the worker stops after a few
 * attempts and dead-letters the job, and this is a person asking again, which gives the
 * stages being re-run their attempt budget back. Everything before `fromStage` keeps its
 * completed step and its artifacts, and the worker skips it.
 */
export function useRetryJob() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ jobId, fromStage }: { jobId: string; fromStage?: string }) =>
      unwrap<Job>(
        api.POST("/api/v1/jobs/{job_id}/retry", {
          params: { path: { job_id: jobId } },
          body: { fromStage: fromStage ?? null } satisfies JobRetry,
        }),
      ),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: queryKeys.captures });
      void client.invalidateQueries({ queryKey: queryKeys.jobs });
    },
  });
}

/**
 * One step's log, fetched only when the drawer is open.
 *
 * Logs live in object storage, not in the database, so this is a request per drawer
 * rather than a field that would ride along on every poll of the job list.
 */
export function useStepLog(jobId: string, stepId: string, enabled: boolean) {
  return useQuery({
    queryKey: queryKeys.stepLog(jobId, stepId),
    queryFn: () =>
      unwrap<JobStepLog>(
        api.GET("/api/v1/jobs/{job_id}/steps/{step_id}/log", {
          params: { path: { job_id: jobId, step_id: stepId } },
        }),
      ),
    enabled,
    retry: false,
  });
}

/**
 * Mint a phone-handoff link for one capture.
 *
 * Not cached and not retried: a handoff token is short-lived by design, so a stale one
 * held in a query cache would be worse than none at all.
 */
export function useCreateHandoff() {
  return useMutation({
    mutationFn: ({ captureId }: { captureId: string }) =>
      unwrap(
        api.POST("/api/v1/captures/{capture_id}/handoff", {
          params: { path: { capture_id: captureId } },
        }),
      ),
    retry: false,
  });
}
