import {
  useMutation,
  useQuery,
  useQueryClient,
  type QueryClient,
  type UseQueryResult,
} from "@tanstack/react-query";

import type {
  AssetCreate,
  AssetScaleUpdate,
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
  LandGround,
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
  SiteAsset,
  SiteCreate,
  SiteSummary,
  WeakAreas,
  WeakAreasRequest,
  WorkspaceMissionCreate,
} from "@twin/contracts";

import { env } from "@/app/env";
import { landIdOfProject } from "@/missions/landGround";
import { useLand } from "@/state/land";
import { landUsesOidc, useLandAccessReady, useLandScope } from "@/state/landIdentity";
import { useMission } from "@/state/mission";
import { useSettings } from "@/state/settings";

import { api, isOffline, unwrap } from "./client";
import {
  builtinLayers,
  builtinScanSite,
  builtinSite,
  builtinSites,
  fetchOfflineCatalog,
  toSummary,
} from "./fallback";

export const queryKeys = {
  health: ["health"] as const,
  sites: ["sites"] as const,
  site: (id: string) => ["sites", id] as const,
  layers: ["layers"] as const,
  ion: ["ion"] as const,
  offlineCatalog: ["offline-catalog"] as const,
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

/**
 * The API scales to zero (fly.toml), so a first request can meet a cold start. The built-in
 * catalog stands in after one quick retry, as before, but a query that failed now asks
 * again every few seconds until the API answers: the real sites replace it without a reload.
 */
const RETRY = {
  retry: 1,
  retryDelay: 800,
  staleTime: 30_000,
  refetchInterval: (query: { state: { status: string } }) =>
    query.state.status === "error" ? RECOVER_POLL_MS : false,
} as const;
const RECOVER_POLL_MS = 5000;

/**
 * The captures published to object storage, fetched only once the API has actually
 * failed.
 *
 * `enabled` is the whole point: a healthy deployment never requests this file, so the
 * offline catalog costs nothing until it is the only catalog there is. It is not retried
 * either — if the bucket is unreachable too, the built-in demo is the honest answer and a
 * retry loop would only delay it. With no `VITE_OFFLINE_CATALOG_URL` configured (a local
 * checkout with no bucket) there is nothing to fetch and the query never runs.
 */
function useOfflineSites(enabled: boolean): Site[] {
  const url = env.offlineCatalogUrl;
  const query = useQuery({
    queryKey: queryKeys.offlineCatalog,
    queryFn: ({ signal }) => fetchOfflineCatalog(url ?? "", signal),
    enabled: enabled && Boolean(url),
    retry: false,
    staleTime: Infinity,
  });
  return query.data ?? [];
}

export function useHealth() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: () => unwrap<HealthStatus>(api.GET("/api/v1/health")),
    ...RETRY,
    refetchInterval: 60_000,
    // TanStack Query's default, stated so nobody turns it on: /health queries the database,
    // and a background tab polling it every minute would keep Neon from scaling to zero.
    refetchIntervalInBackground: false,
  });
}

export function useSites(): CatalogResult<SiteSummary[]> {
  const query = useQuery({
    queryKey: queryKeys.sites,
    queryFn: () => unwrap<SiteSummary[]>(api.GET("/api/v1/sites")),
    ...RETRY,
  });
  const captures = useOfflineSites(query.isError && isOffline(query.error));
  return withFallback(query, () => {
    const builtins = builtinSites();
    // The published catalog lists the seeded ranch and scan showcase too; the built-in copy
    // of each stands for it, so the fleet is not attached (and counted) twice.
    const slugs = new Set(builtins.map((site) => site.slug));
    return [
      ...builtins.map(toSummary),
      ...captures.filter((site) => !slugs.has(site.slug)).map((site) => toSummary(site)),
    ];
  });
}

export function useSite(id: string | null): CatalogResult<Site | null> {
  const query = useQuery({
    queryKey: queryKeys.site(id ?? "none"),
    queryFn: () =>
      unwrap<Site>(api.GET("/api/v1/sites/{site_id}", { params: { path: { site_id: id ?? "" } } })),
    enabled: Boolean(id) && !id?.startsWith("builtin-"),
    ...RETRY,
  });
  const captures = useOfflineSites(query.isError && isOffline(query.error));
  if (id?.startsWith("builtin-")) {
    return {
      data: builtinSite(id) ?? builtinScanSite(),
      builtin: true,
      isLoading: false,
      error: null,
      refetch: () => Promise.resolve(null),
    };
  }
  // A published capture keeps its real id, so the same id that resolved against the API
  // resolves against the catalog in the bucket — the site the user was looking at does not
  // change identity when the API goes away.
  const result = withFallback(query, () =>
    id ? (captures.find((site) => site.id === id) ?? builtinScanSite()) : null,
  );
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

/**
 * An epoch's weak cells (weak-v1) as at most `maxAreas` re-scan zones, each with the pattern
 * to fly it again (ADR 0016). Stores nothing, but takes the write token: it is the first step
 * of planning a flight. A 409 says why there is nothing to outline (`problem.code`).
 */
export function requestWeakAreas(
  siteId: string,
  epochId: string,
  body: WeakAreasRequest,
): Promise<WeakAreas> {
  return unwrap<WeakAreas>(
    api.POST("/api/v1/sites/{site_id}/epochs/{epoch_id}/weak-areas", {
      params: { path: { site_id: siteId, epoch_id: epochId } },
      body,
    }),
  );
}

const NO_PLANS: PlanRecord[] = [];
const resolveNoPlans = (): Promise<PlanRecord[]> => Promise.resolve(NO_PLANS);

/**
 * A project's persisted plans; `builtin` when the API is offline (the browser store then holds
 * them). A Land's own project (`land:<id>`) holds only private missions, which the public
 * route can never carry, so it is not asked at all (and the land id stays out of its logs):
 * the answer is a settled empty list, and `useWorkspacePlans` brings that project's plans.
 */
export function usePlans(projectId: string | null): CatalogResult<PlanRecord[]> {
  const landScoped = landIdOfProject(projectId) !== null;
  const query = useQuery({
    queryKey: ["plans", projectId],
    queryFn: () =>
      unwrap<PlanRecord[]>(
        api.GET("/api/v1/plans", { params: { query: { projectId: projectId ?? "" } } }),
      ),
    enabled: projectId !== null && !landScoped,
    staleTime: 15_000,
  });
  const result = withFallback(query, () => []);
  if (!landScoped) return result;
  return { data: NO_PLANS, builtin: false, isLoading: false, error: null, refetch: resolveNoPlans };
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

/**
 * The current workspace's missions for a project: plans on private ground (a saved Land),
 * which the public `/plans` list never includes. Off when land exploration is off or the
 * workspace is not ready; an error leaves the public list to stand alone. In pilot mode it
 * asks only once land access is in use (a token entered, or a Land selected or engaged), so
 * a visitor without the write token is never prompted for it by the plan list.
 */
export function useWorkspacePlans(projectId: string | null) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  const hasToken = useSettings((s) => s.writeToken.trim() !== "");
  const landSelected = useLand((s) => s.active !== null);
  const landEngaged = useMission(
    (s) => s.landGround !== null || landIdOfProject(s.project?.id) !== null,
  );
  return useQuery({
    queryKey: ["workspace-plans", scope, projectId],
    queryFn: () =>
      unwrap<PlanRecord[]>(
        api.GET("/api/v1/workspaces/missions", {
          params: { query: { projectId: projectId ?? undefined } },
        }),
      ),
    enabled:
      env.landExplorationEnabled &&
      ready &&
      projectId !== null &&
      (landUsesOidc || hasToken || landSelected || landEngaged),
    retry: false,
    staleTime: 15_000,
  });
}

/** Writes to a workspace's missions: the private twin of `plansApi`. */
export const workspaceMissionsApi = {
  create: (body: WorkspaceMissionCreate) =>
    unwrap<PlanRecord>(api.POST("/api/v1/workspaces/missions", { body })),
  revise: (planId: string, body: PlanRecordRevise) =>
    unwrap<PlanRecord>(
      api.PUT("/api/v1/workspaces/missions/{plan_id}", {
        params: { path: { plan_id: planId } },
        body,
      }),
    ),
  setStatus: (planId: string, status: PlanRecordStatus) =>
    unwrap<PlanRecord>(
      api.PATCH("/api/v1/workspaces/missions/{plan_id}/status", {
        params: { path: { plan_id: planId } },
        body: { status },
      }),
    ),
};

const groundInFlight = new Map<string, Promise<LandGround>>();

/**
 * A saved Land as mission ground (its boundary and inventory features as zones). Calls for
 * the same land and boundary revision while one is in flight share it; a settled call is
 * never reused, so a feature edited since is always fetched fresh.
 */
export function fetchLandGround(landId: string, revision: number): Promise<LandGround> {
  const key = `${landId}@${revision}`;
  const pending = groundInFlight.get(key);
  if (pending) return pending;
  const request = unwrap<LandGround>(
    api.GET("/api/v1/land/{land_id}/ground", { params: { path: { land_id: landId } } }),
  ).finally(() => groundInFlight.delete(key));
  groundInFlight.set(key, request);
  return request;
}

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

/**
 * Renames a site. Only `name` is sent: the PATCH replaces whatever else it is given, and
 * `metadata` in particular would replace the whole dict.
 *
 * The answer is the site's full record, put straight into its query: `watchSiteRecords` hands
 * it to the scene, whose copy names the cards it builds, even for a site no component has a
 * query open for. The catalog list (the switcher, the command box) takes the name at once,
 * so the row does not show the old one while the list is fetched afresh.
 */
export function useRenameSite() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ siteId, name }: { siteId: string; name: string }) =>
      unwrap<Site>(
        api.PATCH("/api/v1/sites/{site_id}", {
          params: { path: { site_id: siteId } },
          body: { name },
        }),
      ),
    onSuccess: (site) => {
      client.setQueryData(queryKeys.site(site.id), site);
      client.setQueryData<SiteSummary[]>(queryKeys.sites, (list) =>
        list?.map((entry) => (entry.id === site.id ? { ...entry, name: site.name } : entry)),
      );
      void client.invalidateQueries({ queryKey: queryKeys.sites, exact: true });
    },
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

/**
 * Calls `onRecord` with every site record the API answers through the query cache: a site's
 * own query (`queryKeys.site`) fetched, refetched after a bookmark was saved or deleted or an
 * asset added (the mutations below invalidate it), or set by hand. Only what the API sent:
 * `useSite`'s offline fallback is made in the hook and never stored here.
 *
 * The scene keeps its own copy of each record it has flown to (SiteManager, so a second
 * flight leaves for the bookmark at once), and a bookmark mutation leaves the catalog list --
 * all the scene is otherwise handed -- equal, so react-query keeps its identity and nothing
 * told the scene. This is how it hears.
 */
export function watchSiteRecords(client: QueryClient, onRecord: (site: Site) => void): () => void {
  return client.getQueryCache().subscribe((event) => {
    if (event.type !== "updated" || event.action.type !== "success") return;
    const key = event.query.queryKey as readonly unknown[];
    if (key.length !== 2 || key[0] !== queryKeys.sites[0]) return;
    const site = event.query.state.data as Site | null | undefined;
    if (site && typeof site === "object" && Array.isArray(site.cameraBookmarks)) onRecord(site);
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

/**
 * Sets a scan's real-world size (`PUT /assets/{id}/scale`: the Set real size tool), or puts it
 * back as registered with `RESET_SCALE_BODY` (lib/realSize.ts). The answer is the asset, and
 * every globe position of its site moved with it -- boundary, centroid, footprint, ground
 * samples (docs/DATA_MODEL.md "Runtime scale") -- so:
 *
 * - the asset goes straight into its site's record, which `watchSiteRecords` hands the scene
 *   at once: it draws the scan at the saved scale (the preview gives way to it, with no flash
 *   of the old size between) and the inspector says how the scale was set;
 * - the site's record is fetched afresh for its boundary and centroid, and the catalog list
 *   for the summary's centroid and area.
 */
export function useSetAssetScale() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ assetId, body }: { siteId: string; assetId: string; body: AssetScaleUpdate }) =>
      unwrap<SiteAsset>(
        api.PUT("/api/v1/assets/{asset_id}/scale", {
          params: { path: { asset_id: assetId } },
          body,
        }),
      ),
    onSuccess: (asset, { siteId }) => {
      client.setQueryData<Site | null>(queryKeys.site(siteId), (site) =>
        site
          ? { ...site, assets: site.assets.map((entry) => (entry.id === asset.id ? asset : entry)) }
          : site,
      );
      void client.invalidateQueries({ queryKey: queryKeys.site(siteId), exact: true });
      void client.invalidateQueries({ queryKey: queryKeys.sites, exact: true });
    },
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
