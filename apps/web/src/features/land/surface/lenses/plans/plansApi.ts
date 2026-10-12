import { useQuery, useQueryClient } from "@tanstack/react-query";

import type { LandArea } from "@twin/contracts";

import { ApiError, api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLandAccessReady, useLandScope } from "@/state/landIdentity";

export const PLANS_PAGE = 50;

/** The land's plans, newest first, a page at a time. Stale reasons follow the boundary. */
export function useActions(land: LandArea, offset: number) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: ["land-actions", scope, land.id, land.revision, offset],
    enabled: ready,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/actions", {
          params: { path: { land_id: land.id }, query: { limit: PLANS_PAGE, offset } },
        }),
      ),
    retry: false,
  });
}

/** Restoration and solar scenarios, newest first. Solar ones are made in the Solar lens. */
export function useScenarios(land: LandArea) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: ["land-scenarios", scope, land.id],
    enabled: ready,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/scenarios", { params: { path: { land_id: land.id } } }),
      ),
    retry: false,
  });
}

/** One plan, fresh from the server (the list may lag a save by a moment). */
export function useActionRecord(land: LandArea, id: string | null) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: ["land-action-record", scope, land.id, id],
    enabled: ready && Boolean(id),
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/actions/{action_id}", {
          params: { path: { land_id: land.id, action_id: id ?? "" } },
        }),
      ),
    retry: false,
  });
}

export function useActionHistory(land: LandArea, id: string | null, revision?: number) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: ["land-action-history", scope, land.id, id, revision],
    enabled: ready && Boolean(id),
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/actions/{action_id}/revisions", {
          params: { path: { land_id: land.id, action_id: id ?? "" } },
        }),
      ),
    retry: false,
  });
}

/** The robot mission a scheduled plan became. */
export function useActionMission(land: LandArea, id: string | null, missionId?: string | null) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: ["land-action-mission", scope, id, missionId],
    enabled: ready && Boolean(id && missionId),
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/actions/{action_id}/mission", {
          params: { path: { land_id: land.id, action_id: id ?? "" } },
        }),
      ),
    retry: false,
  });
}

/** Mapped assets and areas, for step areas, areas to avoid and references. */
export function useLandFeatures(land: LandArea, enabled: boolean) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: ["land-action-features", scope, land.id],
    enabled: ready && enabled,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/features", {
          params: { path: { land_id: land.id }, query: { limit: 500 } },
        }),
      ),
    retry: false,
  });
}

/** Field surveys a restoration scenario can cite. */
export function useScenarioSurveys(land: LandArea) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: ["land-surveys", scope, land.id, "scenario-references"],
    enabled: ready,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/surveys", {
          params: { path: { land_id: land.id }, query: { limit: 100 } },
        }),
      ),
    retry: false,
  });
}

/** Refetches every plan list and record for this land. */
export function useRefreshPlans(land: LandArea) {
  const scope = useLandScope();
  const cache = useQueryClient();
  return () =>
    Promise.all([
      cache.invalidateQueries({ queryKey: ["land-actions", scope, land.id] }),
      cache.invalidateQueries({ queryKey: ["land-action-record", scope, land.id] }),
    ]);
}

/** What failed, in words: the server's field errors when it gave some. */
export function planError(cause: unknown): string {
  return cause instanceof ApiError && cause.fieldErrors.length
    ? cause.fieldErrors.join(". ")
    : describeError(cause);
}
