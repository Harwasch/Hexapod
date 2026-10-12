import { useMemo } from "react";
import { useQueries, useQuery, useQueryClient } from "@tanstack/react-query";

import type { LandArea } from "@twin/contracts";

import { api, unwrap } from "@/api/client";
import { useLandAccessReady, useLandScope } from "@/state/landIdentity";

import type { Asset, Inspection } from "./assetsModel";

/** One page of the asset list; the newest changes come first. */
export const ASSET_PAGE = 500;
/** How many assets the card checks the newest inspection of (the most recently changed). */
export const CONDITION_LIMIT = 60;

export const assetKeys = {
  all: (scope: string, landId: string) => ["land-inventory", scope, landId] as const,
  list: (scope: string, landId: string, revision: number, offset: number) =>
    ["land-inventory", scope, landId, revision, offset] as const,
  records: (scope: string, landId: string) => ["land-inventory-record", scope, landId] as const,
  record: (scope: string, landId: string, revision: number, id: string | null) =>
    ["land-inventory-record", scope, landId, revision, id] as const,
  inspections: (scope: string, landId: string, id?: string) =>
    id
      ? (["land-inspections", scope, landId, id] as const)
      : (["land-inspections", scope, landId] as const),
  latest: (scope: string, landId: string, id: string) =>
    ["land-inspections", scope, landId, id, "latest"] as const,
  history: (scope: string, landId: string, id: string, revision: number | undefined) =>
    ["land-feature-history", scope, landId, id, revision] as const,
  imports: (scope: string, landId: string) => ["land-inventory-imports", scope, landId] as const,
};

/** Everything mapped or inspected on this land, one page at a time. */
export function useAssetList(land: LandArea, offset: number) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: assetKeys.list(scope, land.id, land.revision, offset),
    enabled: ready,
    retry: false,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/features", {
          params: { path: { land_id: land.id }, query: { limit: ASSET_PAGE, offset } },
        }),
      ),
  });
}

/** The selected asset as saved now (it may sit on another page of the list). */
export function useAssetRecord(land: LandArea, id: string | null) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: assetKeys.record(scope, land.id, land.revision, id),
    enabled: ready && Boolean(id),
    retry: false,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/features/{feature_id}", {
          params: { path: { land_id: land.id, feature_id: id ?? "" } },
        }),
      ),
  });
}

/** One asset's inspections, newest first. */
export function useAssetInspections(land: LandArea, id: string | null) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: assetKeys.inspections(scope, land.id, id ?? ""),
    enabled: ready && Boolean(id),
    retry: false,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/features/{feature_id}/inspections", {
          params: { path: { land_id: land.id, feature_id: id ?? "" } },
        }),
      ),
  });
}

/** One asset's saved revisions, read when its history is opened. */
export function useAssetHistory(land: LandArea, asset: Asset | null, enabled: boolean) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: assetKeys.history(scope, land.id, asset?.id ?? "", asset?.revision),
    enabled: ready && enabled && Boolean(asset),
    retry: false,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/features/{feature_id}/revisions", {
          params: { path: { land_id: land.id, feature_id: asset?.id ?? "" } },
        }),
      ),
  });
}

/**
 * Each asset's newest inspection, for the condition colours and "needs attention": `undefined`
 * while loading or unreadable, `null` when the asset was never inspected. Bounded to the most
 * recently changed assets.
 */
export function useLatestInspections(land: LandArea, assets: readonly Asset[]) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  const checked = assets.slice(0, CONDITION_LIMIT);
  const results = useQueries({
    queries: checked.map((asset) => ({
      queryKey: assetKeys.latest(scope, land.id, asset.id),
      enabled: ready,
      retry: false,
      staleTime: 60_000,
      queryFn: async (): Promise<Inspection | null> => {
        const list = await unwrap(
          api.GET("/api/v1/land/{land_id}/features/{feature_id}/inspections", {
            params: { path: { land_id: land.id, feature_id: asset.id }, query: { limit: 1 } },
          }),
        );
        return list[0] ?? null;
      },
    })),
  });
  const signature = results.map((result) => result.dataUpdatedAt).join(",");
  const byId = useMemo(() => {
    const value: Record<string, Inspection | null | undefined> = {};
    checked.forEach((asset, index) => {
      value[asset.id] = results[index]?.data;
    });
    // Assets past the bound count as unknown rather than as still loading.
    for (const asset of assets.slice(CONDITION_LIMIT)) value[asset.id] = null;
    return value;
    // `results` is a new array every render; its data changes only with `signature`.
  }, [signature, assets]);
  return {
    byId,
    failed: results.some((result) => result.isError),
    limited: assets.length > CONDITION_LIMIT,
  };
}

/**
 * Refreshes the asset list and records after a save; with an asset id (a new inspection), only
 * that asset's inspections and condition, not every asset's.
 */
export function useRefreshAssets(land: LandArea) {
  const scope = useLandScope();
  const cache = useQueryClient();
  return async (inspected?: string) => {
    if (inspected) {
      await cache.invalidateQueries({ queryKey: assetKeys.inspections(scope, land.id, inspected) });
      return;
    }
    await Promise.all([
      cache.invalidateQueries({ queryKey: assetKeys.all(scope, land.id) }),
      cache.invalidateQueries({ queryKey: assetKeys.records(scope, land.id) }),
    ]);
  };
}
