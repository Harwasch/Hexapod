import { useMutation, useQuery, useQueryClient, type UseQueryResult } from "@tanstack/react-query";

import type {
  AssetCreate,
  CameraBookmarkCreate,
  HealthStatus,
  IonStatus,
  Layer,
  LayerCreate,
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
};

/** Result decorated with whether the data came from the built-in fallback catalog. */
export interface CatalogResult<T> {
  data: T;
  builtin: boolean;
  isLoading: boolean;
  error: unknown;
}

function withFallback<T>(query: UseQueryResult<T>, fallback: () => T): CatalogResult<T> {
  const offline = query.isError && isOffline(query.error);
  return {
    data: query.data ?? (offline ? fallback() : (query.data as T)),
    builtin: offline,
    isLoading: query.isPending,
    error: query.isError && !offline ? query.error : null,
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
    return { data: builtinDemoSite(), builtin: true, isLoading: false, error: null };
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
