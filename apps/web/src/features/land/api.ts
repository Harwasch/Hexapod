import { useQuery } from "@tanstack/react-query";

import type { LandCreate, LandRevise } from "@twin/contracts";

import { api, unwrap } from "@/api/client";

export function useLandAreas(enabled: boolean) {
  return useQuery({
    queryKey: ["land-areas"],
    queryFn: () => unwrap(api.GET("/api/v1/land", { params: { query: { limit: 200 } } })),
    enabled,
    retry: false,
  });
}

export function useBoundaryHistory(id: string | null) {
  return useQuery({
    queryKey: ["land-history", id],
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/revisions", { params: { path: { land_id: id ?? "" } } }),
      ),
    enabled: id !== null,
    retry: false,
  });
}

export const createLand = (body: LandCreate) => unwrap(api.POST("/api/v1/land", { body }));
export const reviseLand = (id: string, body: LandRevise) =>
  unwrap(api.PUT("/api/v1/land/{land_id}", { params: { path: { land_id: id } }, body }));
export const createCorridor = (coordinates: number[][], widthM: number) =>
  unwrap(api.POST("/api/v1/land/corridor", { body: { coordinates, widthM, cap: "round" } }));
