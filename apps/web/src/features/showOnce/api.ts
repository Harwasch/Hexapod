/**
 * Show once, everywhere from the API: the Land's rasters (to find its signatures), one tap's
 * similarity and the alike cells as areas. Read-only computation over a stored raster.
 */

import { useQuery } from "@tanstack/react-query";

import type {
  LandArea,
  SignatureSimilarity,
  SignatureZones,
  SignatureZonesRequest,
  components,
} from "@twin/contracts";

import { api, unwrap } from "@/api/client";
import { useLandAccessReady, useLandScope } from "@/state/landIdentity";

type LandRaster = components["schemas"]["LandRasterRead"];

/**
 * The Land's saved rasters, newest first. Under "land-rasters", so a finished research run
 * (`invalidateLandDerived`) asks again; keyed by the boundary revision, so a revised outline
 * re-reads which rasters are stale.
 */
export function useLandRasters(land: Pick<LandArea, "id" | "revision">, enabled = true) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: ["land-rasters", scope, "land", land.id, land.revision],
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/rasters", {
          params: { path: { land_id: land.id }, query: { limit: 100 } },
        }),
      ),
    enabled: enabled && ready,
    staleTime: 30_000,
    retry: false,
  });
}

/** The newest signature raster made for the Land's current boundary with cells to compare. */
export function freshSignature(
  rasters: readonly LandRaster[] | undefined,
  land: Pick<LandArea, "revision">,
): LandRaster | null {
  return (
    rasters?.find(
      (raster) =>
        raster.request.dataset === "sentinel-2-signature" &&
        !raster.stale &&
        raster.boundaryRevision === land.revision &&
        (raster.metadata.signature?.commonCells ?? 0) > 0,
    ) ?? null
  );
}

/** Whether signatures exist only for an earlier boundary (the card says to prepare them again). */
export function onlyStaleSignatures(
  rasters: readonly LandRaster[] | undefined,
  land: Pick<LandArea, "revision">,
): boolean {
  return (
    !freshSignature(rasters, land) &&
    (rasters?.some((raster) => raster.request.dataset === "sentinel-2-signature") ?? false)
  );
}

export function fetchSimilar(
  rasterId: string,
  longitude: number,
  latitude: number,
): Promise<SignatureSimilarity> {
  return unwrap(
    api.GET("/api/v1/land/rasters/{raster_id}/similar", {
      params: { path: { raster_id: rasterId }, query: { longitude, latitude } },
    }),
  );
}

export function requestSignatureZones(
  rasterId: string,
  body: SignatureZonesRequest,
): Promise<SignatureZones> {
  return unwrap(
    api.POST("/api/v1/land/rasters/{raster_id}/similar/zones", {
      params: { path: { raster_id: rasterId } },
      body,
    }),
  );
}
