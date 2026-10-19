import { useQuery } from "@tanstack/react-query";

import type { LandArea } from "@twin/contracts";

import { api, unwrap } from "@/api/client";
import { useLandAccessReady, useLandScope } from "@/state/landIdentity";

/** Saved assessments per page, newest first. */
export const PAGE = 10;

/** One page of this land's saved hourly assessments. `refresh` changes when a run settles. */
export function useSolarAssessments(land: LandArea, offset: number, refresh: string) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: ["land-solar", scope, land.id, land.revision, offset, refresh],
    enabled: ready,
    retry: false,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/solar-assessments", {
          params: { path: { land_id: land.id }, query: { limit: PAGE, offset } },
        }),
      ),
  });
}

/** One saved assessment by id (a selection from a saved view may be off the current page). */
export function useSolarAssessment(id: string | null, revision?: number) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: ["land-solar-assessment", scope, id, revision],
    enabled: ready && Boolean(id),
    retry: false,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/solar-assessments/{assessment_id}", {
          params: { path: { assessment_id: id ?? "" } },
        }),
      ),
  });
}

/** Every scenario on this land; the solar ones are this lens's. Shares Plans' cache entry. */
export function useLandScenarios(land: LandArea) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: ["land-scenarios", scope, land.id],
    enabled: ready,
    retry: false,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/scenarios", { params: { path: { land_id: land.id } } }),
      ),
  });
}

/** Mapped shapes on this land that could be an array (roofs, fields), when asked for. */
export function useMappedShapes(land: LandArea, enabled: boolean) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  return useQuery({
    queryKey: ["land-solar-features", scope, land.id],
    enabled: ready && enabled,
    retry: false,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/features", {
          params: { path: { land_id: land.id }, query: { limit: 100 } },
        }),
      ),
    select: (features) =>
      features.filter(
        (feature) =>
          feature.geometry.type === "Polygon" || feature.geometry.type === "MultiPolygon",
      ),
  });
}

/** Saves an assessment's hourly data and sources as a zip. */
export async function downloadAssessment(id: string): Promise<void> {
  const blob = await unwrap(
    api.GET("/api/v1/land/solar-assessments/{assessment_id}/download", {
      params: { path: { assessment_id: id } },
      parseAs: "blob",
    }),
  );
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `solar-${id}.zip`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
