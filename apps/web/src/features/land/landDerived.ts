import type { QueryClient } from "@tanstack/react-query";

/**
 * Query keys whose answers are computed from a Land's stored records: its mission ground
 * (boundary and inventory as zones), the Fog of Knowing, its rasters, and the workable ground
 * the dress rehearsal offers (read on its elevation, named for its outline's revision). One
 * place to say "this land's records changed", so every view computed from them asks again.
 */
export const LAND_DERIVED_KEYS = [
  "land-ground",
  "land-knowledge",
  "land-rasters",
  "land-workable",
] as const;

/**
 * Marks everything derived from a Land's records as stale after one of them changed: an
 * inspection saved, an inventory feature added or revised, a research run finished. With
 * `landId` null (a run whose land is not at hand) every land's derived answers are, except
 * workable ground: its key names everything it is read from (the outline, the machine class,
 * the elevation raster, the creek rule), so a run finishing cannot change it, and each answer
 * asked again is a slope and creek computation on the API. It is asked again only when its own
 * land's records change (a revised boundary).
 * Rasters are keyed by raster id, so all of them are refreshed; they are few and small.
 */
export function invalidateLandDerived(
  queryClient: QueryClient,
  landId: string | null,
): Promise<void> {
  return queryClient.invalidateQueries({
    predicate: (query) => {
      const [name, ...rest] = query.queryKey;
      if (!LAND_DERIVED_KEYS.includes(name as (typeof LAND_DERIVED_KEYS)[number])) return false;
      if (name === "land-workable") return landId !== null && rest.includes(landId);
      return name === "land-rasters" || landId === null || rest.includes(landId);
    },
  });
}
