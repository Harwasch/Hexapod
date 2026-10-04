import type { SiteSummary } from "@twin/contracts";

import { useSites as useSiteCatalog } from "@/api/queries";
import { DemoMissionProvider } from "@/missions/demo";
import type { MissionProvider } from "@/missions/types";

/**
 * One name per site, everywhere it is shown (docs/GLOSSARY.md, "Site").
 *
 * A site is known by its project's name when a mission project is attached to it — the demo
 * site is "Blackrock Mesa" in the site switcher, the status line and the agent's words — and
 * by its catalog name otherwise. Without this the same place was "Blackrock Mesa" in one
 * corner and "Cesium Gaussian splat demo" (the record of the tileset it was made from) in
 * another. The provider is the same seam `SceneBridge` resolves projects through.
 */
const provider: MissionProvider = new DemoMissionProvider();

type Named = Pick<SiteSummary, "id" | "slug" | "name">;

export function siteDisplayName(site: Named, missions: MissionProvider = provider): string {
  return missions.projectForSite(site.id, site.slug)?.name ?? site.name;
}

/** The display name of a catalog site, by id; null while the catalog has no such site. */
export function useSiteName(siteId: string | null | undefined): string | null {
  const catalog = useSiteCatalog();
  const site = siteId ? catalog.data?.find((candidate) => candidate.id === siteId) : undefined;
  return site ? siteDisplayName(site) : null;
}
