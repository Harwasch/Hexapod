import { useMemo } from "react";

import type { SiteSummary } from "@twin/contracts";

import { useSites as useSiteCatalog } from "@/api/queries";
import { PROCEDURAL_SLUGS } from "@/lib/procedural";
import { useSettings } from "@/state/settings";
import { useSites } from "@/state/sites";

/**
 * Developer fixtures in the site catalog, and the one rule for where they are listed.
 *
 * The catalog holds sites that exist for the code rather than for anyone's land: the synthetic
 * tree the Living Survey motion model is tested against ("Synthetic tree (Living Survey fixture)",
 * procedural, never captured), and records an audit or a probe script left behind. Listed beside
 * the demo ranch and real scans they read as places someone captured. They stay in the catalog
 * (tests and the developer panel use them) and are listed when developer readouts are on
 * (Settings › Advanced), and the site you are at is always listed.
 *
 * A fixture is known by an explicit marker in its slug, never by ordinary words in its name: "Barn
 * roof audit" or "Soil probe plot" is someone's site. The markers are the known fixture slugs and
 * the prefixes an audit or probe script gives the records it creates (`audit-…`, `probe-…`). Even
 * then a site with an observation date (a capture's photos have one) is never taken for one.
 */

type Described = Pick<SiteSummary, "slug" | "latestObservedAt">;

/** Fixtures known by their catalog slug. */
const FIXTURE_SLUGS = PROCEDURAL_SLUGS;

/** The slug prefixes audit and probe scripts give the sites they create. */
const SCRIPT_SLUG = /^(audit|probe)-/;

/** Whether a catalog site is a developer fixture rather than a place someone captured. */
export function isDeveloperFixture(site: Described): boolean {
  if (site.latestObservedAt) return false;
  return FIXTURE_SLUGS.has(site.slug) || SCRIPT_SLUG.test(site.slug);
}

/**
 * The sites to list, and how many fixtures were left out: all of them with developer readouts
 * on, else every site but the fixtures, keeping the one you are at.
 */
export function listedSites<S extends Described & { id: string }>(
  sites: readonly S[],
  options: { developer: boolean; hereId?: string | null },
): { shown: S[]; hidden: number } {
  if (options.developer) return { shown: [...sites], hidden: 0 };
  const shown = sites.filter((site) => site.id === options.hereId || !isDeveloperFixture(site));
  return { shown, hidden: sites.length - shown.length };
}

/** The catalog's sites as the site switcher, the command box and the status line list them. */
export function useListedSites(): { shown: SiteSummary[]; hidden: number } {
  const catalog = useSiteCatalog();
  const developer = useSettings((s) => s.devReadouts);
  const hereId = useSites((s) => s.activeSiteId);
  return useMemo(
    () => listedSites(catalog.data ?? [], { developer, hereId }),
    [catalog.data, developer, hereId],
  );
}
