/**
 * Catalog sites generated for the code rather than captured: the synthetic tree the Living
 * Survey's motion model is tested against ("procedural; no capture, no reconstruction"). Known
 * by slug, never by name. The site lists hide them (`features/sites/siteFixtures.ts`), and As
 * captured draws none of them (`SiteManager.hideFixtures`, ADR 0019).
 */
export const PROCEDURAL_SLUGS: ReadonlySet<string> = new Set(["synthetic-tree"]);
