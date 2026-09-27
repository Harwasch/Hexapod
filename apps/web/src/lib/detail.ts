/**
 * The phone's "Detail" choice, read wherever a splat is drawn.
 *
 * It used to be a packaging cut: `package.max_gaussians` kept the 200k / 400k / 800k most
 * opaque gaussians of a scan and threw the rest away, for every viewer, for good. The
 * pipeline now packs every gaussian as a level-of-detail tileset
 * (tools/captures/splat_tiles.py), so how many are drawn is decided on the device that pays
 * for drawing them, and the choice is this device's, not the scan's:
 *
 * - the scan viewer (view.html, Spark) loads tiles coarsest-first until this many gaussians
 *   are loaded (`view/tiles.ts`);
 * - the globe (CesiumJS) scales a splat tileset's `maximumScreenSpaceError` by
 *   `detailScreenSpaceScale`, since it refines by screen-space error and has no count budget.
 *
 * The upload page is where it is chosen (`upload/options.ts`), and the three pages share an
 * origin, so the choice saved there is read here.
 */

import { readJson } from "./storage";

/** Where `upload/options.ts` saves the phone's choices, `detail` among them. */
export const OPTIONS_STORAGE = "twin.phoneOptions.v3";

/** The Detail choices, as the gaussians a view may hold at once. */
export const DETAIL_BUDGETS = [200_000, 400_000, 800_000] as const;

/** "Standard": what a phone was sent before the hierarchy, and the reference below. */
export const DEFAULT_SPLAT_BUDGET = 400_000;

/** This device's gaussian budget: its saved Detail choice, or Standard. */
export function splatBudget(): number {
  const stored = readJson(OPTIONS_STORAGE) as { detail?: unknown } | undefined;
  const budget = Number(stored?.detail);
  return (DETAIL_BUDGETS as readonly number[]).includes(budget) ? budget : DEFAULT_SPLAT_BUDGET;
}

/**
 * The factor on a splat tileset's screen-space error that makes the globe draw about
 * `budget` gaussians where Standard draws 400k.
 *
 * Tiles refine until the gaussian spacing projects to under the error in pixels, so a scan
 * filling the view draws about (covered pixels) / error^2 gaussians: to draw k times as many,
 * divide the error by sqrt(k). Light is x1.41 (coarser), Full x0.71 (finer).
 */
export function detailScreenSpaceScale(budget: number): number {
  return Math.sqrt(DEFAULT_SPLAT_BUDGET / Math.max(1, budget));
}
