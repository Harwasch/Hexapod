/**
 * The phone's "Detail" choice, read wherever a splat is drawn.
 *
 * It used to be a packaging cut: `package.max_gaussians` kept the 200k / 400k / 800k most
 * opaque gaussians of a scan and threw the rest away, for every viewer, for good. The
 * pipeline now packs every gaussian as a level-of-detail tileset
 * (tools/captures/splat_tiles.py), so how many are drawn is decided on the device that pays
 * for drawing them, and the choice is this device's, not the scan's:
 *
 * - the scan viewer (view.html, Spark) hands it to Spark's own level of detail as
 *   `SparkRenderer.lodSplatCount`, the most splats drawn a frame, and downloads up to
 *   `LOAD_FACTOR` times as many (`view/tiles.ts`);
 * - the globe (CesiumJS) scales a splat tileset's `maximumScreenSpaceError` by
 *   `detailScreenSpaceScale`, since Cesium refines by screen-space error, and counts the
 *   splats it has loaded against it as a memory budget (`cesium/splatCount.ts`), since
 *   Cesium's own byte count misses splats.
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

/** Splats a desktop draws at Standard, by the memory it reports (`navigator.deviceMemory`,
 *  Chromium only; absent means a desktop browser that does not say, taken as 8 GB). */
const DESKTOP_BUDGETS: [minGb: number, splats: number][] = [
  [8, 3_000_000],
  [4, 1_500_000],
  [0, 800_000],
];

/** Whether this looks like a phone or tablet: a coarse pointer and a small screen. */
/** Desktop views may grow past the device budget to this many times it, while frames stay
 *  fast (lib/splatBudget.ts): memory sets the budget, but only trying shows what the GPU
 *  draws smoothly, and a dense scan seen from inside needs about twice the standard 3M. */
const DESKTOP_GROWTH = 2;
/** However fast the frames, never more than this. */
const MAX_SPLAT_CEILING = 8_000_000;

/** The most gaussians this device's views may grow to: the budget itself on a handheld. */
export function deviceSplatCeiling(): number {
  const budget = deviceSplatBudget();
  return isHandheld() ? budget : Math.min(budget * DESKTOP_GROWTH, MAX_SPLAT_CEILING);
}

export function isHandheld(): boolean {
  try {
    const coarse = window.matchMedia("(pointer: coarse)").matches;
    return coarse && Math.min(window.screen.width, window.screen.height) < 900;
  } catch {
    return false;
  }
}

/**
 * The gaussians this device draws at once. The Detail choice was made for phones (400k at
 * Standard); a desktop's GPU draws several times that -- the web splat viewers show millions
 * -- and holding a large scan to a phone's budget is what kept a close look blurry. So a
 * desktop scales its own budget by the same choice: Light, Standard and Full are half, one
 * and two times its Standard.
 */
export function deviceSplatBudget(): number {
  const detail = splatBudget();
  if (isHandheld()) return detail;
  const memory =
    (typeof navigator !== "undefined"
      ? (navigator as Navigator & { deviceMemory?: number }).deviceMemory
      : undefined) ?? 8;
  const standard = DESKTOP_BUDGETS.find(([minGb]) => memory >= minGb)?.[1] ?? 800_000;
  return Math.round(standard * (detail / DEFAULT_SPLAT_BUDGET));
}
