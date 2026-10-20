/**
 * Data age for one dated capture (epoch): how old what it shows is, per cell it observed, and
 * where a later date of the same site has evidence of its own (design M4-6, W4a). Cesium-free.
 *
 * One epoch has one date, so its observed cells all share one age bucket: the view says how old
 * this date is where it saw anything, honestly, rather than inventing a per-cell age. What it
 * adds is "newer data here": the later dates' confidence grids (each published epoch's current
 * tileset, its root `extras.confidence`, fetched directly; no API) are compared cell by cell on
 * the one site lattice (C1), and the cells a later date scored are striped.
 *
 * A later date is compared only when its tileset is in the same frame as the shown one (the
 * same root transform: one site frame, D3). One placed in another frame, one without a grid, or
 * one that could not be read is left out and named, never guessed at.
 */

import type { Epoch, Representation, SiteAsset } from "@twin/contracts";

import {
  cellGridMeta,
  cellTexelLayout,
  channelAt,
  evidenceAt,
  loadCellGrid,
  type CellGrid,
} from "./cellGrid";
import { epochOptions, formatEpochDate } from "./epochs";

const DAY_MS = 86_400_000;

/** Age buckets, freshest first: the Data age legend and the ramp (`splatCellView.ts`). */
export const AGE_BUCKETS: readonly { label: string; underDays: number }[] = [
  { label: "Under a month", underDays: 30 },
  { label: "1–3 months", underDays: 91 },
  { label: "3–6 months", underDays: 182 },
  { label: "6–12 months", underDays: 365 },
  { label: "1–2 years", underDays: 730 },
  { label: "Over 2 years", underDays: Number.POSITIVE_INFINITY },
];

/** The bucket of data observed at `observedAt`, seen at `now`; null without a usable date. */
export function ageBucket(observedAt: string | null | undefined, now: number): number | null {
  const at = observedAt ? Date.parse(observedAt) : Number.NaN;
  if (!Number.isFinite(at)) return null;
  const days = Math.max(0, now - at) / DAY_MS;
  const index = AGE_BUCKETS.findIndex((b) => days < b.underDays);
  return index < 0 ? AGE_BUCKETS.length - 1 : index;
}

/** What the scene knows of a site's dates, for one of its models. */
export interface DataAgeContext {
  epochs: readonly Epoch[];
  assets: readonly SiteAsset[];
}

/** The later published dates of the site than `assetId`'s, with their current models. */
export function laterDates(
  context: DataAgeContext,
  assetId: string,
  representation: Representation,
): { shown: Epoch | null; later: { epoch: Epoch; asset: SiteAsset }[] } {
  const asset = context.assets.find((a) => a.id === assetId);
  const shown = context.epochs.find((e) => e.id === asset?.epochId) ?? null;
  if (!shown) return { shown: null, later: [] };
  const at = Date.parse(shown.observedAt);
  const later = epochOptions(context.epochs, context.assets, representation).filter(
    (o) => o.epoch.id !== shown.id && Date.parse(o.epoch.observedAt) > at,
  );
  return { shown, later };
}

/** Whether two root transforms are one frame (rotation to 1e-9, translation to a millimetre). */
export function sameFrame(a: ArrayLike<number> | undefined, b: ArrayLike<number> | undefined) {
  if (!a || !b || a.length !== 16 || b.length !== 16) return a === b;
  for (let i = 0; i < 16; i += 1) {
    const tolerance = i >= 12 && i <= 14 ? 1e-3 : 1e-9;
    if (Math.abs((a[i] ?? 0) - (b[i] ?? 0)) > tolerance) return false;
  }
  return true;
}

interface TilesetJson {
  root?: { transform?: number[]; extras?: unknown };
}

/** A later date's grid, or why it is left out. */
export type LaterGrid = { epoch: Epoch; grid: CellGrid } | { epoch: Epoch; skipped: string };

/**
 * Fetches each later date's tileset and confidence grid. `shownTransform` is the shown model's
 * root transform as its tileset.json declares it (`undefined` when it declares none).
 */
export async function loadLaterGrids(
  later: readonly { epoch: Epoch; asset: SiteAsset }[],
  shownTransform: readonly number[] | undefined,
  fetchJson: (url: string) => Promise<unknown> = async (url) => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`answered ${String(response.status)}`);
    return response.json();
  },
): Promise<LaterGrid[]> {
  return Promise.all(
    later.map(async ({ epoch, asset }): Promise<LaterGrid> => {
      const date = formatEpochDate(epoch.observedAt);
      if (asset.source.type !== "3d-tiles-url") {
        return { epoch, skipped: `${date}: not a tileset this page can read` };
      }
      const url = new URL(asset.source.url, globalThis.location?.href).toString();
      try {
        const tileset = (await fetchJson(url)) as TilesetJson;
        if (!sameFrame(tileset.root?.transform, shownTransform)) {
          return { epoch, skipped: `${date}: placed in another frame` };
        }
        const meta = cellGridMeta(tileset.root?.extras, "confidence");
        if (!meta) return { epoch, skipped: `${date}: not measured` };
        return { epoch, grid: await loadCellGrid(url, meta) };
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return { epoch, skipped: `${date}: could not be read (${message})` };
      }
    }),
  );
}

/**
 * The Data age texture's bytes for `grid` (the shown date's), in `cellTexelLayout` order:
 * R = 1 + `bucket` where the shown date scored the cell (0 elsewhere, or with no date), G =
 * 255 where a later grid has evidence under the cell. `newerCells` counts the finest level's.
 */
export function dataAgeTexels(
  grid: CellGrid,
  bucket: number | null,
  later: readonly CellGrid[],
): { rg: Uint8Array; newerCells: number } {
  const score = channelAt(grid.meta, "score");
  const { cells } = cellTexelLayout(grid.meta);
  const rg = new Uint8Array(cells * 2);
  let newerCells = 0;
  let k = 0;
  for (const [n, l] of grid.meta.levels.entries()) {
    for (let cj = 0; cj < l.nj; cj += 1) {
      for (let ci = 0; ci < l.ni; ci += 1) {
        const at = l.offset + (cj * l.ni + ci) * grid.stride;
        const scored = score ? (grid.planes[at + score.at] ?? 0) > 0 : false;
        rg[k] = scored && bucket !== null ? bucket + 1 : 0;
        const newer = later.some((g) => evidenceAt(g, l.level, l.i0 + ci, l.j0 + cj));
        rg[k + 1] = newer ? 255 : 0;
        if (newer && n === 0) newerCells += 1;
        k += 2;
      }
    }
  }
  return { rg, newerCells };
}
