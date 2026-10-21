/**
 * One splat model's Confidence / Data age view (`splatCellView.ts`), driven by the view the
 * person chose (state/viewMode.ts). Loaded on demand by `cellViewSlot.ts` the first time a view
 * other than Photo is chosen, so the globe's first load carries none of it.
 *
 * The grid is fetched once, on the first analytic view; the part joins the primitive's colour
 * chain while a view is on and leaves it on Photo (the chain is rebuilt either way). Data age's
 * texture is built when Data age is first asked for, and again on the next `sync` once the
 * site's dates have changed (`cellViewSlot.ts` calls it on the scene's `site-dates` event, so a
 * date published while Data age is on counts at once) or the day has turned over.
 * Every change asks the scene for a render: the app renders on request.
 *
 * What the view found is reported per model (`useViewMode.report`): not measured (no grid on
 * the root), failed (and why), or ready, with Data age's bucket and the later dates compared.
 *
 * The same view is handed to PlayCanvas or Spark when one of them draws the scan
 * (scanView/scanCellView.ts): the mode drawn, its texture and the grid, or nothing on Photo --
 * and nothing for Data age until its texture is built, as here. CesiumJS's own copy of the scan
 * is hidden then; its part costs nothing there.
 */

import type { Cesium3DTileset, Scene } from "cesium";

import { cellGridMeta, DEFAULT_WEAK_BELOW, loadCellGrid, type CellGrid } from "@/lib/cellGrid";
import {
  ageBucket,
  comparedGrids,
  dataAgeTexels,
  laterDates,
  loadLaterGrids,
  type DataAgeContext,
} from "@/lib/dataAge";
import { observedAtFromUpload } from "@/lib/epochs";
import { createLogger } from "@/lib/log";
import { useViewMode, type CellViewReport, type ViewMode } from "@/state/viewMode";

import {
  cesiumCellViewGpu,
  SplatCellView,
  type CellViewGpu,
  type CellViewPrimitive,
} from "./splatCellView";
import type { Mat4 } from "./splatFrames";
import { publishScanCellView } from "./scanView/scanCellView";
import { splatTilesetOf } from "./splatInternals";

const log = createLogger("cellView");

export interface CellViewHandle {
  sync(mode: ViewMode): void;
  dispose(): void;
}

interface Tile {
  computedTransform?: Mat4;
  transform?: Mat4;
  extras?: unknown;
}

function asArray(m: Mat4 | undefined): number[] | undefined {
  return m ? Array.from({ length: 16 }, (_, i) => m[i] ?? 0) : undefined;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function startCellView(
  tileset: Cesium3DTileset,
  scene: Pick<Scene, "requestRender">,
  assetId: string,
  context: () => DataAgeContext,
  options: { gpu?: CellViewGpu | undefined; now?: () => number } = {},
): CellViewHandle {
  const report = useViewMode.getState().report;
  const root = (): Tile | undefined => tileset.root;
  const meta = cellGridMeta(root()?.extras, "confidence");
  const url = (tileset as unknown as { resource?: { url?: string } }).resource?.url;
  const gpu = "gpu" in options ? options.gpu : cesiumCellViewGpu();
  const now = options.now ?? Date.now;
  let mode: ViewMode = "photo";
  let disposed = false;
  let grid: Promise<CellGrid> | undefined;
  let part: SplatCellView | undefined;
  let ageKey: string | undefined;
  let age: Promise<CellViewReport> | undefined;

  const primitive = (): CellViewPrimitive | undefined =>
    splatTilesetOf(tileset).gaussianSplatPrimitive;

  /** What a dedicated renderer drawing the scan draws now. */
  const publish = (): void => {
    publishScanCellView(
      assetId,
      !disposed && part && mode !== "photo" ? part.lookView(mode) : null,
    );
  };

  /** Puts the part where the view says, on the primitive there is now. */
  const place = (): void => {
    if (disposed || !part) return;
    publish();
    const target = primitive();
    if (mode === "photo") {
      if (part.installed) {
        part.uninstall();
        scene.requestRender();
      }
      return;
    }
    part.mode = mode;
    if (target && !part.installed && part.install(target)) scene.requestRender();
  };

  const ageKeyOf = (c: DataAgeContext): string =>
    JSON.stringify([
      c.epochs.map((e) => [e.id, e.status, e.observedAt, e.currentAssetId]),
      c.assets.map((a) => [a.id, a.epochId, a.validTo]),
      // A bucket boundary may pass while the page is open; one a day is plenty.
      Math.floor(now() / 86_400_000),
    ]);

  /** Data age's texture for the dates the site has now, and what it found. */
  const buildAge = (g: CellGrid): Promise<CellViewReport> => {
    const c = context();
    const key = ageKeyOf(c);
    if (age && key === ageKey) return age;
    ageKey = key;
    const representation =
      c.assets.find((a) => a.id === assetId)?.representation ?? "gaussian-splat";
    const { shown, later } = laterDates(c, assetId, representation);
    const bucket = ageBucket(shown?.observedAt, now());
    age = loadLaterGrids(later, asArray(root()?.transform)).then((grids): CellViewReport => {
      const { compared, skipped } = comparedGrids(grids, g);
      const { rg, newerCells } = dataAgeTexels(g, bucket, compared);
      if (!disposed && key === ageKey) {
        part?.setDataAge(rg);
        publish();
      }
      return {
        kind: "ready",
        mode: "data-age",
        ageBucket: bucket,
        ...(shown && observedAtFromUpload(shown) ? { ageFromUpload: true } : {}),
        newer: { dates: later.length, compared: compared.length, cells: newerCells, skipped },
      };
    });
    return age;
  };

  const sync = (next: ViewMode): void => {
    mode = next;
    if (disposed) return;
    if (part && next !== "photo") part.mode = next;
    if (next === "photo" || part) {
      // Photo, or a mode the part already holds the bytes for: uniforms only.
      place();
      scene.requestRender();
      if (next === "photo") return;
    }
    if (!meta || !url) {
      report(assetId, { kind: "not-measured" });
      return;
    }
    if (!gpu) {
      report(assetId, { kind: "failed", message: "this CesiumJS build cannot make the texture" });
      return;
    }
    if (!part || (next === "data-age" && !part.ready("data-age"))) {
      report(assetId, { kind: "loading" });
    }
    grid ??= loadCellGrid(url, meta);
    grid
      .then(async (g) => {
        if (disposed || mode !== next) return;
        part ??= new SplatCellView(g, gpu, () => root()?.computedTransform);
        part.mode = next;
        // Installed now; Data age draws the photo until its texture is here.
        place();
        scene.requestRender();
        if (next === "confidence") {
          report(assetId, {
            kind: "ready",
            mode: "confidence",
            weakBelow: g.meta.weakBelow ?? DEFAULT_WEAK_BELOW,
          });
          return;
        }
        const building = buildAge(g);
        const found = await building;
        // A comparison a newer one overtook (the site's dates changed meanwhile) says
        // nothing: its texture was not applied, and its legend would describe other dates.
        if (disposed || mode !== next || building !== age) return;
        report(assetId, found);
        scene.requestRender();
      })
      .catch((error: unknown) => {
        if (disposed) return;
        log.warn("cell view did not load", { asset: assetId, message: message(error) });
        report(assetId, { kind: "failed", message: message(error) });
      });
  };

  // The primitive appears with the first tile, and an unloaded tileset's comes back with one.
  const unsubscribe = tileset.tileLoad.addEventListener(place);
  return {
    sync,
    dispose: () => {
      disposed = true;
      unsubscribe();
      publish();
      part?.destroy();
      part = undefined;
      report(assetId, null);
    },
  };
}
