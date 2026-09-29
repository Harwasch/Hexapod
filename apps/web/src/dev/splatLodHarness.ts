/**
 * A bare CesiumJS page that shows a REPLACE splat tileset with merged parents from a few
 * distances, next to the same splat as one full-resolution tile -- the driver for the
 * level-of-detail end-to-end check (e2e/splatLod.spec.ts).
 *
 * What it answers, in a real engine: does CesiumJS 1.145 draw a merged-parent REPLACE tileset
 * (tools/captures/splat_tiles.py) as the tiles it should -- the root alone far away, the
 * leaves close up -- and does each of those views cover the screen where the full splat
 * does, with no holes where a parent was dropped before its children arrived or where merged
 * coarse splats fail to fill what they stand for?
 *
 * With `shUrl`, it also answers whether CesiumJS draws a tileset's spherical harmonics
 * (splat_tiles.py packs a trained PLY's `f_rest_*` into every tile): the degree each drawn
 * tile and the primitive ended at, whether the SH texture was built, and the splat's mean
 * colour from a given heading -- which moves with the heading only if the bands are evaluated.
 *
 * Loaded dynamically by the spec; nothing imports it, so it never reaches the production
 * bundle. Headless GL is SwiftShader: coverage is a fair measure there, looks are not.
 */

import {
  Cesium3DTileset,
  CesiumWidget,
  Color,
  HeadingPitchRange,
  Math as CesiumMath,
  type Cesium3DTile,
  type Scene,
} from "cesium";

import { installSplatDecoder } from "@/cesium/splatDecoder";
import { incrementalSplats, splatTilesetOf } from "@/cesium/splatInternals";

/** The incremental state the harness waits on (patched GaussianSplatPrimitive). */
interface IncrementalProbe {
  _incremental?: {
    batch?: unknown;
    pendingRemovals: unknown[];
    sortedGeneration: number;
  };
  _splatDataGeneration?: number;
  _tileSlots?: ReadonlyMap<unknown, unknown>;
}

export interface SplatLodHarnessOptions {
  readonly container: HTMLElement;
  /** The level-of-detail tileset (REPLACE, merged parents). */
  readonly lodUrl: string;
  /** The same gaussians as one tile: what "no holes" is measured against. */
  readonly fullUrl: string;
  /** Run the level-of-detail tileset's primitive in incremental mode (patched engine). */
  readonly incremental?: boolean;
  /** The level-of-detail tileset again, packed with spherical harmonics. */
  readonly shUrl?: string;
  /** Decode SPZ in the app's workers (splatDecoder.ts), as the globe does. */
  readonly workerDecode?: boolean;
}

export interface SplatLodView {
  /** Tiles drawn, by content uri, in the order Cesium reported them. */
  readonly tiles: string[];
  /** Gaussians in those tiles, by their `extras.gaussians`. */
  readonly gaussians: number;
  /** Share of the canvas that is not background. */
  readonly coverage: number;
}

export interface SplatShView extends SplatLodView {
  /** `content.sphericalHarmonicsDegree` of each drawn tile, in `tiles` order. */
  readonly tileDegrees: number[];
  /** The degree the primitive draws at (CesiumJS takes the first selected tile's). */
  readonly primitiveDegree: number;
  /** Whether the primitive built its SH texture. */
  readonly shTexture: boolean;
  /** Mean 0-255 RGB of the pixels that are not background. */
  readonly meanRgb: [number, number, number];
}

export interface SplatLodHarness {
  /** Frames the tileset from `rangeM` metres and settles; `full` shows the one-tile splat. */
  view(rangeM: number, full: boolean): Promise<SplatLodView>;
  /**
   * Frames the SH tileset (`sh`) or the level-of-detail one (`lod`, no SH) looking along
   * `headingDeg` (0 north, 90 east) from `rangeM` metres, and settles.
   */
  shView(headingDeg: number, rangeM: number, which: "sh" | "lod"): Promise<SplatShView>;
}

/** The SH internals read here (CesiumJS 1.145 GaussianSplatPrimitive.js; not in Cesium.d.ts). */
interface ShPrimitive {
  readonly _sphericalHarmonicsDegree?: number;
  readonly sphericalHarmonicsTexture?: unknown;
}

const BACKGROUND = "#10141a";

function nextFrame(scene: Scene): Promise<void> {
  return new Promise((resolve) => {
    const remove = scene.postRender.addEventListener(() => {
      remove();
      resolve();
    });
    scene.requestRender();
  });
}

export async function startSplatLodHarness(
  options: SplatLodHarnessOptions,
): Promise<SplatLodHarness> {
  const widget = new CesiumWidget(options.container, {
    baseLayer: false,
    requestRenderMode: false,
    msaaSamples: 1,
    // So the frame can be read back after it is presented, for the coverage count.
    contextOptions: { webgl: { preserveDrawingBuffer: true } },
  });
  const { scene } = widget;
  if (options.workerDecode) installSplatDecoder();
  scene.globe.show = false;
  if (scene.skyBox) scene.skyBox.show = false;
  if (scene.skyAtmosphere) scene.skyAtmosphere.show = false;
  if (scene.sun) scene.sun.show = false;
  scene.backgroundColor = Color.fromCssColorString(BACKGROUND);

  // Cesium's default error, and the traversal site tilesets use for splats (providers/tiles.ts
  // keeps skipLevelOfDetail off for them): the base traversal quoted in splat_tiles.convert.
  const settings = { maximumScreenSpaceError: 16, skipLevelOfDetail: false };
  const lod = await Cesium3DTileset.fromUrl(options.lodUrl, settings);
  if (options.incremental) incrementalSplats(lod, 0);
  const full = await Cesium3DTileset.fromUrl(options.fullUrl, settings);
  const sh = options.shUrl ? await Cesium3DTileset.fromUrl(options.shUrl, settings) : undefined;
  const tilesets = [lod, full, ...(sh ? [sh] : [])];

  let drawn: Cesium3DTile[] = [];
  const seen: Cesium3DTile[] = [];
  const collect = (tile: Cesium3DTile): void => {
    seen.push(tile);
  };
  for (const tileset of tilesets) {
    tileset.show = false;
    scene.primitives.add(tileset);
    tileset.tileVisible.addEventListener(collect);
  }
  scene.postRender.addEventListener(() => {
    drawn = seen.splice(0);
  });

  const counted = (tile: Cesium3DTile): number => {
    const value = (tile.extras as { gaussians?: unknown } | undefined)?.gaussians;
    return typeof value === "number" ? value : 0;
  };

  const uriOf = (tile: Cesium3DTile): string => {
    const url = (tile.content as { url?: string }).url ?? "";
    return url.slice(url.lastIndexOf("/") + 1).split("?")[0] ?? url;
  };

  /** Share of the canvas that is not background, and the mean colour of what is. */
  function pixels(): { coverage: number; meanRgb: [number, number, number] } {
    const canvas = scene.canvas;
    const copy = document.createElement("canvas");
    copy.width = canvas.width;
    copy.height = canvas.height;
    const context = copy.getContext("2d");
    if (!context) return { coverage: 0, meanRgb: [0, 0, 0] };
    context.drawImage(canvas, 0, 0);
    const data = context.getImageData(0, 0, copy.width, copy.height).data;
    const background = Color.fromCssColorString(BACKGROUND);
    const [r, g, b] = [background.red, background.green, background.blue].map((c) => c * 255);
    let covered = 0;
    let [sumR, sumG, sumB] = [0, 0, 0];
    for (let i = 0; i < data.length; i += 4) {
      const [pr, pg, pb] = [data[i] ?? 0, data[i + 1] ?? 0, data[i + 2] ?? 0];
      const d = Math.abs(pr - (r ?? 0)) + Math.abs(pg - (g ?? 0)) + Math.abs(pb - (b ?? 0));
      if (d > 24) {
        covered += 1;
        sumR += pr;
        sumG += pg;
        sumB += pb;
      }
    }
    const n = Math.max(covered, 1);
    return { coverage: covered / (data.length / 4), meanRgb: [sumR / n, sumG / n, sumB / n] };
  }

  /** Shows `tileset` alone, looking along `headingDeg`, and waits until it has settled. */
  async function settle(
    tileset: Cesium3DTileset,
    headingDeg: number,
    rangeM: number,
  ): Promise<Cesium3DTile[]> {
    for (const each of tilesets) each.show = each === tileset;
    scene.camera.lookAt(
      tileset.boundingSphere.center,
      new HeadingPitchRange(CesiumMath.toRadians(headingDeg), CesiumMath.toRadians(-12), rangeM),
    );
    // Settled: every selected tile loaded, the primitive rebuilt over exactly them, and the
    // selection unchanged for a few frames (the primitive waits two stable frames before a
    // rebuild, GaussianSplatPrimitive.js DEFAULT_STABLE_FRAMES, then sorts in a worker).
    let key = "";
    let stable = 0;
    for (let frame = 0; frame < 400 && stable < 6; frame += 1) {
      await nextFrame(scene);
      const next = drawn.map(uriOf).join(" ");
      const primitive = splatTilesetOf(tileset).gaussianSplatPrimitive;
      const splats = primitive?._numSplats ?? -1;
      const wanted = drawn.reduce((sum, tile) => sum + counted(tile), 0);
      // Incremental: the drawn tiles hold exactly the slots, nothing is in flight or waiting
      // to be zeroed, and the sort on screen is of the latest slots.
      const probe = primitive as IncrementalProbe | undefined;
      const inc = probe?._incremental;
      const settled =
        inc === undefined
          ? splats === wanted
          : inc.batch === undefined &&
            inc.pendingRemovals.length === 0 &&
            inc.sortedGeneration >= (probe?._splatDataGeneration ?? 0) &&
            probe?._tileSlots?.size === drawn.length &&
            drawn.every((tile) => probe._tileSlots?.has(tile));
      const ready = tileset.tilesLoaded && drawn.length > 0 && settled;
      stable = ready && next === key ? stable + 1 : 0;
      key = next;
    }
    await nextFrame(scene);
    return drawn;
  }

  return {
    async view(rangeM: number, showFull: boolean): Promise<SplatLodView> {
      const tiles = await settle(showFull ? full : lod, 35, rangeM);
      return {
        tiles: tiles.map(uriOf),
        gaussians: tiles.reduce((sum, tile) => sum + counted(tile), 0),
        coverage: pixels().coverage,
      };
    },
    async shView(headingDeg: number, rangeM: number, which: "sh" | "lod"): Promise<SplatShView> {
      const tileset = which === "sh" ? sh : lod;
      if (!tileset) throw new Error("The harness was started without shUrl.");
      const tiles = await settle(tileset, headingDeg, rangeM);
      const primitive = splatTilesetOf(tileset).gaussianSplatPrimitive as ShPrimitive | undefined;
      const { coverage, meanRgb } = pixels();
      return {
        tiles: tiles.map(uriOf),
        gaussians: tiles.reduce((sum, tile) => sum + counted(tile), 0),
        coverage,
        tileDegrees: tiles.map(
          (tile) =>
            (tile.content as { sphericalHarmonicsDegree?: number }).sphericalHarmonicsDegree ?? -1,
        ),
        primitiveDegree: primitive?._sphericalHarmonicsDegree ?? -1,
        shTexture: primitive?.sphericalHarmonicsTexture !== undefined,
        meanRgb,
      };
    },
  };
}
