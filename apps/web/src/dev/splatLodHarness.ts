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

import { splatTilesetOf } from "@/cesium/splatInternals";

export interface SplatLodHarnessOptions {
  readonly container: HTMLElement;
  /** The level-of-detail tileset (REPLACE, merged parents). */
  readonly lodUrl: string;
  /** The same gaussians as one tile: what "no holes" is measured against. */
  readonly fullUrl: string;
}

export interface SplatLodView {
  /** Tiles drawn, by content uri, in the order Cesium reported them. */
  readonly tiles: string[];
  /** Gaussians in those tiles, by their `extras.gaussians`. */
  readonly gaussians: number;
  /** Share of the canvas that is not background. */
  readonly coverage: number;
}

export interface SplatLodHarness {
  /** Frames the tileset from `rangeM` metres and settles; `full` shows the one-tile splat. */
  view(rangeM: number, full: boolean): Promise<SplatLodView>;
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
  scene.globe.show = false;
  if (scene.skyBox) scene.skyBox.show = false;
  if (scene.skyAtmosphere) scene.skyAtmosphere.show = false;
  if (scene.sun) scene.sun.show = false;
  scene.backgroundColor = Color.fromCssColorString(BACKGROUND);

  // Cesium's default error, and the traversal site tilesets use for splats (providers/tiles.ts
  // keeps skipLevelOfDetail off for them): the base traversal quoted in splat_tiles.convert.
  const settings = { maximumScreenSpaceError: 16, skipLevelOfDetail: false };
  const lod = await Cesium3DTileset.fromUrl(options.lodUrl, settings);
  const full = await Cesium3DTileset.fromUrl(options.fullUrl, settings);
  scene.primitives.add(lod);
  scene.primitives.add(full);

  let drawn: Cesium3DTile[] = [];
  const seen: Cesium3DTile[] = [];
  const collect = (tile: Cesium3DTile): void => {
    seen.push(tile);
  };
  lod.tileVisible.addEventListener(collect);
  full.tileVisible.addEventListener(collect);
  scene.postRender.addEventListener(() => {
    drawn = seen.splice(0);
  });

  const counted = (tile: Cesium3DTile): number => {
    const value = (tile.extras as { gaussians?: unknown } | undefined)?.gaussians;
    return typeof value === "number" ? value : 0;
  };

  function coverage(): number {
    const canvas = scene.canvas;
    const copy = document.createElement("canvas");
    copy.width = canvas.width;
    copy.height = canvas.height;
    const context = copy.getContext("2d");
    if (!context) return 0;
    context.drawImage(canvas, 0, 0);
    const pixels = context.getImageData(0, 0, copy.width, copy.height).data;
    const background = Color.fromCssColorString(BACKGROUND);
    const [r, g, b] = [background.red, background.green, background.blue].map((c) => c * 255);
    let covered = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      const d =
        Math.abs((pixels[i] ?? 0) - (r ?? 0)) +
        Math.abs((pixels[i + 1] ?? 0) - (g ?? 0)) +
        Math.abs((pixels[i + 2] ?? 0) - (b ?? 0));
      if (d > 24) covered += 1;
    }
    return covered / (pixels.length / 4);
  }

  return {
    async view(rangeM: number, showFull: boolean): Promise<SplatLodView> {
      lod.show = !showFull;
      full.show = showFull;
      const tileset = showFull ? full : lod;
      const bounds = tileset.boundingSphere;
      scene.camera.lookAt(
        bounds.center,
        new HeadingPitchRange(CesiumMath.toRadians(35), CesiumMath.toRadians(-12), rangeM),
      );
      // Settled: every selected tile loaded, the primitive rebuilt over exactly them, and the
      // selection unchanged for a few frames (the primitive waits two stable frames before a
      // rebuild, GaussianSplatPrimitive.js DEFAULT_STABLE_FRAMES, then sorts in a worker).
      let key = "";
      let stable = 0;
      for (let frame = 0; frame < 400 && stable < 6; frame += 1) {
        await nextFrame(scene);
        const uris = drawn.map((tile) => (tile.content as { url?: string }).url ?? "");
        const next = uris.join(" ");
        const splats = splatTilesetOf(tileset).gaussianSplatPrimitive?._numSplats ?? -1;
        const wanted = drawn.reduce((sum, tile) => sum + counted(tile), 0);
        const ready = tileset.tilesLoaded && drawn.length > 0 && splats === wanted;
        stable = ready && next === key ? stable + 1 : 0;
        key = next;
      }
      await nextFrame(scene);
      return {
        tiles: drawn.map((tile) => {
          const url = (tile.content as { url?: string }).url ?? "";
          return url.slice(url.lastIndexOf("/") + 1).split("?")[0] ?? url;
        }),
        gaussians: drawn.reduce((sum, tile) => sum + counted(tile), 0),
        coverage: coverage(),
      };
    },
  };
}
