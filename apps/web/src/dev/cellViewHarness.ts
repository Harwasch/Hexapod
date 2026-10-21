/**
 * A bare CesiumJS page that draws the synthetic yard through the Confidence / Data age colour
 * part -- the driver for e2e/confidenceViewScene.spec.ts.
 *
 * What it answers, in a real engine with the patch: does `splatCellViewColor` compile into the
 * splat vertex shader, read each splat's cell through the root frame, and tint weak cells red?
 * The grids are built here by the TypeScript fixture encoder (`cellGridFixture.ts`): every cell
 * weak, every cell strong, or weak west of the yard's middle and strong east of it, so that
 * looking north the left of the screen must turn red and the right green.
 *
 * Loaded dynamically by the spec; nothing imports it, so it never reaches the production
 * bundle. Headless GL is SwiftShader: pixels are counted, not eyeballed.
 */

import {
  Cesium3DTileset,
  CesiumWidget,
  Color,
  HeadingPitchRange,
  Math as CesiumMath,
  type Scene,
} from "cesium";

import {
  cesiumCellViewGpu,
  SplatCellView,
  type CellViewMode,
  type CellViewPrimitive,
} from "@/cesium/splatCellView";
import { incrementalSplats, splatTilesetOf } from "@/cesium/splatInternals";
import { decodeCellGrid } from "@/lib/cellGrid";
import { dataAgeTexels } from "@/lib/dataAge";

import { encodeConfidenceGrid, REASON_BITS } from "./cellGridFixture";

export type GridName = "weak" | "strong" | "split";

export interface PixelCounts {
  /** Share of the canvas the scan covers. */
  coverage: number;
  /** Of the covered pixels: red-dominant, green-dominant (whole view, left and right halves). */
  red: number;
  green: number;
  leftRed: number;
  rightRed: number;
  leftGreen: number;
  rightGreen: number;
  /**
   * Of the covered pixels: saturated ones whose hue is the 3-6 months bucket's (230-275°, the
   * age ramp's blue-violet), and light, washed-out ones (the "newer data here" stripes).
   */
  ageHue: number;
  light: number;
}

export interface CellViewHarness {
  view(grid: GridName | "off", mode?: CellViewMode): Promise<PixelCounts>;
  installed(): boolean;
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

export async function startCellViewHarness(options: {
  container: HTMLElement;
  url: string;
}): Promise<CellViewHarness> {
  const widget = new CesiumWidget(options.container, {
    baseLayer: false,
    requestRenderMode: true,
    msaaSamples: 1,
    contextOptions: { webgl: { preserveDrawingBuffer: true } },
  });
  const { scene } = widget;
  scene.globe.show = false;
  if (scene.skyBox) scene.skyBox.show = false;
  if (scene.skyAtmosphere) scene.skyAtmosphere.show = false;
  if (scene.sun) scene.sun.show = false;
  scene.backgroundColor = Color.fromCssColorString(BACKGROUND);

  const tileset = await Cesium3DTileset.fromUrl(options.url, {
    maximumScreenSpaceError: 4,
    skipLevelOfDetail: false,
  });
  // As the app draws a site's splats (providers/tiles.ts): one slot range per tile.
  incrementalSplats(tileset, 0);
  scene.primitives.add(tileset);
  const gpu = cesiumCellViewGpu();
  if (!gpu) throw new Error("no RG8 texture support");

  // The yard: 32 x 24 m east/north of its frame's origin. 1 m cells, a margin of 2.
  const build = async (name: GridName) => {
    const { meta, planes } = await encodeConfidenceGrid({
      level: 2,
      i0: -2,
      j0: -2,
      ni: 36,
      nj: 28,
      cell: (i) =>
        name === "weak" || (name === "split" && i < 16)
          ? { score: 60, reasons: REASON_BITS.views }
          : { score: 240 },
      coarser: 8,
    });
    return decodeCellGrid(meta, planes);
  };
  const root = () =>
    (tileset.root as unknown as { computedTransform?: ArrayLike<number> }).computedTransform;
  const parts: Record<GridName, SplatCellView> = {
    weak: new SplatCellView(await build("weak"), gpu, root),
    strong: new SplatCellView(await build("strong"), gpu, root),
    split: new SplatCellView(await build("split"), gpu, root),
  };
  // Data age, four months old (bucket 2): on the split grid with a later date's evidence
  // everywhere (striped), on the weak one with no later date (not striped).
  const later = await build("strong");
  parts.split.setDataAge(dataAgeTexels(parts.split.grid, 2, [later]).rg);
  parts.weak.setDataAge(dataAgeTexels(parts.weak.grid, 2, []).rg);
  const primitive = (): CellViewPrimitive | undefined =>
    splatTilesetOf(tileset).gaussianSplatPrimitive;

  function count(): PixelCounts {
    const canvas = scene.canvas;
    const copy = document.createElement("canvas");
    copy.width = canvas.width;
    copy.height = canvas.height;
    const context = copy.getContext("2d");
    const empty = {
      coverage: 0,
      red: 0,
      green: 0,
      leftRed: 0,
      rightRed: 0,
      leftGreen: 0,
      rightGreen: 0,
      ageHue: 0,
      light: 0,
    };
    if (!context) return empty;
    context.drawImage(canvas, 0, 0);
    const data = context.getImageData(0, 0, copy.width, copy.height).data;
    const bg = Color.fromCssColorString(BACKGROUND);
    const [br, bgG, bb] = [bg.red * 255, bg.green * 255, bg.blue * 255];
    const half = copy.width / 2;
    const n = {
      covered: 0,
      red: 0,
      green: 0,
      left: 0,
      right: 0,
      lr: 0,
      rr: 0,
      lg: 0,
      rg: 0,
      ageHue: 0,
      light: 0,
    };
    for (let p = 0; p < data.length; p += 4) {
      const r = data[p] ?? 0;
      const g = data[p + 1] ?? 0;
      const b = data[p + 2] ?? 0;
      if (Math.abs(r - br) + Math.abs(g - bgG) + Math.abs(b - bb) <= 24) continue;
      const left = (p / 4) % copy.width < half;
      n.covered += 1;
      if (left) n.left += 1;
      else n.right += 1;
      const red = r > 40 && r > 1.6 * Math.max(g, b);
      const green = g > 40 && g > 1.25 * Math.max(r, b);
      if (red) {
        n.red += 1;
        if (left) n.lr += 1;
        else n.rr += 1;
      }
      if (green) {
        n.green += 1;
        if (left) n.lg += 1;
        else n.rg += 1;
      }
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      const saturation = max > 0 ? (max - min) / max : 0;
      if (max > 140 && saturation < 0.3) n.light += 1;
      if (saturation > 0.25 && max === b) {
        const hue = 240 + (60 * (r - g)) / (max - min);
        if (hue >= 230 && hue <= 275) n.ageHue += 1;
      }
    }
    const share = (a: number, b: number) => (b > 0 ? a / b : 0);
    return {
      coverage: n.covered / (data.length / 4),
      red: share(n.red, n.covered),
      green: share(n.green, n.covered),
      leftRed: share(n.lr, n.left),
      rightRed: share(n.rr, n.right),
      leftGreen: share(n.lg, n.left),
      rightGreen: share(n.rg, n.right),
      ageHue: share(n.ageHue, n.covered),
      light: share(n.light, n.covered),
    };
  }

  let shown: SplatCellView | undefined;
  return {
    async view(grid, mode = "confidence") {
      // Looking north, nearly straight down: east is the right of the screen.
      scene.camera.lookAt(
        tileset.boundingSphere.center,
        new HeadingPitchRange(0, CesiumMath.toRadians(-85), 48),
      );
      for (let frame = 0; frame < 600 && !(tileset.tilesLoaded && primitive()); frame += 1) {
        await nextFrame(scene);
      }
      const target = primitive();
      if (!target) throw new Error("no splat primitive");
      for (const part of Object.values(parts)) part.uninstall();
      shown = undefined;
      if (grid !== "off") {
        shown = parts[grid];
        shown.mode = mode;
        if (!shown.install(target)) throw new Error("the engine refused the colour part");
      }
      // The draw command is rebuilt on the next render; the sort settles over a few more.
      for (let frame = 0; frame < 30; frame += 1) await nextFrame(scene);
      return count();
    },
    installed() {
      return Object.values(parts).some((p) => p.installed);
    },
  };
}
