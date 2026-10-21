/**
 * A bare CesiumJS page that draws the sunlight slab (`sunlightSlab.ts`) through the app's own
 * relight path -- the slot `attachSunlight` on the tileset, the stores it reads -- under
 * CesiumJS or PlayCanvas (the app's own scan renderer, ScanRendererHost): the driver for
 * e2e/sunlightScene.spec.ts.
 *
 * What it answers, in a real engine: does the relight compile -- `splatSunlightColor` into
 * CesiumJS's splat vertex shader (with `czm_sunDirectionEC` declared by the engine because the
 * text uses it), PlayCanvas's per-frame hook (cesium/scanView/sunHook.ts) into its draw
 * material -- find the discs' normals, and light the east-facing side by the sun of the scene
 * clock, brighter in the morning than in the evening, while the round splats beside it stay as
 * they are by day? With the sun off, or As captured on, is the picture the photograph's? And
 * with Confidence drawn over it (`cell`: every cell strong), is the look tinted by the relit
 * luminance under both -- CesiumJS's order, sun before look?
 *
 * The camera stands east of the wall looking west, so the discs (south) are the left of the
 * screen and the round splats (north) the right. Loaded dynamically by the spec; nothing
 * imports it, so it never reaches the production bundle. Headless GL is SwiftShader: pixels
 * are measured, not eyeballed.
 */

import {
  Cartesian3,
  Cesium3DTileset,
  CesiumWidget,
  Color,
  Ellipsoid,
  HeadingPitchRange,
  JulianDate,
  Math as CesiumMath,
  Matrix3,
  Simon1994PlanetaryPositions,
  Transforms,
  type Scene,
} from "cesium";

import { publishScanCellView } from "@/cesium/scanView/scanCellView";
import { ScanRendererHost } from "@/cesium/scanView/ScanRendererHost";
import { scanSunReading } from "@/cesium/scanView/scanSunlight";
import { cesiumCellViewGpu, SplatCellView } from "@/cesium/splatCellView";
import { colorChainOf, type ColorPrimitive } from "@/cesium/splatColor";
import { incrementalSplats, splatTilesetOf } from "@/cesium/splatInternals";
import { sunSine } from "@/cesium/splatSunlight";
import { attachSunlight } from "@/cesium/sunlightSlot";
import { decodeCellGrid } from "@/lib/cellGrid";
import { cellViewColor } from "@/lib/cellLook";
import { SUN_AMBIENT, sunUpFactor } from "@/lib/splatNormals";
import { useAsCaptured } from "@/state/asCaptured";
import { useLighting, type SunMode } from "@/state/lighting";
import { useSettings } from "@/state/settings";
import { useSunReport } from "@/state/sunReport";

import { encodeConfidenceGrid } from "./cellGridFixture";
import { SLAB_GREY } from "./sunlightSlab";

export interface SlabView {
  /** Share of the canvas the wall covers. */
  coverage: number;
  /** Mean brightness (0-255, the mean of r, g, b) of covered pixels: the discs, the round splats. */
  flat: number;
  round: number;
  /** What lib/splatNormals.ts says the discs are multiplied by, at this sun. */
  predictedFlat: number;
  /**
   * With Confidence drawn: what the discs' look is multiplied by, relit as CesiumJS orders it
   * (the look of the relit grey, over the look of the grey), and what it would be relit after
   * the look instead (`predictedFlat`).
   */
  predictedCellFlat: number;
  /** Sine of the sun's elevation at the wall, as the relight works it out. */
  sine: number;
  /**
   * The colour chain of the slab's primitive under CesiumJS; under PlayCanvas the frame's hook,
   * `sunHook`, while it is in the draw material (scanView/scanSunlight.ts's reading).
   */
  parts: string[];
  /** What the relight reported drawing, if anything. */
  reported: { at: string; sine: number } | null;
}

export interface SunlightHarness {
  view(options: {
    sun: SunMode;
    at: string;
    asCaptured?: boolean;
    /** Confidence drawn over the slab, every cell strong. */
    cell?: boolean;
  }): Promise<SlabView>;
}

/** Who draws the slab. */
export type SlabRenderer = "cesium" | "playcanvas";

const ASSET = "slab";
/** A strong cell's score: the top bin of the ramp, which has no dots to darken it. */
const STRONG = 240;

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

/** The sun's direction, Earth-fixed, by the engine's arithmetic (as `sunSine`). */
function sunDirection(time: JulianDate): Cartesian3 {
  const m = new Matrix3();
  Transforms.computeIcrfToCentralBodyFixedMatrix(time, m);
  const p = Simon1994PlanetaryPositions.computeSunPositionInEarthInertialFrame(time);
  return Cartesian3.normalize(Matrix3.multiplyByVector(m, p, p), p);
}

export async function startSunlightHarness(options: {
  container: HTMLElement;
  url: string;
  renderer?: SlabRenderer;
}): Promise<SunlightHarness> {
  const renderer = options.renderer ?? "cesium";
  const widget = new CesiumWidget(options.container, {
    baseLayer: false,
    requestRenderMode: true,
    msaaSamples: 1,
    contextOptions: { webgl: { preserveDrawingBuffer: true } },
  });
  const { scene, clock } = widget;
  scene.globe.show = false;
  if (scene.skyBox) scene.skyBox.show = false;
  if (scene.skyAtmosphere) scene.skyAtmosphere.show = false;
  if (scene.sun) scene.sun.show = false;
  scene.backgroundColor = Color.fromCssColorString(BACKGROUND);

  const tileset = await Cesium3DTileset.fromUrl(options.url, {
    maximumScreenSpaceError: 4,
    skipLevelOfDetail: false,
  });
  incrementalSplats(tileset, 0);
  scene.primitives.add(tileset);
  // The app's path: the slot, which loads the relight when the sun is on and As captured off,
  // under the renderer that draws this page -- so the setting says which.
  useSettings.getState().set({ splatRenderer: renderer });
  attachSunlight(tileset, scene, ASSET);
  const primitive = (): ColorPrimitive | undefined =>
    splatTilesetOf(tileset).gaussianSplatPrimitive;
  // PlayCanvas draws the slab as the app has it: CesiumJS's tileset hidden, the app's scan
  // renderer over the globe from Cesium's camera.
  const host =
    renderer === "playcanvas"
      ? new ScanRendererHost(widget, { preserveDrawingBuffer: true })
      : null;
  if (host) {
    tileset.show = false;
    host.setRenderer("playcanvas");
    host.setTarget({ key: ASSET, tileset, assetId: ASSET });
  }
  const overlay = (): HTMLCanvasElement | null =>
    document.querySelector<HTMLCanvasElement>("canvas[data-scan-renderer]");

  // Confidence over the slab, every cell strong: under CesiumJS the cell part in the chain,
  // under PlayCanvas the look in its copy pass (scanView/scanCellView.ts).
  let cells: SplatCellView | null = null;
  const confidence = async (on: boolean): Promise<void> => {
    if (on && !cells) {
      const gpu = cesiumCellViewGpu();
      if (!gpu) throw new Error("no RG8 texture support");
      // 1 m cells over the wall (east -2..2 m, north -4..4 m of the tile's frame).
      const { meta, planes } = await encodeConfidenceGrid({
        level: 2,
        i0: -2,
        j0: -4,
        ni: 4,
        nj: 8,
        cell: () => ({ score: STRONG }),
        coarser: 4,
      });
      const root = () =>
        (tileset.root as unknown as { computedTransform?: ArrayLike<number> }).computedTransform;
      cells = new SplatCellView(await decodeCellGrid(meta, planes), gpu, root);
    }
    if (host) {
      publishScanCellView(ASSET, on && cells ? cells.lookView("confidence") : null);
      return;
    }
    const target = primitive();
    if (on && cells && target) cells.install(target);
    if (!on) cells?.uninstall();
  };

  const settle = async (): Promise<void> => {
    if (!host) {
      for (let frame = 0; frame < 600 && !(tileset.tilesLoaded && primitive()); frame += 1) {
        await nextFrame(scene);
      }
      // The relight loads on first use; the draw command is rebuilt on the next render.
      for (let frame = 0; frame < 30; frame += 1) await nextFrame(scene);
      return;
    }
    let still = 0;
    for (let frame = 0; frame < 3000 && still < 30; frame += 1) {
      await nextFrame(scene);
      const status = host.status();
      const ready = status.active && status.tiles > 0 && status.loading === 0 && status.settled;
      still = ready ? still + 1 : 0;
    }
  };

  function measureOverlay(
    canvas: HTMLCanvasElement,
  ): Pick<SlabView, "coverage" | "flat" | "round"> {
    const copy = document.createElement("canvas");
    copy.width = canvas.width;
    copy.height = canvas.height;
    const context = copy.getContext("2d");
    if (!context) return { coverage: 0, flat: 0, round: 0 };
    context.drawImage(canvas, 0, 0);
    const data = context.getImageData(0, 0, copy.width, copy.height).data;
    const half = copy.width / 2;
    let covered = 0;
    const sum = { left: 0, right: 0 };
    const count = { left: 0, right: 0 };
    for (let p = 0; p < data.length; p += 4) {
      // The overlay is transparent where nothing is drawn: covered is a quarter opaque or more.
      if ((data[p + 3] ?? 0) < 64) continue;
      covered += 1;
      const side = (p / 4) % copy.width < half ? "left" : "right";
      sum[side] += ((data[p] ?? 0) + (data[p + 1] ?? 0) + (data[p + 2] ?? 0)) / 3;
      count[side] += 1;
    }
    return {
      coverage: covered / (data.length / 4),
      flat: count.left > 0 ? sum.left / count.left : 0,
      round: count.right > 0 ? sum.right / count.right : 0,
    };
  }

  function measure(): Pick<SlabView, "coverage" | "flat" | "round"> {
    const drawn = host ? overlay() : null;
    if (host) return drawn ? measureOverlay(drawn) : { coverage: 0, flat: 0, round: 0 };
    const canvas = scene.canvas;
    const copy = document.createElement("canvas");
    copy.width = canvas.width;
    copy.height = canvas.height;
    const context = copy.getContext("2d");
    if (!context) return { coverage: 0, flat: 0, round: 0 };
    context.drawImage(canvas, 0, 0);
    const data = context.getImageData(0, 0, copy.width, copy.height).data;
    const bg = Color.fromCssColorString(BACKGROUND);
    const [br, bgG, bb] = [bg.red * 255, bg.green * 255, bg.blue * 255];
    const half = copy.width / 2;
    let covered = 0;
    const sum = { left: 0, right: 0 };
    const count = { left: 0, right: 0 };
    for (let p = 0; p < data.length; p += 4) {
      const r = data[p] ?? 0;
      const g = data[p + 1] ?? 0;
      const b = data[p + 2] ?? 0;
      if (Math.abs(r - br) + Math.abs(g - bgG) + Math.abs(b - bb) <= 12) continue;
      covered += 1;
      const side = (p / 4) % copy.width < half ? "left" : "right";
      sum[side] += (r + g + b) / 3;
      count[side] += 1;
    }
    return {
      coverage: covered / (data.length / 4),
      flat: count.left > 0 ? sum.left / count.left : 0,
      round: count.right > 0 ? sum.right / count.right : 0,
    };
  }

  const parts = (): string[] => {
    if (host) return scanSunReading(ASSET)?.installed ? ["sunHook"] : [];
    const target = primitive();
    return (target ? (colorChainOf(target)?.parts ?? []) : []).map((p) => p.colorFunction);
  };

  return {
    async view({ sun, at, asCaptured = false, cell = false }) {
      useAsCaptured.getState().set(asCaptured);
      useLighting.getState().set(sun);
      const time = JulianDate.fromIso8601(at);
      clock.shouldAnimate = false;
      clock.currentTime = time;
      // East of the wall, looking west: the discs (south) left, the round splats right.
      scene.camera.lookAt(
        tileset.boundingSphere.center,
        new HeadingPitchRange(CesiumMath.toRadians(270), CesiumMath.toRadians(-4), 9),
      );
      await settle();
      await confidence(cell);
      await settle();
      const centre = tileset.boundingSphere.center;
      const sine = sunSine(time, centre);
      const east = Transforms.eastNorthUpToFixedFrame(centre, Ellipsoid.WGS84);
      const eastAxis = new Cartesian3(east[0], east[1], east[2]);
      const lambert = Math.max(Cartesian3.dot(eastAxis, sunDirection(time)), 0);
      const relit = sun !== "off" && !asCaptured;
      const factor = relit ? SUN_AMBIENT + (1 - SUN_AMBIENT) * lambert * sunUpFactor(sine) : 1;
      // The slab's grey through the strong cell's look, as the shaders draw it.
      const grey = SLAB_GREY / 255;
      const look = (k: number) => {
        const [r, g, b] = cellViewColor(
          "confidence",
          [STRONG, 0],
          [k * grey, k * grey, k * grey, 1],
        );
        return (r + g + b) / 3;
      };
      return {
        ...measure(),
        predictedFlat: factor,
        predictedCellFlat: look(factor) / look(1),
        sine,
        parts: parts(),
        reported: useSunReport.getState().reports.slab ?? null,
      };
    },
  };
}
