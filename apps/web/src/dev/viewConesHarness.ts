/**
 * A bare CesiumJS page that draws a splat tileset through the view-cone hook -- the driver for
 * e2e/viewCones.spec.ts.
 *
 * What it answers, in a real engine with the patch: does `splatVertexVisibility` compile into
 * the splat vertex shader, map the draw command's frame to the scan's grid, and fade a splat
 * only when it is seen from outside its cell's cone? The committed synthetic tree is seen from
 * everywhere (its own `viewcones.bin` fades nothing), so the harness can also install a grid
 * of its own -- every cell "seen from below only", a 45 degree cone straight up -- under which
 * the tree must vanish seen from above and stay as it was seen from below.
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
  type Scene,
} from "cesium";

import { splatTilesetOf } from "@/cesium/splatInternals";
import {
  cesiumViewConeGpu,
  SplatViewCones,
  type VisibilityPrimitive,
} from "@/cesium/splatViewCones";
import { cellCount, loadViewCones, viewConesMetaOf } from "@/lib/viewCones";

export type ConeMode = "off" | "file" | "below";

export interface ViewConesHarness {
  /** Coverage of the canvas, looking at the tree at `pitchDeg` from `rangeM`, under `mode`. */
  view(pitchDeg: number, rangeM: number, mode: ConeMode): Promise<{ coverage: number }>;
  /** Whether the primitive carries our hook. */
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

export async function startViewConesHarness(options: {
  container: HTMLElement;
  url: string;
}): Promise<ViewConesHarness> {
  const widget = new CesiumWidget(options.container, {
    baseLayer: false,
    requestRenderMode: false,
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
    maximumScreenSpaceError: 16,
    skipLevelOfDetail: false,
  });
  scene.primitives.add(tileset);
  const meta = viewConesMetaOf((tileset.root as { extras?: unknown }).extras);
  const gpu = cesiumViewConeGpu();
  if (!meta || !gpu) throw new Error("no extras.viewCones, or no texture support");
  const fromFile = await loadViewCones(options.url, meta);
  // Every cell seen from below only: axis straight up, a 45 degree half-angle.
  const fromBelow = new Uint8Array(cellCount(meta) * 4);
  for (let i = 0; i < fromBelow.length; i += 4) {
    fromBelow.set([128, 128, Math.round((45 * 254) / 180), 255], i);
  }
  const hooks = {
    file: new SplatViewCones(meta, fromFile, gpu),
    below: new SplatViewCones(meta, fromBelow, gpu),
  };
  const primitive = (): VisibilityPrimitive | undefined =>
    splatTilesetOf(tileset).gaussianSplatPrimitive;

  function coverage(): number {
    const canvas = scene.canvas;
    const copy = document.createElement("canvas");
    copy.width = canvas.width;
    copy.height = canvas.height;
    const context = copy.getContext("2d");
    if (!context) return 0;
    context.drawImage(canvas, 0, 0);
    const data = context.getImageData(0, 0, copy.width, copy.height).data;
    const bg = Color.fromCssColorString(BACKGROUND);
    const [r, g, b] = [bg.red * 255, bg.green * 255, bg.blue * 255];
    let covered = 0;
    for (let i = 0; i < data.length; i += 4) {
      const d =
        Math.abs((data[i] ?? 0) - r) +
        Math.abs((data[i + 1] ?? 0) - g) +
        Math.abs((data[i + 2] ?? 0) - b);
      if (d > 24) covered += 1;
    }
    return covered / (data.length / 4);
  }

  return {
    async view(pitchDeg, rangeM, mode) {
      scene.camera.lookAt(
        tileset.boundingSphere.center,
        new HeadingPitchRange(CesiumMath.toRadians(30), CesiumMath.toRadians(pitchDeg), rangeM),
      );
      for (let frame = 0; frame < 400 && !(tileset.tilesLoaded && primitive()); frame += 1) {
        await nextFrame(scene);
      }
      const target = primitive();
      if (!target) throw new Error("no splat primitive");
      if (mode === "off") target.vertexVisibility = undefined;
      else hooks[mode].install(target);
      // The draw command is rebuilt on the next render; the sort settles over a few more.
      for (let frame = 0; frame < 30; frame += 1) await nextFrame(scene);
      return { coverage: coverage() };
    },
    installed() {
      const target = primitive();
      return target?.vertexVisibility === hooks.file || target?.vertexVisibility === hooks.below;
    },
  };
}
