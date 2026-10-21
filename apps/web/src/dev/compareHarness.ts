/**
 * A bare CesiumJS page that draws two dates of the synthetic yard side by side under PlayCanvas,
 * as Compare › Dates does -- the driver for e2e/compareScene.spec.ts.
 *
 * What it answers, in the real PlayCanvas the app runs: is each date drawn only on its side of
 * the divider, cut exactly there, and does moving the divider copy nothing into the work
 * buffer? The two dates are the same yard loaded twice (so without the cut they overlap splat
 * for splat), June the scan the app's renderer draws (ScanRendererHost, handed the scan as
 * SiteManager's `scanTarget` hands it under a comparison: the right-hand date) and March, the
 * left-hand date, streamed beside it by the comparison's session (compareSession.ts). Each tile
 * is coloured by the side its copy writes (March red, June blue): once every tile is in, each
 * tile's work-buffer modifier is handed a copy of itself that colours by `uHexapodSide` right
 * after writing it (PlayCanvas's public `setWorkBufferModifier`, given the code the tile holds
 * -- a read of a private member, for evidence only, as is the share of the work buffer PlayCanvas
 * copied in a frame, `renderer._gsplatBufferCopy`). The overlay's canvas is what is counted.
 *
 * Loaded dynamically by the spec; nothing imports it, so it never reaches the production
 * bundle. Headless GL is SwiftShader: pixels are counted, not eyeballed.
 */

import type { Epoch, SiteAsset } from "@twin/contracts";
import {
  Cesium3DTileset,
  CesiumWidget,
  Color,
  HeadingPitchRange,
  Math as CesiumMath,
  type Scene,
} from "cesium";
import * as pc from "playcanvas";

import { startCompare, type CompareSite } from "@/cesium/compareSession";
import { RULE_KIT } from "@/cesium/scanView/ruleKit";
import { ScanRendererHost } from "@/cesium/scanView/ScanRendererHost";
import { scanCompareReading, SIDE_WRITE_STEP } from "@/cesium/scanView/scanCompare";
import { incrementalSplats } from "@/cesium/splatInternals";

/** Of the covered pixels on one side of the divider: March's red, June's blue. */
export interface Side {
  covered: number;
  red: number;
  blue: number;
}

export interface CompareCounts {
  /** The divider, in the overlay's pixels, and the overlay's width. */
  dividerPx: number;
  width: number;
  /** Left and right of the divider, leaving out `tolerancePx` either side of it. */
  left: Side;
  right: Side;
  tolerancePx: number;
  /**
   * The columns either side of the cut, nearest first: of each, the covered pixels and those of
   * the other side's date (a blue pixel left of the divider, a red one right of it).
   */
  cut: { offset: number; covered: number; wrong: number }[];
  /** Wrong-side pixels anywhere: blue left of the divider, red right of it. */
  wrong: number;
  /** The overlay's pixels, for `diff` (base64 RGBA). */
  pixels: string;
}

export interface CompareHarness {
  /** June alone, before any comparison: drawn whole, every covered pixel its own colours. */
  alone(): Promise<Omit<CompareCounts, "pixels">>;
  /**
   * Both dates coloured, the divider where it is but the cut switched off (the frame's
   * `uHexapodDivider.y`): every splat of both drawn in one sort, red and blue interleaved on
   * either side -- what the cut is measured against. The cut is switched on again after.
   */
  uncut(): Promise<Omit<CompareCounts, "pixels">>;
  /** Both dates compared, the divider at `split` of the width, coloured by side, counted. */
  compare(split: number): Promise<CompareCounts>;
  /**
   * Moves the divider to `split` in `steps` frames, as a swipe does: the share of the work
   * buffer PlayCanvas copied in each frame drawn on the way, and the counts where it stops.
   */
  swipe(split: number, steps: number): Promise<{ copies: number[]; counts: CompareCounts }>;
  /** The comparison ended: June drawn alone again, whole, each tile's copy writing no side. */
  end(): Promise<Omit<CompareCounts, "pixels">>;
  /** Pixels that differ between two counts' images outside columns [x0, x1). */
  diff(a: CompareCounts, b: CompareCounts, x0: number, x1: number): number;
}

const BACKGROUND = "#10141a";
const SCAN_ASSET = "june-splat";
const LEFT_ASSET = "march-splat";

/** The tile's modifier coloured by the side it writes: March (1) red, June (2) blue. */
function debugModifier(code: string): string {
  return code.replace(
    SIDE_WRITE_STEP,
    `${SIDE_WRITE_STEP}\n    color.rgb = uHexapodSide < 1.5 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 0.0, 1.0);`,
  );
}

function nextFrame(scene: Scene): Promise<void> {
  return new Promise((resolve) => {
    const remove = scene.postRender.addEventListener(() => {
      remove();
      resolve();
    });
    scene.requestRender();
  });
}

const epoch = (id: string, observedAt: string, ordinal: number): Epoch =>
  ({ id, ordinal, observedAt, status: "published", currentAssetId: `${id}-splat` }) as Epoch;
const asset = (e: Epoch): SiteAsset =>
  ({
    id: `${e.id}-splat`,
    epochId: e.id,
    representation: "gaussian-splat",
    sourceKind: "observed",
    validTo: null,
    createdAt: e.observedAt,
  }) as unknown as SiteAsset;

export async function startCompareHarness(options: {
  container: HTMLElement;
  url: string;
}): Promise<CompareHarness> {
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

  const load = async (query: string) => {
    const tileset = await Cesium3DTileset.fromUrl(`${options.url}?${query}`, {
      maximumScreenSpaceError: 4,
      skipLevelOfDetail: false,
    });
    incrementalSplats(tileset, 0);
    scene.primitives.add(tileset);
    tileset.show = false;
    return tileset;
  };
  const march = await load("epoch=1");
  const june = await load("epoch=2");
  // Looking north, nearly straight down: the yard across the middle of the screen.
  scene.camera.lookAt(
    june.boundingSphere.center,
    new HeadingPitchRange(0, CesiumMath.toRadians(-85), 48),
  );
  const host = new ScanRendererHost(widget, { preserveDrawingBuffer: true });
  host.setRenderer("playcanvas");
  host.setTarget({ key: "june", tileset: june, assetId: SCAN_ASSET });

  const E1 = epoch("march", "2026-03-01T15:00:00Z", 1);
  const E2 = epoch("june", "2026-06-01T15:00:00Z", 2);
  const entry: CompareSite = {
    compare: null,
    representation: "gaussian-splat",
    epochs: [E1, E2],
    site: { assets: [asset(E1), asset(E2)] },
    handles: new Map([
      [LEFT_ASSET, { tileset: march }],
      [SCAN_ASSET, { tileset: june }],
    ]),
  };

  const overlay = (): HTMLCanvasElement | null =>
    document.querySelector<HTMLCanvasElement>("canvas[data-scan-renderer]");
  const app = (): pc.AppBase => {
    const found = pc.AppBase.getApplication();
    if (!found) throw new Error("no PlayCanvas app");
    return found;
  };
  const copyShare = (): number =>
    (app().renderer as unknown as { _gsplatBufferCopy?: number })._gsplatBufferCopy ?? 0;

  /** Frames until the renderer is still, with `more` also true, for 30 frames in a row. */
  const settle = async (more: () => boolean = () => true): Promise<void> => {
    let still = 0;
    for (let frame = 0; frame < 3000 && still < 30; frame += 1) {
      await nextFrame(scene);
      const status = host.status();
      const ready =
        status.active && status.tiles > 0 && status.loading === 0 && status.settled && more();
      still = ready ? still + 1 : 0;
    }
    if (still < 30) throw new Error("PlayCanvas did not settle");
  };

  const TOLERANCE = 2;
  function count(): CompareCounts {
    const canvas = overlay();
    if (!canvas) throw new Error("no overlay");
    const copy = document.createElement("canvas");
    copy.width = canvas.width;
    copy.height = canvas.height;
    const context = copy.getContext("2d");
    if (!context) throw new Error("no 2D context");
    context.drawImage(canvas, 0, 0);
    const w = copy.width;
    const image = context.getImageData(0, 0, w, copy.height);
    const data = image.data;
    const divider = scanCompareReading(SCAN_ASSET)?.dividerPx ?? Number.NaN;
    const left: Side = { covered: 0, red: 0, blue: 0 };
    const right: Side = { covered: 0, red: 0, blue: 0 };
    const cut = new Map<number, { covered: number; wrong: number }>();
    let wrong = 0;
    for (let p = 0; p < data.length; p += 4) {
      if ((data[p + 3] ?? 0) < 64) continue;
      const r = data[p] ?? 0;
      const b = data[p + 2] ?? 0;
      const x = (p / 4) % w;
      // A pixel's centre is x + 0.5, as gl_FragCoord.x has it.
      const isLeft = Number.isFinite(divider) ? x + 0.5 < divider : x < w / 2;
      const red = r > 1.6 * b;
      const blue = b > 1.6 * r;
      const wrongSide = isLeft ? blue : red;
      if (wrongSide && Number.isFinite(divider)) wrong += 1;
      const offset = Number.isFinite(divider) ? Math.floor(x + 0.5 - divider) : 0;
      if (Math.abs(offset + 0.5) <= 3) {
        const column = cut.get(offset) ?? { covered: 0, wrong: 0 };
        column.covered += 1;
        if (wrongSide) column.wrong += 1;
        cut.set(offset, column);
      }
      if (Math.abs(x + 0.5 - divider) <= TOLERANCE) continue;
      const side = isLeft ? left : right;
      side.covered += 1;
      if (red) side.red += 1;
      if (blue) side.blue += 1;
    }
    const share = (s: Side): Side => ({
      covered: s.covered,
      red: s.covered ? s.red / s.covered : 0,
      blue: s.covered ? s.blue / s.covered : 0,
    });
    let binary = "";
    const bytes = new Uint8Array(data.buffer);
    for (let i = 0; i < bytes.length; i += 0x8000)
      binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    return {
      dividerPx: divider,
      width: w,
      left: share(left),
      right: share(right),
      tolerancePx: TOLERANCE,
      cut: [...cut.entries()]
        .sort(([a], [b]) => Math.abs(a + 0.5) - Math.abs(b + 0.5) || a - b)
        .map(([offset, c]) => ({ offset, ...c })),
      wrong,
      pixels: btoa(binary),
    };
  }

  /** Draws the frames a moved divider asks for, then counts. */
  const drawAt = async (split: number): Promise<CompareCounts> => {
    scene.splitPosition = split;
    await settle(
      () => scanCompareReading(SCAN_ASSET)?.dividerPx === split * (overlay()?.width ?? 0),
    );
    for (let frame = 0; frame < 10; frame += 1) await nextFrame(scene);
    return count();
  };

  /** Has PlayCanvas draw `frames` frames itself: a copy set on a tile lands in one of them. */
  const drawFrames = async (frames: number): Promise<void> => {
    for (let frame = 0; frame < frames; frame += 1) {
      app().render();
      await nextFrame(scene);
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
  };

  let coloured = false;
  return {
    async alone() {
      await settle();
      const { pixels: _pixels, ...counts } = count();
      return counts;
    },
    async uncut() {
      const material = app().scene.gsplat.material;
      const at = scanCompareReading(SCAN_ASSET)?.dividerPx ?? 0;
      material.setParameter("uHexapodDivider", new Float32Array([at, 0, 0, 0]));
      material.update();
      await drawFrames(10);
      const { pixels: _pixels, ...counts } = count();
      material.setParameter("uHexapodDivider", new Float32Array([at, 1, 0, 0]));
      material.update();
      await drawFrames(10);
      return counts;
    },
    async compare(split) {
      if (!entry.compare) {
        scene.splitPosition = split;
        entry.compare = { left: E1.id, right: E2.id };
        startCompare(scene, entry, RULE_KIT);
        await settle(() => {
          const reading = scanCompareReading(SCAN_ASSET);
          return (
            reading?.on === true &&
            reading.left.dates === 1 &&
            reading.left.tiles > 0 &&
            reading.left.loading === 0
          );
        });
      }
      if (!coloured) {
        // Each tile's own modifier, coloured by the side it writes; a parameter set copies it.
        for (const component of app().root.findComponents("gsplat") as pc.GSplatComponent[]) {
          const code = (component as unknown as { _workBufferModifier?: { code?: string } | null })
            ._workBufferModifier?.code;
          if (!code?.includes(SIDE_WRITE_STEP)) continue;
          component.setWorkBufferModifier({ glsl: debugModifier(code) });
          component.setParameter("uHexapodHarness", 1);
        }
        coloured = true;
        await drawFrames(40);
        await settle();
      }
      return drawAt(split);
    },
    async swipe(split, steps) {
      const from = scene.splitPosition;
      const copies: number[] = [];
      for (let i = 1; i <= steps; i += 1) {
        scene.splitPosition = from + ((split - from) * i) / steps;
        await nextFrame(scene);
        await nextFrame(scene);
        copies.push(copyShare());
      }
      return { copies, counts: await drawAt(split) };
    },
    async end() {
      entry.compare = null;
      await settle(() => scanCompareReading(SCAN_ASSET)?.on === false);
      await drawFrames(10);
      const { pixels: _pixels, ...counts } = count();
      return counts;
    },
    diff(a, b, x0, x1) {
      const p = Uint8Array.from(atob(a.pixels), (c) => c.charCodeAt(0));
      const q = Uint8Array.from(atob(b.pixels), (c) => c.charCodeAt(0));
      let differ = 0;
      for (let i = 0; i < p.length; i += 4) {
        const x = (i / 4) % a.width;
        if (x >= x0 && x < x1) continue;
        const d =
          Math.abs((p[i] ?? 0) - (q[i] ?? 0)) +
          Math.abs((p[i + 1] ?? 0) - (q[i + 1] ?? 0)) +
          Math.abs((p[i + 2] ?? 0) - (q[i + 2] ?? 0)) +
          Math.abs((p[i + 3] ?? 0) - (q[i + 3] ?? 0));
        if (d > 24) differ += 1;
      }
      return differ;
    },
  };
}
