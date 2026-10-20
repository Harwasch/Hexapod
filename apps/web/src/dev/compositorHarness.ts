/**
 * A bare CesiumJS page that draws two dates of the synthetic yard as one under a per-cell view
 * rule -- the driver for e2e/compositorScene.spec.ts.
 *
 * What it answers, in a real engine with the patch: does the frame anchor's primitive draw the
 * other date as a companion, do the compositor's two shader parts compile into its vertex
 * shader, and does each cell's pixels come from one date? The two dates are the same yard
 * loaded twice (so without the rule they overlap splat for splat), each splat tinted by its
 * source with a debug colour (March red, June blue, an inferred layer green) by a colour part
 * of this harness's own that reads the compositor's slot sources. The grids come from the
 * TypeScript fixture encoder (`cellGridFixture.ts`).
 *
 * - `both`: the rule's parts installed but off (`enabled = false`): every splat of both dates
 *   drawn in one sort, red and blue interleaved -- mixed, purple pixels.
 * - `latest`: March saw the whole yard, June its east half. Looking north, the left of the
 *   screen must be red and the right blue, with no mix.
 * - `gap-latest` / `gap-truth`: March saw only the north-west, June the east; an inferred layer
 *   (the yard again, registered as one) covers everything by its tiles' bounds. Latest fills
 *   the south-west green from it; Ground truth draws nothing there.
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

import { setCompanions } from "@/cesium/companions";
import {
  startCompositor,
  type CompositorHandle,
  type CompositorSite,
} from "@/cesium/compositorSession";
import { addColorPart, removeColorPart, type SplatColorPart } from "@/cesium/splatColor";
import { incrementalSplats, splatTilesetOf } from "@/cesium/splatInternals";
import { decodeCellGrid, type CellGrid } from "@/lib/cellGrid";
import type { ViewRule } from "@/lib/compositor";
import { useSettings } from "@/state/settings";

import { encodeConfidenceGrid } from "./cellGridFixture";

export type Scenario = "both" | "latest" | "gap-latest" | "gap-truth";

/** Of the covered pixels in a region: red (March), blue (June), green (inferred), mixed. */
export interface Region {
  covered: number;
  red: number;
  blue: number;
  green: number;
  mixed: number;
}

export interface CompositorCounts {
  /** Share of the canvas the yard covers. */
  coverage: number;
  /** Left and right of the screen, leaving out a band around the seam (x = 16 m). */
  left: Region;
  right: Region;
  /** The south-west quarter (left, lower half), where only the inferred layer may draw. */
  southWest: Region;
  all: Region;
  /** Whether June's own primitive is drawn by March's, and the slots know their sources. */
  companion: boolean;
  sources: { march: number; june: number; fill: number };
}

export interface CompositorHarness {
  view(scenario: Scenario): Promise<CompositorCounts>;
}

const BACKGROUND = "#10141a";

/** A debug colour per source, after the compositor's own colour part (11). */
const DEBUG_GLSL = `
vec4 splatCompositorDebugColor(uint splatIndex, vec3 position, vec4 color) {
    int source = splatCompSource(splatIndex);
    if (source == 0) {
        return vec4(1.0, 0.0, 0.0, color.a);
    }
    if (source == 1) {
        return vec4(0.0, 0.0, 1.0, color.a);
    }
    return vec4(0.0, 1.0, 0.0, color.a);
}
`;

const DEBUG: SplatColorPart = {
  colorFunction: "splatCompositorDebugColor",
  colorOrder: 12,
  addToShader: (builder) => builder.addVertexLines(DEBUG_GLSL),
};

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

export async function startCompositorHarness(options: {
  container: HTMLElement;
  url: string;
}): Promise<CompositorHarness> {
  const widget = new CesiumWidget(options.container, {
    baseLayer: false,
    requestRenderMode: true,
    msaaSamples: 1,
    contextOptions: { webgl: { preserveDrawingBuffer: true } },
  });
  const { scene } = widget;
  // Inferred layers are drawn (Show), so a rule may give them the cells no date saw.
  useSettings.getState().set({ inferredStyle: "show" });
  scene.globe.show = false;
  if (scene.skyBox) scene.skyBox.show = false;
  if (scene.skyAtmosphere) scene.skyAtmosphere.show = false;
  if (scene.sun) scene.sun.show = false;
  scene.backgroundColor = Color.fromCssColorString(BACKGROUND);

  // The same yard three times: March (the frame's own), June, and an inferred layer of June's.
  const load = async (query: string) => {
    const tileset = await Cesium3DTileset.fromUrl(`${options.url}?${query}`, {
      maximumScreenSpaceError: 4,
      skipLevelOfDetail: false,
    });
    incrementalSplats(tileset, 0);
    scene.primitives.add(tileset);
    return tileset;
  };
  const march = await load("epoch=1");
  const june = await load("epoch=2");
  const fill = await load("fill=1");
  fill.show = false;
  const primitiveOf = (t: Cesium3DTileset) => splatTilesetOf(t).gaussianSplatPrimitive;

  // The yard: 32 x 24 m east/north of its frame's origin. 1 m cells, a margin of 2.
  const grid = async (seen: (i: number, j: number) => boolean): Promise<CellGrid> => {
    const { meta, planes } = await encodeConfidenceGrid({
      level: 2,
      i0: -2,
      j0: -2,
      ni: 36,
      nj: 28,
      cell: (i, j) => ({ score: seen(i, j) ? 220 : 0 }),
    });
    return decodeCellGrid(meta, planes);
  };
  const grids = {
    everywhere: await grid(() => true),
    east: await grid((i) => i >= 16),
    northWest: await grid((i, j) => i < 16 && j >= 12),
  };
  // The yard's tileset.json carries no confidence grid: each date's root says it has one, as a
  // published run's does (extras.confidence), and `loadGrid` below hands the scenario's over.
  for (const date of [march, june]) {
    const root = date.root as unknown as { _header: { extras?: Record<string, unknown> } };
    root._header.extras = { ...root._header.extras, confidence: grids.everywhere.meta };
  }

  const E1 = epoch("march", "2026-03-01T15:00:00Z", 1);
  const E2 = epoch("june", "2026-06-01T15:00:00Z", 2);
  const entry: CompositorSite = {
    rule: { rule: "latest" },
    representation: "gaussian-splat",
    epochs: [E1, E2],
    epochId: null,
    site: { assets: [asset(E1), asset(E2)] },
    handles: new Map([
      [`${E1.id}-splat`, { tileset: march }],
      [`${E2.id}-splat`, { tileset: june, follows: `${E1.id}-splat` }],
    ]),
  };
  let session: CompositorHandle | undefined;
  const inferred = Symbol("harness inferred");

  const start = (rule: ViewRule, marchGrid: CellGrid, withFill: boolean) => {
    const host = primitiveOf(march);
    // The debug colour reads the compositor's helpers: it leaves before they do.
    if (host) removeColorPart(host, DEBUG);
    session?.dispose();
    if (host) setCompanions(host, inferred, []);
    const own = primitiveOf(fill);
    if (own) own.drawnBy = undefined;
    fill.show = withFill;
    entry.rule = { rule };
    session = startCompositor(
      {
        scene,
        entry,
        cesium: () => true,
        end: (reason) => {
          throw new Error(`the rule ended: ${reason}`);
        },
      },
      {
        // March has its grid; June has the east half; the fill has none (tile bounds).
        loadGrid: (url) => Promise.resolve(url.includes("epoch=1") ? marchGrid : grids.east),
      },
    );
  };

  function count(): Omit<CompositorCounts, "companion" | "sources"> {
    const canvas = scene.canvas;
    const copy = document.createElement("canvas");
    copy.width = canvas.width;
    copy.height = canvas.height;
    const context = copy.getContext("2d");
    const region = (): Region => ({ covered: 0, red: 0, blue: 0, green: 0, mixed: 0 });
    const n = { left: region(), right: region(), southWest: region(), all: region() };
    if (!context) throw new Error("no 2D context");
    context.drawImage(canvas, 0, 0);
    const data = context.getImageData(0, 0, copy.width, copy.height).data;
    const bg = Color.fromCssColorString(BACKGROUND);
    const [br, bgG, bb] = [bg.red * 255, bg.green * 255, bg.blue * 255];
    const w = copy.width;
    const h = copy.height;
    for (let p = 0; p < data.length; p += 4) {
      const r = data[p] ?? 0;
      const g = data[p + 1] ?? 0;
      const b = data[p + 2] ?? 0;
      if (Math.abs(r - br) + Math.abs(g - bgG) + Math.abs(b - bb) <= 24) continue;
      const x = (p / 4) % w;
      const y = Math.floor(p / 4 / w);
      const max = Math.max(r, g, b);
      const kind =
        r > 40 && b > 40 && Math.min(r, b) / Math.max(r, b) > 0.45
          ? "mixed"
          : max === r && r > 1.6 * Math.max(g, b)
            ? "red"
            : max === b && b > 1.6 * Math.max(r, g)
              ? "blue"
              : max === g && g > 1.6 * Math.max(r, b)
                ? "green"
                : null;
      const into = [n.all];
      // A band a tenth of the width about the middle (the seam at x = 16 m) is left out.
      if (x < w * 0.45) into.push(n.left);
      if (x > w * 0.55) into.push(n.right);
      if (x < w * 0.45 && y > h * 0.55) into.push(n.southWest);
      for (const r0 of into) {
        r0.covered += 1;
        if (kind) r0[kind] += 1;
      }
    }
    const share = (r0: Region): Region => ({
      covered: r0.covered,
      red: r0.covered ? r0.red / r0.covered : 0,
      blue: r0.covered ? r0.blue / r0.covered : 0,
      green: r0.covered ? r0.green / r0.covered : 0,
      mixed: r0.covered ? r0.mixed / r0.covered : 0,
    });
    return {
      coverage: n.all.covered / (w * h),
      left: share(n.left),
      right: share(n.right),
      southWest: share(n.southWest),
      all: share(n.all),
    };
  }

  return {
    async view(scenario) {
      // Looking north, nearly straight down: east is the right of the screen.
      scene.camera.lookAt(
        march.boundingSphere.center,
        new HeadingPitchRange(0, CesiumMath.toRadians(-85), 48),
      );
      const gap = scenario === "gap-latest" || scenario === "gap-truth";
      start(
        scenario === "gap-truth" ? "ground-truth" : "latest",
        gap ? grids.northWest : grids.everywhere,
        gap,
      );
      let host: ReturnType<typeof primitiveOf>;
      for (let frame = 0; frame < 900; frame += 1) {
        await nextFrame(scene);
        host = primitiveOf(march);
        // An inferred layer of June's, drawn on the host as inferredLayers.ts would put it.
        if (gap && host) {
          setCompanions(host, inferred, [fill], june);
          const own = primitiveOf(fill);
          if (own) own.drawnBy = host;
        }
        const parts = session?.compositor();
        const plane = parts?.plane;
        const ready =
          march.tilesLoaded &&
          june.tilesLoaded &&
          (!gap || fill.tilesLoaded) &&
          parts?.installed === true &&
          plane !== undefined &&
          plane.owners.includes(0) &&
          plane.owners.includes(1) &&
          (scenario !== "gap-latest" || plane.owners.includes(2));
        if (ready) break;
      }
      const parts = session?.compositor();
      if (!host || !parts) throw new Error("no compositor on the host");
      parts.enabled = scenario !== "both";
      if (!addColorPart(host, DEBUG)) throw new Error("the engine refused the debug colour");
      // The draw command is rebuilt on the next render; the sort settles over a few more.
      for (let frame = 0; frame < 40; frame += 1) await nextFrame(scene);
      const sources = { march: 0, june: 0, fill: 0 };
      for (let slot = 0; slot < parts.sources.covered; slot += 1) {
        const s = parts.sources.sourceAt(slot);
        if (s === 0) sources.march += 1;
        else if (s === 1) sources.june += 1;
        else if (s === 2) sources.fill += 1;
      }
      const companion = primitiveOf(june)?.drawnBy === host;
      return { ...count(), companion, sources };
    },
  };
}
