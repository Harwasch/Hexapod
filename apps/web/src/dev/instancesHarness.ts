/**
 * A bare CesiumJS page that draws a segmented splat tileset through the scene-instance hooks
 * (`cesium/splatInstances.ts`) -- the driver for e2e/instances.spec.ts.
 *
 * What it answers, in a real engine with the patch: does the instance id texture line up with
 * the splats drawn (hiding every instance must empty the scan), does a hidden instance vanish,
 * does a highlight tint it amber and dim the rest, do both GLSL hooks compile, and do they
 * compose with the view-cone part of the visibility chain? The tileset is attached exactly as
 * `SiteManager` attaches one (`attachViewCones`, `attachInstances`, the store in
 * `state/instances.ts`), so what is driven here is what the app runs.
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
  SceneTransforms,
  Cartesian2,
  Cartesian3,
  type Scene,
} from "cesium";

import { attachInstances, instanceSphere, type InstancePrimitive } from "@/cesium/splatInstances";
import { incrementalSplats, keepOffscreenSplats, splatTilesetOf } from "@/cesium/splatInternals";
import { cesiumViewConeGpu, SplatViewCones } from "@/cesium/splatViewCones";
import { visibilityChainOf } from "@/cesium/splatVisibility";
import { cellCount, loadViewCones, viewConesMetaOf } from "@/lib/viewCones";
import { useInstances } from "@/state/instances";

const BACKGROUND = "#10141a";
const ASSET = "harness";

/** A screen rectangle, in canvas pixels. */
export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** What a frame looks like, counted. */
export interface Measure {
  /** Share of the canvas (or rect) that is not background. */
  coverage: number;
  /** Share of the covered pixels that read as amber (the highlight tint). */
  amber: number;
  /** Mean luma (0..255) of covered pixels that are not amber. */
  luma: number;
  /** Mean red minus blue (0..255) of covered pixels: the tint pulls every colour up it. */
  warmth: number;
}

export interface InstancesHarness {
  /** Points the camera and waits for the tiles and the sort to settle. */
  view(headingDeg: number, pitchDeg: number, rangeM: number): Promise<Measure>;
  /** Every instance, largest first. */
  instances(): { id: number; parent: number | null; level: number; splats: number }[];
  /** Hides and highlights through the store, then measures (in `rect` when given). */
  set(
    state: { hidden?: number[]; highlighted?: number[]; dimOthers?: boolean },
    rect?: Rect,
  ): Promise<Measure>;
  /**
   * Swaps the scan's own view-cone grid (installed with the first view, as the app does) for
   * one "seen from below only" (`below`), or back (`off`).
   */
  cones(mode: "off" | "below"): Promise<Measure>;
  /** Where instance `id` is on screen, or null when it is not in front of the camera. */
  rectOf(id: number): Rect | null;
  measure(rect?: Rect): Measure;
  /** The hooks on the primitive. */
  hooks(): { visibility: string[]; color: boolean; table: boolean };
  /** The tiles drawn now, and their splats. */
  tiles(): { uri: string; splats: number }[];
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

/** Amber: red high, green below it, blue well below green (HIGHLIGHT_STYLE.tint, mixed in). */
function isAmber(r: number, g: number, b: number): boolean {
  return r > 110 && g > 0.5 * r && g < 0.92 * r && b < 0.62 * g;
}

export async function startInstancesHarness(options: {
  container: HTMLElement;
  url: string;
  /** The app's incremental primitive (true, as `providers/tiles.ts`), or aggregated. */
  incremental?: boolean;
  /** 16 by default; a low one draws the leaves wherever the camera is. */
  maximumScreenSpaceError?: number;
}): Promise<InstancesHarness> {
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
    maximumScreenSpaceError: options.maximumScreenSpaceError ?? 16,
    skipLevelOfDetail: false,
  });
  if (options.incremental ?? true) {
    keepOffscreenSplats(tileset);
    incrementalSplats(tileset, 0);
  }
  scene.primitives.add(tileset);
  attachInstances(tileset, scene, ASSET);

  const primitive = (): InstancePrimitive | undefined =>
    splatTilesetOf(tileset).gaussianSplatPrimitive;
  const settle = async (frames: number): Promise<void> => {
    for (let frame = 0; frame < frames; frame += 1) await nextFrame(scene);
  };

  // The view cones as `attachViewCones` installs them (the file's grid), plus a grid "seen
  // from below only" to swap in. Held here rather than attached, since two cone parts in one
  // chain would define the same GLSL function twice.
  const meta = viewConesMetaOf((tileset.root as { extras?: unknown }).extras);
  const coneGpu = cesiumViewConeGpu();
  let file: SplatViewCones | undefined;
  let below: SplatViewCones | undefined;
  if (meta && coneGpu) {
    file = new SplatViewCones(meta, await loadViewCones(options.url, meta), coneGpu);
    const texels = new Uint8Array(cellCount(meta) * 4);
    for (let i = 0; i < texels.length; i += 4) {
      texels.set([128, 128, Math.round((45 * 254) / 180), 255], i);
    }
    below = new SplatViewCones(meta, texels, coneGpu);
  }

  function measure(rect?: Rect): Measure {
    const canvas = scene.canvas;
    const copy = document.createElement("canvas");
    copy.width = canvas.width;
    copy.height = canvas.height;
    const context = copy.getContext("2d");
    if (!context) return { coverage: 0, amber: 0, luma: 0, warmth: 0 };
    context.drawImage(canvas, 0, 0);
    const x0 = Math.max(0, Math.floor(rect?.x ?? 0));
    const y0 = Math.max(0, Math.floor(rect?.y ?? 0));
    const x1 = Math.min(copy.width, Math.ceil(rect ? rect.x + rect.width : copy.width));
    const y1 = Math.min(copy.height, Math.ceil(rect ? rect.y + rect.height : copy.height));
    if (x1 <= x0 || y1 <= y0) return { coverage: 0, amber: 0, luma: 0, warmth: 0 };
    const data = context.getImageData(x0, y0, x1 - x0, y1 - y0).data;
    const bg = Color.fromCssColorString(BACKGROUND);
    const [br, bgG, bb] = [bg.red * 255, bg.green * 255, bg.blue * 255];
    let covered = 0;
    let amber = 0;
    let luma = 0;
    let warmth = 0;
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i] ?? 0;
      const g = data[i + 1] ?? 0;
      const b = data[i + 2] ?? 0;
      if (Math.abs(r - br) + Math.abs(g - bgG) + Math.abs(b - bb) <= 24) continue;
      covered += 1;
      warmth += r - b;
      if (isAmber(r, g, b)) amber += 1;
      else luma += 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }
    const total = data.length / 4;
    return {
      coverage: covered / total,
      amber: covered > 0 ? amber / covered : 0,
      luma: covered - amber > 0 ? luma / (covered - amber) : 0,
      warmth: covered > 0 ? warmth / covered : 0,
    };
  }

  return {
    async view(headingDeg, pitchDeg, rangeM) {
      scene.camera.lookAt(
        tileset.boundingSphere.center,
        new HeadingPitchRange(
          CesiumMath.toRadians(headingDeg),
          CesiumMath.toRadians(pitchDeg),
          rangeM,
        ),
      );
      for (let frame = 0; frame < 600; frame += 1) {
        if (tileset.tilesLoaded && primitive() && useInstances.getState().assets[ASSET]) break;
        await nextFrame(scene);
      }
      const target = primitive();
      if (!target) throw new Error("no splat primitive");
      if (file && !below?.installed) file.install(target);
      if (!useInstances.getState().assets[ASSET]) throw new Error("instances never loaded");
      await settle(30);
      return measure();
    },
    instances() {
      const entry = useInstances.getState().assets[ASSET];
      return (entry?.instances ?? [])
        .map(({ id, parent, level, splats }) => ({ id, parent, level, splats }))
        .sort((a, b) => b.splats - a.splats || a.id - b.id);
    },
    async set(state, rect) {
      const store = useInstances.getState();
      store.showAll(ASSET);
      store.setHidden(ASSET, state.hidden ?? [], true);
      store.highlight(ASSET, state.highlighted ?? []);
      store.setDimOthers(state.dimOthers ?? true);
      await settle(20);
      return measure(rect);
    },
    async cones(mode) {
      const target = primitive();
      if (!file || !below || !target) throw new Error("no view-cone grid on this tileset");
      if (mode === "below") {
        file.uninstall();
        below.install(target);
      } else {
        below.uninstall();
        file.install(target);
      }
      await settle(20);
      return measure();
    },
    rectOf(id) {
      const sphere = instanceSphere(ASSET, id);
      if (!sphere) return null;
      const centre = SceneTransforms.worldToWindowCoordinates(
        scene,
        sphere.center,
        new Cartesian2(),
      );
      if (!centre) return null;
      // The sphere's radius on screen, from a point offset along the camera's right.
      const edge = Cartesian3.add(
        sphere.center,
        Cartesian3.multiplyByScalar(scene.camera.rightWC, sphere.radius, new Cartesian3()),
        new Cartesian3(),
      );
      const side = SceneTransforms.worldToWindowCoordinates(scene, edge, new Cartesian2());
      if (!side) return null;
      const radius = Math.hypot(side.x - centre.x, side.y - centre.y);
      const ratio = scene.canvas.width / Math.max(1, scene.canvas.clientWidth);
      return {
        x: (centre.x - radius) * ratio,
        y: (centre.y - radius) * ratio,
        width: 2 * radius * ratio,
        height: 2 * radius * ratio,
      };
    },
    measure,
    tiles() {
      const selected = (
        tileset as unknown as {
          _selectedTiles?: { content?: { url?: string; pointsLength?: number } }[];
        }
      )._selectedTiles;
      return (selected ?? []).map((tile) => ({
        uri: (tile.content?.url ?? "").replace(/^.*\//, "").replace(/\?.*$/, ""),
        splats: tile.content?.pointsLength ?? 0,
      }));
    },
    hooks() {
      const target = primitive();
      const chain = target ? visibilityChainOf(target) : undefined;
      return {
        visibility: chain?.parts.map((p) => p.visibilityFunction) ?? [],
        color: target?.vertexColor !== undefined,
        table: useInstances.getState().assets[ASSET] !== undefined,
      };
    },
  };
}
