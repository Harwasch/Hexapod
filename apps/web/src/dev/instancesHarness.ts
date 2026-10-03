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
 * With `renderer: "playcanvas"` the scan is drawn as the app draws it by default: CesiumJS's
 * tileset hidden (but loaded, for its frame and its instances.json) and PlayCanvas streaming
 * the same tiles over it (cesium/scanView), with the objects bound by tile checksum there
 * (scanView/scanInstances.ts); `renderer: "spark"` the same with Spark, and
 * `renderer: "playcanvas-webgpu"` PlayCanvas on WebGPU (its WGSL modifier) where the browser
 * has it. Measures then read both canvases, the globe's under the renderer's.
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
  type BoundingSphere,
  type Scene,
} from "cesium";

import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { ScanRendererHost, type ScanRendererStatus } from "@/cesium/scanView/ScanRendererHost";
import { paintedDocOf } from "@/cesium/scanView/scanInstances";
import type { SplatRendererKind } from "@/cesium/scanView/types";
import { pickSourceOf } from "@/cesium/sceneSelect/pickSources";
import { attachInstances, instanceSphere, type InstancePrimitive } from "@/cesium/splatInstances";
import { incrementalSplats, keepOffscreenSplats, splatTilesetOf } from "@/cesium/splatInternals";
import { cesiumViewConeGpu, SplatViewCones } from "@/cesium/splatViewCones";
import { visibilityChainOf } from "@/cesium/splatVisibility";
import { tileInstanceIds } from "@/lib/instances";
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
  /** Waits `frames` frames (for what the renderer applies over several). */
  wait(frames: number): Promise<void>;
  /** The hooks on the primitive. */
  hooks(): { visibility: string[]; color: boolean; table: boolean };
  /** The tiles drawn now, and their splats. */
  tiles(): { uri: string; splats: number }[];
  /** The dedicated renderer's status, when one draws the scan. */
  scan(): ScanRendererStatus | null;
  /** Keeps the frame drawn now, for `changed`. */
  remember(): void;
  /** Share of the canvas whose colour moved (by more than 48 in r + g + b) since `remember`. */
  changed(): number;
  /** The scan's categories as the objects panel lists them, largest first. */
  categories(): { id: string; name: string; objects: number; splats: number }[];
  /**
   * The splats drawn now (the renderer's tiles, as scene selection reads them): how many,
   * in how many tiles, how many tiles carry ids, how many splats carry none (id 0), and how
   * many are in `category`'s objects.
   */
  drawn(category: string): {
    tiles: number;
    tilesWithIds: number;
    splats: number;
    unlabelled: number;
    inCategory: number;
  };
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

/** The last frame a WebGPU overlay drew, copied before it was presented (see below). */
const webgpuMirror = document.createElement("canvas");

/** Amber: red high, green below it, blue well below green (HIGHLIGHT_STYLE.tint, mixed in). */
function isAmber(r: number, g: number, b: number): boolean {
  return r > 110 && g > 0.5 * r && g < 0.92 * r && b < 0.62 * g;
}

/**
 * The app's objects panel over the harness, in the app's styles; "go to" flies the harness's
 * camera (the app's camera controller is not here).
 */
async function mountPanel(element: HTMLElement, scene: Scene): Promise<void> {
  // The page is not Vite's index.html, so React Fast Refresh's preamble was never injected;
  // the components' dev transform expects its globals. No-ops will do: nothing is refreshed.
  const refresh = window as unknown as Record<string, unknown>;
  refresh.$RefreshReg$ ??= () => undefined;
  refresh.$RefreshSig$ ??= () => (type: unknown) => type;
  refresh.__vite_plugin_react_preamble_installed__ = true;
  const [{ createElement }, { createRoot }, { InstancePanel }, { SceneContext, SceneRegistry }] =
    await Promise.all([
      import("react"),
      import("react-dom/client"),
      import("@/features/sites/InstanceSearch"),
      import("@/cesium/SceneContext"),
      import("@/styles/app.css"),
    ]);
  const registry = new SceneRegistry();
  registry.set({
    camera: {
      flyToBoundingSphere: (sphere: BoundingSphere) =>
        scene.camera.flyToBoundingSphere(sphere, {
          offset: new HeadingPitchRange(0, CesiumMath.toRadians(-35), sphere.radius * 3.2),
          duration: 0,
        }),
    },
  } as unknown as CesiumSceneManager);
  createRoot(element).render(
    createElement(
      SceneContext.Provider,
      { value: registry },
      createElement(InstancePanel, { assetId: ASSET }),
    ),
  );
}

export async function startInstancesHarness(options: {
  container: HTMLElement;
  url: string;
  /** The app's incremental primitive (true, as `providers/tiles.ts`), or aggregated. */
  incremental?: boolean;
  /** 16 by default; a low one draws the leaves wherever the camera is. */
  maximumScreenSpaceError?: number;
  /** Who draws the splats: CesiumJS (default) or PlayCanvas over it, as the app's default. */
  renderer?: SplatRendererKind;
  /** Where to mount the app's objects panel (`InstancePanel`), driving this scan. */
  panel?: HTMLElement;
}): Promise<InstancesHarness> {
  const dedicated = options.renderer !== undefined && options.renderer !== "cesium";
  if (dedicated) {
    // The renderer's own canvas keeps its pixels between frames, so they can be counted.
    const getContext = Object.getOwnPropertyDescriptor(HTMLCanvasElement.prototype, "getContext")
      ?.value as (this: HTMLCanvasElement, type: string, attributes?: unknown) => unknown;
    HTMLCanvasElement.prototype.getContext = function (
      this: HTMLCanvasElement,
      type: string,
      attributes?: Record<string, unknown>,
    ) {
      const forced = this.dataset.scanRenderer
        ? { ...attributes, preserveDrawingBuffer: true }
        : attributes;
      return getContext.call(this, type, forced);
    } as typeof HTMLCanvasElement.prototype.getContext;
    // A WebGPU canvas has no such attribute: once its frame is presented, reading it gives
    // transparent black. So each frame the renderer draws is copied, still unpresented, into a
    // 2D mirror -- a microtask queued as the frame takes its texture runs once PlayCanvas's
    // `render` (which takes the texture and submits the frame in one go) has returned.
    const contexts = (
      globalThis as {
        GPUCanvasContext?: {
          prototype: { getCurrentTexture: (this: { canvas: HTMLCanvasElement }) => unknown };
        };
      }
    ).GPUCanvasContext;
    if (contexts) {
      const getCurrentTexture = contexts.prototype.getCurrentTexture;
      let queued = false;
      contexts.prototype.getCurrentTexture = function (this: { canvas: HTMLCanvasElement }) {
        const canvas = this.canvas;
        if (!queued && canvas.dataset.scanRenderer) {
          queued = true;
          queueMicrotask(() => {
            queued = false;
            webgpuMirror.width = canvas.width;
            webgpuMirror.height = canvas.height;
            webgpuMirror.getContext("2d")?.drawImage(canvas, 0, 0);
          });
        }
        return getCurrentTexture.call(this);
      };
    }
  }
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
  if (options.panel) await mountPanel(options.panel, scene);
  let host: ScanRendererHost | undefined;
  if (dedicated && options.renderer) {
    // As SiteManager.setSplatRenderer: CesiumJS's scan hidden, and not streamed.
    tileset.show = false;
    tileset.preloadWhenHidden = false;
    host = new ScanRendererHost(widget);
    host.setRenderer(options.renderer);
    host.setTarget({ key: ASSET, tileset, assetId: ASSET });
  }

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

  /** The globe's canvas with the dedicated renderer's drawn over it. */
  function composite(): CanvasRenderingContext2D | null {
    const canvas = scene.canvas;
    const copy = document.createElement("canvas");
    copy.width = canvas.width;
    copy.height = canvas.height;
    const context = copy.getContext("2d", { willReadFrequently: true });
    if (!context) return null;
    context.drawImage(canvas, 0, 0);
    const overlay = document.querySelector<HTMLCanvasElement>("canvas[data-scan-renderer]");
    // On WebGPU, the last frame drawn as the mirror kept it (see above).
    const drawn = overlay?.dataset.api === "webgpu" ? webgpuMirror : overlay;
    if (host && drawn) context.drawImage(drawn, 0, 0, copy.width, copy.height);
    return context;
  }
  let remembered: Uint8ClampedArray | null = null;

  function measure(rect?: Rect): Measure {
    const context = composite();
    if (!context) return { coverage: 0, amber: 0, luma: 0, warmth: 0 };
    const copy = context.canvas;
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
      if (host) {
        // The renderer's tiles: settled once the count has held for a second.
        let settled = 0;
        let last = -1;
        for (let frame = 0; frame < 3000 && settled < 60; frame += 1) {
          await nextFrame(scene);
          const status = host.status();
          const ready =
            status.frames > 5 &&
            status.tiles > 0 &&
            status.loading === 0 &&
            useInstances.getState().assets[ASSET] !== undefined &&
            (status.instances?.matched ?? 0) > 0;
          settled = ready && status.tiles === last ? settled + 1 : 0;
          last = status.tiles;
        }
        if (!useInstances.getState().assets[ASSET]) throw new Error("instances never loaded");
        await settle(30);
        return measure();
      }
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
    scan: () => host?.status() ?? null,
    remember() {
      const context = composite();
      remembered = context
        ? context.getImageData(0, 0, context.canvas.width, context.canvas.height).data
        : null;
    },
    changed() {
      const context = composite();
      if (!context || !remembered) return 0;
      const now = context.getImageData(0, 0, context.canvas.width, context.canvas.height).data;
      if (now.length !== remembered.length) return 1;
      let moved = 0;
      for (let i = 0; i < now.length; i += 4) {
        const d =
          Math.abs((now[i] ?? 0) - (remembered[i] ?? 0)) +
          Math.abs((now[i + 1] ?? 0) - (remembered[i + 1] ?? 0)) +
          Math.abs((now[i + 2] ?? 0) - (remembered[i + 2] ?? 0));
        if (d > 48) moved += 1;
      }
      return moved / (now.length / 4);
    },
    categories: () =>
      (useInstances.getState().assets[ASSET]?.index.groups ?? []).map((g) => ({
        id: g.category.id,
        name: g.category.name,
        objects: g.objects.length,
        splats: g.splats,
      })),
    drawn(category) {
      const tiles = pickSourceOf(ASSET)?.tiles() ?? [];
      const doc = paintedDocOf(ASSET);
      const categoryOf = useInstances.getState().assets[ASSET]?.index.categoryOf;
      const out = { tiles: tiles.length, tilesWithIds: 0, splats: 0, unlabelled: 0, inCategory: 0 };
      for (const tile of tiles) {
        const ids = doc ? tileInstanceIds(doc, tile.checksum) : undefined;
        out.splats += tile.count;
        if (ids?.length !== tile.count) {
          out.unlabelled += tile.count;
          continue;
        }
        out.tilesWithIds += 1;
        for (const id of ids) {
          if (id === 0) out.unlabelled += 1;
          else if (categoryOf?.get(id) === category) out.inCategory += 1;
        }
      }
      return out;
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
    wait: (frames) => settle(frames),
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
