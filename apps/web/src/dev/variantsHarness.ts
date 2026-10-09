/**
 * A bare CesiumJS page that draws a scan with method variants (lib/variants.ts) the way the app
 * does, with the app's own controls beside it -- the driver for e2e/variants.spec.ts.
 *
 * The tileset is attached exactly as `SiteManager.attachTileset` attaches a splat scan:
 * `attachVariants`, `attachInstances`, `attachInferredLayers` (drawn by whichever renderer draws
 * the scan, in its sort) and `attachSkin`; with a dedicated renderer the scan and its layers are
 * drawn by the overlay (cesium/scanView) over CesiumJS's hidden tileset, as the app's default.
 * It can also put the scan at a runtime scale and lift (`placement`), and measure the frame
 * inside a polygon of the screen (`changedIn`) with every measured object hidden or not. The panel
 * holds the app's "Compare methods" rows, the inferred Show · Highlight · Hide control and the
 * objects panel, wired to this scan, so the spec drives them as a person would.
 *
 * Loaded dynamically by the spec; nothing imports it, so it never reaches the production
 * bundle. Headless GL is SwiftShader: pixels are counted, not eyeballed.
 */

import {
  Cartesian2,
  Cartesian3,
  Cesium3DTileset,
  CesiumWidget,
  Color,
  HeadingPitchRange,
  Math as CesiumMath,
  Matrix4,
  SceneTransforms,
  type BoundingSphere,
  type Scene,
} from "cesium";

import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { attachInferredLayers } from "@/cesium/inferredLayers";
import { ScanRendererHost, type ScanRendererStatus } from "@/cesium/scanView/ScanRendererHost";
import type { SplatRendererKind } from "@/cesium/scanView/types";
import { attachVariants } from "@/cesium/scanVariants";
import { attachInstances } from "@/cesium/splatInstances";
import { incrementalSplats, keepOffscreenSplats, splatTilesetOf } from "@/cesium/splatInternals";
import { attachSkin, skinningOf } from "@/cesium/splatSkin";
import { placedMatrix, placementFrame } from "@/cesium/tilesetScale";
import { evidenceOf, type InferredStyle } from "@/lib/inferred";
import type { VariantSystem } from "@/lib/variants";
import { useInferred } from "@/state/inferred";
import { useInstances } from "@/state/instances";
import { useSettings } from "@/state/settings";
import { useVariants, type VariantStatus } from "@/state/variants";

const BACKGROUND = "#10141a";
const ASSET = "harness";

/** A screen rectangle, in canvas pixels. */
export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** What a part of the frame looks like, counted. */
export interface Measure {
  /** Share of the pixels that are not background. */
  coverage: number;
  /** Share of the covered pixels that read as purple (Highlight's tint). */
  purple: number;
}

export interface VariantsHarness {
  /** Points the camera and waits for the scan, its objects and its inferred layers to settle. */
  view(headingDeg: number, pitchDeg: number, rangeM: number): Promise<void>;
  /** As `view`, looking at a point of the scan's own frame (local ENU metres). */
  viewAt(
    local: readonly number[],
    headingDeg: number,
    pitchDeg: number,
    rangeM: number,
  ): Promise<void>;
  /** Waits until what the picks load is in and drawn. */
  settle(): Promise<void>;
  /** Picks through the store, as the panel does, then settles. */
  pick(system: VariantSystem, name: string | null): Promise<void>;
  /** Sets the inferred style, as the control does, then settles. */
  style(style: InferredStyle): Promise<void>;
  /** Where a box of the scan's own frame (local ENU metres) is on screen. */
  rectOfLocal(min: readonly number[], max: readonly number[]): Rect | null;
  /** Where points of the scan's own frame are on screen (canvas pixels), null behind the eye. */
  screenOfLocal(points: readonly (readonly number[])[]): ([number, number] | null)[];
  measure(rect?: Rect): Measure;
  /** Keeps the frame drawn now under `slot`, for `changed`. */
  remember(slot?: string): void;
  /**
   * Share of the pixels (in `rect`, or outside it with `outside`) whose colour moved since the
   * frame kept under `slot`.
   */
  changed(rect?: Rect, outside?: boolean, slot?: string): number;
  /** As `changed`, over the pixels inside a polygon of canvas pixels. */
  changedIn(polygon: readonly (readonly [number, number])[], slot?: string): number;
  /**
   * Which is in front, the fill or the measured scan, in `rect`: from three frames kept --
   * both drawn (`both`), the measured scan alone (`measured`: the fill hidden), the fill alone
   * (`fill`: every measured object hidden) -- the pixels where the two alone differ, and the
   * share of them where the frame with both is nearer the measured scan's. Near 1 where the
   * measured splats are in front of the fill, near 0 where the fill is in front of them.
   */
  nearerMeasured(
    rect: Rect,
    slots: { both: string; measured: string; fill: string },
  ): { pixels: number; share: number };
  /**
   * Hides every measured object of the scan (each instance id, as the objects panel's Hide
   * does), or shows them again: what is left is the inferred layer and the few splats no
   * object holds.
   */
  hideMeasured(hidden: boolean): Promise<void>;
  /**
   * Where the scan and each inferred layer CesiumJS loaded are put on the globe: model matrix
   * and root transform (column-major), so a test can check a layer is drawn where the scan is.
   */
  frames(): {
    scan: { model: number[]; root: number[] };
    layers: { model: number[]; root: number[] }[];
  };
  /**
   * Under CesiumJS: how many layers the scan's own primitive draws in its sort (the engine
   * patch's `companions`), and, per layer loaded, whether that primitive draws it (`drawnBy`).
   * Null before the scan's primitive exists.
   */
  companions(): { companions: number; drawnBy: boolean[] } | null;
  /** The objects panel's categories, as its store has them. */
  categories(): { name: string; objects: number }[];
  /** How many objects the table lists. */
  objects(): number;
  /** What each system has picked and how its files are doing. */
  picks(): {
    picks: Partial<Record<VariantSystem, string>>;
    status: Partial<Record<VariantSystem, VariantStatus>>;
  };
  /** The inferred layers loaded, by filler, and how many are shown. */
  inferred(): { fillers: string[]; shown: number };
  /** The skins of the drawn skin (instance ids), or null when none moves. */
  skins(): number[] | null;
  scan(): ScanRendererStatus | null;
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

/** Highlight's purple: blue high, red above green, green well below blue. */
function isPurple(r: number, g: number, b: number): boolean {
  return b > 120 && r > g + 25 && b > g + 50;
}

/** The app's controls for this scan, in the app's styles. */
async function mountPanel(element: HTMLElement, scene: Scene): Promise<void> {
  // The page is not Vite's index.html, so React Fast Refresh's preamble was never injected;
  // the components' dev transform expects its globals. No-ops will do: nothing is refreshed.
  const refresh = window as unknown as Record<string, unknown>;
  refresh.$RefreshReg$ ??= () => undefined;
  refresh.$RefreshSig$ ??= () => (type: unknown) => type;
  refresh.__vite_plugin_react_preamble_installed__ = true;
  const [
    { createElement, Fragment },
    { createRoot },
    { CompareMethodsPanel },
    { InferredLegend, InferredStyleControl },
    { InstancePanel },
    { SceneContext, SceneRegistry },
  ] = await Promise.all([
    import("react"),
    import("react-dom/client"),
    import("@/features/sites/CompareMethods"),
    import("@/features/sites/InferredStyle"),
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
  /** The inferred control while the scan has layers, as the representation switcher shows it. */
  function Inferred() {
    const evidence = useInferred((s) => s.layers[ASSET]);
    const style = useSettings((s) => s.inferredStyle);
    if (!evidence || evidence.length === 0) return null;
    return createElement(
      "div",
      null,
      createElement(
        "div",
        { className: "rep-switch" },
        createElement(InferredStyleControl, { evidence }),
      ),
      style !== "hide" ? createElement(InferredLegend, { style }) : null,
    );
  }
  createRoot(element).render(
    createElement(
      SceneContext.Provider,
      { value: registry },
      createElement(
        Fragment,
        null,
        createElement(CompareMethodsPanel, { assetId: ASSET }),
        createElement(Inferred),
        createElement(InstancePanel, { assetId: ASSET }),
      ),
    ),
  );
}

export async function startVariantsHarness(options: {
  container: HTMLElement;
  url: string;
  renderer?: SplatRendererKind;
  /** Where to mount the app's controls for the scan. */
  panel?: HTMLElement;
  /**
   * Draws the scan at a runtime scale, raised by a lift (metres), as SiteManager places an
   * asset (`placedMatrix`): its inferred layers must follow it there.
   */
  placement?: { scale: number; liftM: number };
}): Promise<VariantsHarness> {
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
    maximumScreenSpaceError: 1,
    skipLevelOfDetail: false,
  });
  keepOffscreenSplats(tileset);
  incrementalSplats(tileset, 0);
  if (options.placement) {
    const { scale, liftM } = options.placement;
    tileset.modelMatrix = placedMatrix(placementFrame(tileset, undefined), scale, liftM);
  }
  scene.primitives.add(tileset);
  let host: ScanRendererHost | undefined;
  // As SiteManager.attachTileset: the scan's variants, objects, inferred layers and skin.
  attachVariants(tileset, ASSET);
  attachInstances(tileset, scene, ASSET);
  // Drawn by CesiumJS while it draws the scan, by the overlay otherwise (scanView/scanLayers.ts).
  attachInferredLayers(tileset, scene, ASSET);
  attachSkin(tileset, scene, ASSET);
  if (options.panel) await mountPanel(options.panel, scene);
  if (dedicated && options.renderer) {
    // As SiteManager.setSplatRenderer: CesiumJS's scan hidden, and not streamed.
    tileset.show = false;
    tileset.preloadWhenHidden = false;
    host = new ScanRendererHost(widget, { preserveDrawingBuffer: true });
    host.setRenderer(options.renderer);
    host.setTarget({ key: ASSET, tileset, assetId: ASSET });
  }

  /** The inferred layers' tilesets in the scene (their roots carry `extras.evidence`). */
  const layers = (): Cesium3DTileset[] => {
    const out: Cesium3DTileset[] = [];
    for (let i = 0; i < scene.primitives.length; i += 1) {
      const p = scene.primitives.get(i) as unknown;
      if (!(p instanceof Cesium3DTileset) || p === tileset) continue;
      if (evidenceOf((p.root as { extras?: unknown } | undefined)?.extras)) out.push(p);
    }
    return out;
  };

  const loading = (): boolean => {
    const status = useVariants.getState().status[ASSET] ?? {};
    return Object.values(status).some((s) => s?.state === "loading");
  };

  async function settle(): Promise<void> {
    let calm = 0;
    let lastTiles = -1;
    for (let frame = 0; frame < 3000 && calm < 25; frame += 1) {
      await nextFrame(scene);
      const status = host?.status();
      const scanReady = host
        ? status !== undefined && status.frames > 5 && status.tiles > 0 && status.loading === 0
        : tileset.tilesLoaded;
      const shown = layers().filter((l) => l.show);
      // The overlay's own layers (scanView/scanLayers.ts): every one wanted is in, or failed.
      const overlay = status?.layers;
      const layersReady =
        shown.every((l) => l.tilesLoaded) &&
        (!overlay || (overlay.loading === 0 && overlay.drawn + overlay.failed >= overlay.wanted)) &&
        (status?.settled ?? true);
      const tiles = (status?.tiles ?? 0) + shown.length + (overlay?.tiles ?? 0);
      calm = scanReady && layersReady && !loading() && tiles === lastTiles ? calm + 1 : 0;
      lastTiles = tiles;
    }
    // A few more for the overlay's sort and the globe's last frame.
    for (let frame = 0; frame < 10; frame += 1) await nextFrame(scene);
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
    if (host && overlay) context.drawImage(overlay, 0, 0, copy.width, copy.height);
    return context;
  }

  const bounds = (rect: Rect | undefined, width: number, height: number) => ({
    x0: Math.max(0, Math.floor(rect?.x ?? 0)),
    y0: Math.max(0, Math.floor(rect?.y ?? 0)),
    x1: Math.min(width, Math.ceil(rect ? rect.x + rect.width : width)),
    y1: Math.min(height, Math.ceil(rect ? rect.y + rect.height : height)),
  });

  /** Frames kept by name, for `changed`. */
  const remembered = new Map<string, Uint8ClampedArray>();
  const bg = Color.fromCssColorString(BACKGROUND);
  const [br, bgG, bb] = [bg.red * 255, bg.green * 255, bg.blue * 255];

  const lookFrom = async (
    target: Cartesian3,
    headingDeg: number,
    pitchDeg: number,
    rangeM: number,
  ): Promise<void> => {
    scene.camera.lookAt(
      target,
      new HeadingPitchRange(
        CesiumMath.toRadians(headingDeg),
        CesiumMath.toRadians(pitchDeg),
        rangeM,
      ),
    );
    scene.camera.lookAtTransform(Matrix4.IDENTITY);
    await settle();
  };

  /** Canvas pixels per CSS pixel. */
  const pixelRatio = (): number => scene.canvas.width / Math.max(1, scene.canvas.clientWidth);

  return {
    view: (headingDeg, pitchDeg, rangeM) =>
      lookFrom(tileset.boundingSphere.center, headingDeg, pitchDeg, rangeM),
    viewAt: (local, headingDeg, pitchDeg, rangeM) =>
      lookFrom(
        Matrix4.multiplyByPoint(
          tileset.root.computedTransform,
          new Cartesian3(local[0] ?? 0, local[1] ?? 0, local[2] ?? 0),
          new Cartesian3(),
        ),
        headingDeg,
        pitchDeg,
        rangeM,
      ),
    settle,
    async pick(system, name) {
      useVariants.getState().pick(ASSET, system, name);
      await settle();
    },
    async style(style) {
      useSettings.getState().set({ inferredStyle: style });
      await settle();
    },
    rectOfLocal(min, max) {
      const toWorld = tileset.root.computedTransform;
      const ratio = scene.canvas.width / Math.max(1, scene.canvas.clientWidth);
      let x0 = Number.POSITIVE_INFINITY;
      let y0 = Number.POSITIVE_INFINITY;
      let x1 = Number.NEGATIVE_INFINITY;
      let y1 = Number.NEGATIVE_INFINITY;
      for (const x of [min[0] ?? 0, max[0] ?? 0])
        for (const y of [min[1] ?? 0, max[1] ?? 0])
          for (const z of [min[2] ?? 0, max[2] ?? 0]) {
            const world = Matrix4.multiplyByPoint(
              toWorld,
              new Cartesian3(x, y, z),
              new Cartesian3(),
            );
            const at = SceneTransforms.worldToWindowCoordinates(scene, world, new Cartesian2());
            if (!at) return null;
            x0 = Math.min(x0, at.x);
            y0 = Math.min(y0, at.y);
            x1 = Math.max(x1, at.x);
            y1 = Math.max(y1, at.y);
          }
      return { x: x0 * ratio, y: y0 * ratio, width: (x1 - x0) * ratio, height: (y1 - y0) * ratio };
    },
    screenOfLocal(points) {
      const toWorld = tileset.root.computedTransform;
      const ratio = pixelRatio();
      return points.map((p) => {
        const world = Matrix4.multiplyByPoint(
          toWorld,
          new Cartesian3(p[0] ?? 0, p[1] ?? 0, p[2] ?? 0),
          new Cartesian3(),
        );
        const at = SceneTransforms.worldToWindowCoordinates(scene, world, new Cartesian2());
        return at ? [at.x * ratio, at.y * ratio] : null;
      });
    },
    measure(rect) {
      const context = composite();
      if (!context) return { coverage: 0, purple: 0 };
      const { x0, y0, x1, y1 } = bounds(rect, context.canvas.width, context.canvas.height);
      if (x1 <= x0 || y1 <= y0) return { coverage: 0, purple: 0 };
      const data = context.getImageData(x0, y0, x1 - x0, y1 - y0).data;
      let covered = 0;
      let purple = 0;
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i] ?? 0;
        const g = data[i + 1] ?? 0;
        const b = data[i + 2] ?? 0;
        if (Math.abs(r - br) + Math.abs(g - bgG) + Math.abs(b - bb) <= 24) continue;
        covered += 1;
        if (isPurple(r, g, b)) purple += 1;
      }
      return { coverage: covered / (data.length / 4), purple: covered ? purple / covered : 0 };
    },
    remember(slot = "frame") {
      const context = composite();
      const data = context
        ? context.getImageData(0, 0, context.canvas.width, context.canvas.height).data
        : null;
      if (data) remembered.set(slot, data);
      else remembered.delete(slot);
    },
    changed(rect, outside = false, slot = "frame") {
      const context = composite();
      const before = remembered.get(slot);
      if (!context || !before) return 0;
      const { width, height } = context.canvas;
      const now = context.getImageData(0, 0, width, height).data;
      if (now.length !== before.length) return 1;
      const { x0, y0, x1, y1 } = bounds(rect, width, height);
      let moved = 0;
      let counted = 0;
      for (let y = 0; y < height; y += 1)
        for (let x = 0; x < width; x += 1) {
          const inside = x >= x0 && x < x1 && y >= y0 && y < y1;
          if (rect && inside === outside) continue;
          counted += 1;
          const i = (y * width + x) * 4;
          const d =
            Math.abs((now[i] ?? 0) - (before[i] ?? 0)) +
            Math.abs((now[i + 1] ?? 0) - (before[i + 1] ?? 0)) +
            Math.abs((now[i + 2] ?? 0) - (before[i + 2] ?? 0));
          if (d > 48) moved += 1;
        }
      return counted ? moved / counted : 0;
    },
    changedIn(polygon, slot = "frame") {
      const context = composite();
      const before = remembered.get(slot);
      if (!context || !before || polygon.length < 3) return 0;
      const { width, height } = context.canvas;
      const now = context.getImageData(0, 0, width, height).data;
      if (now.length !== before.length) return 1;
      const xs = polygon.map((p) => p[0]);
      const ys = polygon.map((p) => p[1]);
      const { x0, y0, x1, y1 } = bounds(
        {
          x: Math.min(...xs),
          y: Math.min(...ys),
          width: Math.max(...xs) - Math.min(...xs),
          height: Math.max(...ys) - Math.min(...ys),
        },
        width,
        height,
      );
      /** Even-odd rule at the pixel's centre. */
      const inside = (x: number, y: number): boolean => {
        let hit = false;
        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
          const [xi, yi] = polygon[i] ?? [0, 0];
          const [xj, yj] = polygon[j] ?? [0, 0];
          if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit;
        }
        return hit;
      };
      let moved = 0;
      let counted = 0;
      for (let y = y0; y < y1; y += 1)
        for (let x = x0; x < x1; x += 1) {
          if (!inside(x + 0.5, y + 0.5)) continue;
          counted += 1;
          const i = (y * width + x) * 4;
          const d =
            Math.abs((now[i] ?? 0) - (before[i] ?? 0)) +
            Math.abs((now[i + 1] ?? 0) - (before[i + 1] ?? 0)) +
            Math.abs((now[i + 2] ?? 0) - (before[i + 2] ?? 0));
          if (d > 48) moved += 1;
        }
      return counted ? moved / counted : 0;
    },
    nearerMeasured(rect, slots) {
      const both = remembered.get(slots.both);
      const measured = remembered.get(slots.measured);
      const fill = remembered.get(slots.fill);
      const width = scene.canvas.width;
      if (!both || !measured || !fill) return { pixels: 0, share: 0 };
      const { x0, y0, x1, y1 } = bounds(rect, width, scene.canvas.height);
      const distance = (a: Uint8ClampedArray, b: Uint8ClampedArray, i: number): number =>
        Math.abs((a[i] ?? 0) - (b[i] ?? 0)) +
        Math.abs((a[i + 1] ?? 0) - (b[i + 1] ?? 0)) +
        Math.abs((a[i + 2] ?? 0) - (b[i + 2] ?? 0));
      let pixels = 0;
      let nearer = 0;
      for (let y = y0; y < y1; y += 1)
        for (let x = x0; x < x1; x += 1) {
          const i = (y * width + x) * 4;
          // Only where the fill alone and the measured scan alone tell apart.
          if (distance(measured, fill, i) <= 48) continue;
          pixels += 1;
          if (distance(both, measured, i) < distance(both, fill, i)) nearer += 1;
        }
      return { pixels, share: pixels ? nearer / pixels : 0 };
    },
    frames() {
      const frame = (t: Cesium3DTileset) => ({
        model: Matrix4.toArray(t.modelMatrix),
        root: Matrix4.toArray(
          (t.root as { transform?: Matrix4 } | undefined)?.transform ?? Matrix4.IDENTITY,
        ),
      });
      return { scan: frame(tileset), layers: layers().map(frame) };
    },
    companions() {
      const primitive = splatTilesetOf(tileset).gaussianSplatPrimitive;
      if (!primitive) return null;
      return {
        companions: primitive.companions?.length ?? 0,
        drawnBy: layers().map(
          (l) => splatTilesetOf(l).gaussianSplatPrimitive?.drawnBy === primitive,
        ),
      };
    },
    async hideMeasured(hidden) {
      const ids = (useInstances.getState().assets[ASSET]?.instances ?? []).map((i) => i.id);
      useInstances.getState().setHidden(ASSET, ids, hidden);
      await settle();
    },
    categories: () =>
      (useInstances.getState().assets[ASSET]?.index.groups ?? []).map((g) => ({
        name: g.category.name,
        objects: g.objects.length,
      })),
    objects: () => useInstances.getState().assets[ASSET]?.instances.length ?? 0,
    picks: () => ({
      picks: { ...(useVariants.getState().picks[ASSET] ?? {}) },
      status: { ...(useVariants.getState().status[ASSET] ?? {}) },
    }),
    inferred: () => ({
      fillers: (useInferred.getState().layers[ASSET] ?? []).map((e) => e.filler),
      // CesiumJS's copies drawn, and the overlay's (which hides CesiumJS's while it draws).
      shown: layers().filter((l) => l.show).length + (host?.status().layers?.drawn ?? 0),
    }),
    skins: () => skinningOf(ASSET)?.doc.skins.map((s) => s.instance) ?? null,
    scan: () => host?.status() ?? null,
  };
}
