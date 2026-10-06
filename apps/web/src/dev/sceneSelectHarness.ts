/**
 * A bare CesiumJS page that draws a segmented splat tileset -- by CesiumJS, PlayCanvas or Spark,
 * as the app does (`instancesHarness.ts`) -- with scene selection on it
 * (cesium/sceneSelect/SceneSelectController.ts) and the HUD's selection card for an object
 * (features/sites/ObjectCard.tsx), in the right-hand corner where the app's dock puts it: the
 * driver for e2e/sceneSelect.spec.ts. The spec clicks, types and paints with the real mouse
 * and keyboard; this only points the camera, says where an object is on screen, and reads the
 * stores and the pixels back.
 *
 * Loaded dynamically by the spec; nothing imports it, so it never reaches the production
 * bundle.
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
  type Scene,
} from "cesium";
import { createElement } from "react";
import { createRoot } from "react-dom/client";

import { ScanRendererHost } from "@/cesium/scanView/ScanRendererHost";
import { paintedDocOf } from "@/cesium/scanView/scanInstances";
import { pickSourceOf } from "@/cesium/sceneSelect/pickSources";
import { SceneSelectController } from "@/cesium/sceneSelect/SceneSelectController";
import { attachInstances, instanceSphere } from "@/cesium/splatInstances";
import { incrementalSplats, keepOffscreenSplats } from "@/cesium/splatInternals";
import { ObjectCard } from "@/features/sites/ObjectCard";
import { tileInstanceIds, withDescendants } from "@/lib/instances";
import { projectTiles, visibleSplats } from "@/lib/splatPaint";
import { castRay } from "@/lib/splatPick";
import { useInstances } from "@/state/instances";
import {
  chosenCombination,
  objectSelected,
  selectedId,
  selectedIds,
  useSceneSelect,
} from "@/state/sceneSelect";
// The styles the card is drawn with in the app: the glass, and the selection card's own.
import "@twin/ui/styles.css";
import "@/styles/mission.css";

const BACKGROUND = "#10141a";
const ASSET = "harness";

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface SceneSelectHarness {
  /** Looks at instance `id` and waits for the tiles to settle. */
  view(id: number, headingDeg: number, pitchDeg: number, rangeM: number): Promise<void>;
  /** Where instance `id`'s drawn splats are on screen (CSS px): their median and 10-90% box. */
  screenOf(id: number): { x: number; y: number; rect: Rect; splats: number } | null;
  /**
   * As `screenOf`, of the splats of `id` the brush would paint: those in front, seen from the
   * camera (lib/splatPaint.ts `visibleSplats`), not hidden behind another object.
   */
  visibleOf(id: number): { x: number; y: number; rect: Rect; splats: number } | null;
  /** Every instance. */
  instances(): { id: number; parent: number | null; splats: number }[];
  /** The selection as the stores have it. */
  state(): {
    candidates: number[];
    chain: number;
    index: number;
    selected: number | null;
    /** What is selected: a combination's members, else the one chosen (`selectedIds`). */
    selectedIds: number[];
    /** The combination chosen, if one is. */
    combination: { ids: number[]; iou: number } | null;
    mode: string;
    paint: {
      ids: readonly number[];
      best: number | null;
      iou: number;
      painted: number;
      live?: boolean;
    } | null;
    hidden: number[];
    highlighted: number[];
    custom: number;
  };
  /** Share of `rect` (CSS px) that is not background, both canvases. */
  coverage(rect?: Rect): number;
  /** Keeps the pixels of `rect` (CSS px), both canvases, for `changed`. */
  hold(rect: Rect): void;
  /** How far the pixels of `rect` moved from what `hold` kept: mean absolute difference, 0..1. */
  changed(rect: Rect): number;
  frames(count: number): Promise<void>;
  /** What a pick at (`x`, `y`) (CSS px) meets: the source's tiles, those with ids, the hits. */
  probe(
    x: number,
    y: number,
  ): {
    renderer: string | null;
    tiles: number;
    splats: number;
    withIds: number;
    hits: number;
    idHits: number;
    front: number | null;
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

export async function startSceneSelectHarness(options: {
  container: HTMLElement;
  url: string;
  renderer?: "cesium" | "playcanvas" | "spark";
}): Promise<SceneSelectHarness> {
  const dedicated = options.renderer === "playcanvas" || options.renderer === "spark";
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
    maximumScreenSpaceError: 16,
    skipLevelOfDetail: false,
  });
  keepOffscreenSplats(tileset);
  incrementalSplats(tileset, 0);
  scene.primitives.add(tileset);
  attachInstances(tileset, scene, ASSET);
  let host: ScanRendererHost | undefined;
  if (dedicated && options.renderer) {
    tileset.show = false;
    tileset.preloadWhenHidden = false;
    host = new ScanRendererHost(widget);
    host.setRenderer(options.renderer);
    host.setTarget({ key: ASSET, tileset, assetId: ASSET });
  }
  const controller = new SceneSelectController(widget);
  const cardRoot = document.createElement("div");
  Object.assign(cardRoot.style, {
    position: "fixed",
    right: "12px",
    bottom: "12px",
    width: "min(22rem, calc(100vw - 24px))",
    zIndex: "10",
  });
  document.body.appendChild(cardRoot);
  // The app's stylesheet (app.css) sizes every box by its border; the card is laid out so.
  const sizing = document.createElement("style");
  sizing.textContent =
    "[data-card-root] *, [data-card-root] *::before, [data-card-root] *::after { box-sizing: border-box; }";
  document.head.appendChild(sizing);
  cardRoot.dataset.cardRoot = "";
  // Shown as the app's selection card is: while an object is selected or the brush is out.
  const Card = () =>
    useSceneSelect(objectSelected) ? createElement(ObjectCard, { controller }) : null;
  createRoot(cardRoot).render(createElement(Card));

  const settle = async (frames: number): Promise<void> => {
    for (let frame = 0; frame < frames; frame += 1) await nextFrame(scene);
  };

  /** The median and 10-90% box of points (CSS px). */
  const spread = (
    xs: number[],
    ys: number[],
  ): { x: number; y: number; rect: Rect; splats: number } | null => {
    if (xs.length === 0) return null;
    xs.sort((a, b) => a - b);
    ys.sort((a, b) => a - b);
    const q = (list: number[], f: number): number =>
      list[Math.min(list.length - 1, Math.floor(f * list.length))] ?? 0;
    return {
      x: q(xs, 0.5),
      y: q(ys, 0.5),
      rect: {
        x: q(xs, 0.1),
        y: q(ys, 0.1),
        width: q(xs, 0.9) - q(xs, 0.1),
        height: q(ys, 0.9) - q(ys, 0.1),
      },
      splats: xs.length,
    };
  };

  let held: Uint8ClampedArray | null = null;
  const pixels = (rect: Rect): Uint8ClampedArray | null => {
    const canvas = scene.canvas;
    const copy = document.createElement("canvas");
    copy.width = canvas.width;
    copy.height = canvas.height;
    const context = copy.getContext("2d");
    if (!context) return null;
    context.drawImage(canvas, 0, 0);
    const overlay = document.querySelector<HTMLCanvasElement>("canvas[data-scan-renderer]");
    if (host && overlay) context.drawImage(overlay, 0, 0, copy.width, copy.height);
    const ratio = canvas.width / Math.max(1, canvas.clientWidth);
    const w = Math.max(1, Math.round(rect.width * ratio));
    const h = Math.max(1, Math.round(rect.height * ratio));
    return context.getImageData(Math.round(rect.x * ratio), Math.round(rect.y * ratio), w, h).data;
  };

  const coverage = (rect?: Rect): number => {
    const canvas = scene.canvas;
    const copy = document.createElement("canvas");
    copy.width = canvas.width;
    copy.height = canvas.height;
    const context = copy.getContext("2d");
    if (!context) return 0;
    context.drawImage(canvas, 0, 0);
    const overlay = document.querySelector<HTMLCanvasElement>("canvas[data-scan-renderer]");
    if (host && overlay) context.drawImage(overlay, 0, 0, copy.width, copy.height);
    const ratio = canvas.width / Math.max(1, canvas.clientWidth);
    const x0 = Math.max(0, Math.floor((rect?.x ?? 0) * ratio));
    const y0 = Math.max(0, Math.floor((rect?.y ?? 0) * ratio));
    const x1 = Math.min(copy.width, Math.ceil(rect ? (rect.x + rect.width) * ratio : copy.width));
    const y1 = Math.min(
      copy.height,
      Math.ceil(rect ? (rect.y + rect.height) * ratio : copy.height),
    );
    if (x1 <= x0 || y1 <= y0) return 0;
    const data = context.getImageData(x0, y0, x1 - x0, y1 - y0).data;
    const bg = Color.fromCssColorString(BACKGROUND);
    const [br, bgG, bb] = [bg.red * 255, bg.green * 255, bg.blue * 255];
    let covered = 0;
    for (let i = 0; i < data.length; i += 4) {
      const d =
        Math.abs((data[i] ?? 0) - br) +
        Math.abs((data[i + 1] ?? 0) - bgG) +
        Math.abs((data[i + 2] ?? 0) - bb);
      if (d > 24) covered += 1;
    }
    return covered / (data.length / 4);
  };

  return {
    async view(id, headingDeg, pitchDeg, rangeM) {
      for (let frame = 0; frame < 600 && !useInstances.getState().assets[ASSET]; frame += 1)
        await nextFrame(scene);
      const sphere = instanceSphere(ASSET, id);
      if (!sphere) throw new Error(`no instance ${String(id)}`);
      scene.camera.lookAt(
        sphere.center,
        new HeadingPitchRange(
          CesiumMath.toRadians(headingDeg),
          CesiumMath.toRadians(pitchDeg),
          rangeM,
        ),
      );
      // Off the look-at frame, so the camera moves freely again (and pickRay is plain).
      scene.camera.lookAtTransform(Matrix4.IDENTITY);
      let settled = 0;
      let last = -1;
      for (let frame = 0; frame < 3000 && settled < 60; frame += 1) {
        await nextFrame(scene);
        const tiles = pickSourceOf(ASSET)?.tiles().length ?? 0;
        const status = host?.status();
        const ready = tiles > 0 && (status ? status.loading === 0 : tileset.tilesLoaded);
        settled = ready && tiles === last ? settled + 1 : 0;
        last = tiles;
      }
      await settle(30);
    },
    screenOf(id) {
      const source = pickSourceOf(ASSET);
      const doc = paintedDocOf(ASSET);
      const toWorld = source?.toWorld();
      if (!source || !doc || !toWorld) return null;
      const wanted = withDescendants(doc, [id]);
      const xs: number[] = [];
      const ys: number[] = [];
      const scratch = new Cartesian3();
      const screen = new Cartesian2();
      for (const tile of source.tiles()) {
        const ids = tileInstanceIds(doc, tile.checksum);
        if (!ids) continue;
        for (let i = 0; i < tile.count; i += 1) {
          if (!wanted.has(ids[i] ?? 0) || (tile.opacity[i] ?? 0) < 0.3) continue;
          const world = Matrix4.multiplyByPoint(
            toWorld,
            Cartesian3.fromElements(
              tile.positions[i * 3] ?? 0,
              tile.positions[i * 3 + 1] ?? 0,
              tile.positions[i * 3 + 2] ?? 0,
            ),
            scratch,
          );
          const at = SceneTransforms.worldToWindowCoordinates(scene, world, screen);
          if (!at) continue;
          if (at.x < 0 || at.y < 0 || at.x > scene.canvas.clientWidth) continue;
          if (at.y > scene.canvas.clientHeight) continue;
          xs.push(at.x);
          ys.push(at.y);
        }
      }
      return spread(xs, ys);
    },
    visibleOf(id) {
      const source = pickSourceOf(ASSET);
      const doc = paintedDocOf(ASSET);
      const toWorld = source?.toWorld();
      if (!source || !doc || !toWorld) return null;
      const tiles = source.tiles();
      const ids = tiles.map((tile) => tileInstanceIds(doc, tile.checksum));
      const hiddenIds = useInstances.getState().assets[ASSET]?.hidden;
      const hidden = hiddenIds?.size ? withDescendants(doc, hiddenIds) : new Set<number>();
      const camera = scene.camera;
      const viewProj = Matrix4.multiply(
        camera.frustum.projectionMatrix,
        Matrix4.multiply(camera.viewMatrix, toWorld, new Matrix4()),
        new Matrix4(),
      );
      // As the controller projects for the brush: 3 px cells, what is hidden left out.
      const cellPx = 3;
      const screen = projectTiles(
        tiles,
        Matrix4.toArray(viewProj),
        scene.canvas.clientWidth,
        scene.canvas.clientHeight,
        cellPx,
        hidden.size ? (tile, index) => !hidden.has(ids[tile]?.[index] ?? 0) : undefined,
      );
      const visible = visibleSplats(screen);
      const wanted = withDescendants(doc, [id]);
      const xs: number[] = [];
      const ys: number[] = [];
      for (let k = 0; k < screen.count; k += 1) {
        if (!visible[k] || (screen.opacity[k] ?? 0) < 0.3) continue;
        if (!wanted.has(ids[screen.tile[k] ?? 0]?.[screen.index[k] ?? 0] ?? 0)) continue;
        const cell = screen.cell[k] ?? 0;
        xs.push(((cell % screen.cols) + 0.5) * cellPx);
        ys.push((Math.floor(cell / screen.cols) + 0.5) * cellPx);
      }
      return spread(xs, ys);
    },
    instances() {
      return (useInstances.getState().assets[ASSET]?.instances ?? []).map(
        ({ id, parent, splats }) => ({ id, parent, splats }),
      );
    },
    state() {
      const s = useSceneSelect.getState();
      const entry = useInstances.getState().assets[ASSET];
      return {
        candidates: s.candidates,
        chain: s.chain,
        index: s.index,
        selected: selectedId(s),
        selectedIds: [...selectedIds(s)],
        combination: (() => {
          const combination = chosenCombination(s);
          return combination ? { ids: [...combination.ids], iou: combination.iou } : null;
        })(),
        mode: s.mode,
        paint: s.paint,
        hidden: [...(entry?.hidden ?? [])],
        highlighted: [...(entry?.highlighted ?? [])],
        custom: (s.custom[ASSET] ?? []).length,
      };
    },
    coverage,
    hold(rect) {
      held = pixels(rect);
    },
    changed(rect) {
      const now = pixels(rect);
      if (!now || held?.length !== now.length) return 0;
      let sum = 0;
      for (let i = 0; i < now.length; i += 4) {
        for (let c = 0; c < 3; c += 1) sum += Math.abs((now[i + c] ?? 0) - (held[i + c] ?? 0));
      }
      return sum / ((now.length / 4) * 3 * 255);
    },
    frames: settle,
    probe(x, y) {
      const source = pickSourceOf(ASSET);
      const doc = paintedDocOf(ASSET);
      const toWorld = source?.toWorld();
      const tiles = source?.tiles() ?? [];
      const ids = doc ? tiles.map((tile) => tileInstanceIds(doc, tile.checksum)) : [];
      const out = {
        renderer: source?.renderer ?? null,
        tiles: tiles.length,
        splats: tiles.reduce((n, t) => n + t.count, 0),
        withIds: ids.filter((list, i) => list?.length === tiles[i]?.count).length,
        hits: 0,
        idHits: 0,
        front: null as number | null,
      };
      const ray = scene.camera.getPickRay(new Cartesian2(x, y));
      if (!toWorld || !ray) return out;
      const toLocal = Matrix4.inverseTransformation(toWorld, new Matrix4());
      const o = Matrix4.multiplyByPoint(toLocal, ray.origin, new Cartesian3());
      const d = Matrix4.multiplyByPointAsVector(toLocal, ray.direction, new Cartesian3());
      const frustum = scene.camera.frustum as { fovy?: number };
      const hits = castRay(
        tiles,
        { origin: [o.x, o.y, o.z], direction: [d.x, d.y, d.z] },
        { pixelAngle: (frustum.fovy ?? 1) / Math.max(1, scene.canvas.clientHeight) },
      );
      out.hits = hits.length;
      out.idHits = hits.filter((h) => (ids[h.tile]?.[h.index] ?? 0) !== 0).length;
      out.front = hits[0]?.t ?? null;
      return out;
    },
  };
}
