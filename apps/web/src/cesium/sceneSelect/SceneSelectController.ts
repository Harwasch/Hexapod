/**
 * Selecting a scan's objects in the scene (docs/SCENE_OBJECTS.md, "Selecting in the scene"),
 * under whichever renderer draws the scan: a click picks the splats under the cursor, a brush
 * paints them. Both read the tiles drawn now from the renderer's pick source
 * (pickSources.ts) and test them on the CPU in the scan's frame (lib/splatPick.ts,
 * lib/splatPaint.ts), so CesiumJS, PlayCanvas and Spark behave alike.
 *
 * - **Click**: the ray through the cursor is composited front to back over the splats it
 *   meets; the instance with most of the pixel, its chain up to the top level and the other
 *   instances met near the front are the candidates (lib/sceneSelect.ts), and the smallest of
 *   the chain that is big enough on screen is chosen. `[` / `]`, Tab / Shift+Tab, or the wheel
 *   with Alt held (or over the chip) cycle; Escape clears.
 * - **Brush** (`B`, or the chip's brush): strokes on screen collect the front-most splats under
 *   them -- Shift adds to the painted area, Alt takes away, a plain stroke starts again -- and
 *   the instance (any level) with the best intersection over union is selected. Below
 *   `PAINT_MIN_IOU` the painted splats can be kept as an object of their own
 *   (lib/customSets.ts), drawn through the same hide and highlight pipeline.
 *
 * The chosen candidate is the objects store's highlight (`state/instances.ts`); the chip
 * (features/sites/SceneSelectChip.tsx) shows it and offers Hide, Show only, Fly to and Clear.
 */

import {
  BoundingSphere,
  Cartesian2,
  Cartesian3,
  HeadingPitchRange,
  Math as CesiumMath,
  Matrix4,
  SceneTransforms,
  type Camera,
  type Scene,
} from "cesium";

import {
  customId,
  decodeRanges,
  encodeRanges,
  isCustomId,
  rangesLength,
  type CustomSet,
} from "@/lib/customSets";
import { tileInstanceIds, withDescendants, type InstancesDoc } from "@/lib/instances";
import { createLogger } from "@/lib/log";
import {
  bestByIoU,
  buildCandidates,
  chainOf,
  defaultIndex,
  hiddenForShowOnly,
  PAINT_MIN_IOU,
} from "@/lib/sceneSelect";
import {
  BrushMask,
  paintedSplats,
  projectTiles,
  visibleSplats,
  type ScreenSplats,
} from "@/lib/splatPaint";
import { castRay, hitWeights, type PickTile } from "@/lib/splatPick";
import { useInstances } from "@/state/instances";
import { selectedId, useSceneSelect } from "@/state/sceneSelect";

import { paintedDocOf } from "../scanView/scanInstances";
import { instanceSphere, instancesDocOf } from "../splatInstances";
import { pickAssets, pickSourceOf } from "./pickSources";

const log = createLogger("scene-select");

/** A press that moves less than this (CSS px) and is released within `CLICK_MS` is a click. */
const CLICK_PX = 5;
const CLICK_MS = 600;
/** Screen cells the brush and the depth test work in (CSS px). */
const CELL_PX = 3;

/** What the controller needs of a CesiumJS viewer or widget. */
export interface SelectViewer {
  readonly scene: Scene;
  readonly camera: Camera;
  readonly canvas: HTMLCanvasElement;
}

export interface SceneSelectOptions {
  /** Whether clicks select now (measuring and exploring take the pointer). */
  enabled?: () => boolean;
}

/** One scan's view while painting: its tiles projected, and the brush. */
interface PaintView {
  assetId: string;
  doc: InstancesDoc;
  tiles: readonly PickTile[];
  screen: ScreenSplats;
  visible: Uint8Array;
  /** Per projected splat, its instance id in `doc`. */
  ids: Uint32Array;
  mask: BrushMask;
  /** The view it was projected from. */
  key: string;
}

/** What a stroke painted, kept for "Use painted area". */
interface Painted {
  assetId: string;
  tiles: readonly PickTile[];
  screen: ScreenSplats;
  painted: Uint8Array;
}

function editable(target: EventTarget | null): boolean {
  const element = target as HTMLElement | null;
  if (!element || typeof element.closest !== "function") return false;
  return element.closest("input, textarea, select, [contenteditable='true']") !== null;
}

/** Instance ids of a tile in a document, decoded once per document. */
class TileIds {
  #doc: InstancesDoc | null = null;
  #ids = new Map<string, Uint32Array | null>();
  of(doc: InstancesDoc, tile: PickTile): Uint32Array | null {
    if (doc !== this.#doc) {
      this.#doc = doc;
      this.#ids.clear();
    }
    let ids = this.#ids.get(tile.checksum);
    if (ids === undefined) {
      const decoded = tileInstanceIds(doc, tile.checksum);
      ids = decoded?.length === tile.count ? decoded : null;
      this.#ids.set(tile.checksum, ids);
    }
    return ids;
  }
}

export class SceneSelectController {
  readonly #viewer: SelectViewer;
  readonly #enabled: () => boolean;
  readonly #ids = new TileIds();
  readonly #off: (() => void)[] = [];
  #down: { x: number; y: number; at: number; id: number } | null = null;
  #stroke: { x: number; y: number; value: 0 | 1 } | null = null;
  #paintView: PaintView | null = null;
  #painted: Painted | null = null;
  #overlay: HTMLCanvasElement | null = null;
  /** The highlight this controller set: cleared when the selection is. */
  #lit: { assetId: string; id: number } | null = null;
  #cameraInputs: boolean | null = null;

  constructor(viewer: SelectViewer, options: SceneSelectOptions = {}) {
    this.#viewer = viewer;
    this.#enabled = options.enabled ?? (() => true);
    const canvas = viewer.canvas;
    const on = <K extends keyof HTMLElementEventMap>(
      target: HTMLElement | Window,
      type: K,
      listener: (event: HTMLElementEventMap[K]) => void,
      capture = false,
    ): void => {
      target.addEventListener(type, listener as EventListener, { capture, passive: false });
      this.#off.push(() =>
        target.removeEventListener(type, listener as EventListener, { capture }),
      );
    };
    on(canvas, "pointerdown", (e) => this.#onDown(e), true);
    on(window, "pointermove", (e) => this.#onMove(e));
    on(window, "pointerup", (e) => this.#onUp(e));
    on(window, "pointercancel", () => this.#endStroke(false));
    on(window, "keydown", (e) => this.#onKey(e));
    on(window, "wheel", (e) => this.#onWheel(e), true);
    this.#off.push(useSceneSelect.subscribe((state, previous) => this.#follow(state, previous)));
  }

  destroy(): void {
    for (const off of this.#off.splice(0)) off();
    this.#setCameraInputs(true);
    this.#overlay?.remove();
    this.#overlay = null;
  }

  // ---- Picking ---------------------------------------------------------------------------

  /** Picks at (`x`, `y`), CSS px from the canvas's top left; true when an object was hit. */
  pickAt(x: number, y: number, anchor?: { x: number; y: number }): boolean {
    const camera = this.#viewer.camera;
    const ray = camera.getPickRay(new Cartesian2(x, y));
    if (!ray) return false;
    const pixelAngle = this.#pixelAngle();
    let best: {
      assetId: string;
      doc: InstancesDoc;
      t: number;
      weights: Map<number, { weight: number; t: number }>;
    } | null = null;
    for (const assetId of this.#assets()) {
      const source = pickSourceOf(assetId);
      const doc = paintedDocOf(assetId);
      const toWorld = source?.toWorld();
      if (!source || !doc || !toWorld) continue;
      const tiles = source.tiles();
      if (tiles.length === 0) continue;
      const toLocal = Matrix4.inverseTransformation(toWorld, new Matrix4());
      const origin = Matrix4.multiplyByPoint(toLocal, ray.origin, new Cartesian3());
      const direction = Matrix4.multiplyByPointAsVector(toLocal, ray.direction, new Cartesian3());
      const ids = tiles.map((tile) => this.#ids.of(doc, tile));
      const hidden = this.#hiddenOf(assetId, doc);
      const hits = castRay(
        tiles,
        {
          origin: [origin.x, origin.y, origin.z],
          direction: [direction.x, direction.y, direction.z],
        },
        {
          pixelAngle,
          include: hidden.size ? (tile, index) => !hidden.has(ids[tile]?.[index] ?? 0) : undefined,
        },
      );
      const front = hits[0];
      if (!front) continue;
      const weights = hitWeights(hits, (hit) => ids[hit.tile]?.[hit.index] ?? 0);
      if (!best || front.t < best.t) best = { assetId, doc, t: front.t, weights };
    }
    if (!best) {
      this.clear();
      return false;
    }
    const candidates = buildCandidates(best.doc, best.weights);
    if (candidates.ids.length === 0) {
      this.clear();
      return false;
    }
    const { assetId, doc } = best;
    const index = defaultIndex(candidates, (id) => this.pixelsOf(assetId, doc, id));
    useSceneSelect
      .getState()
      .select(assetId, candidates.ids, candidates.chain, index, anchor ?? { x, y });
    log.info("picked", { asset: assetId, candidates: candidates.ids, index });
    return true;
  }

  /** How big instance `id` is on screen: its bounds' largest extent, CSS px. */
  pixelsOf(assetId: string, doc: InstancesDoc, id: number): number {
    const instance = doc.byId.get(id);
    const toWorld = pickSourceOf(assetId)?.toWorld();
    if (!instance || !toWorld) return 0;
    const { min, max } = instance.bounds;
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    const scratch = new Cartesian3();
    const screen = new Cartesian2();
    for (const x of [min[0], max[0]])
      for (const y of [min[1], max[1]])
        for (const z of [min[2], max[2]]) {
          const world = Matrix4.multiplyByPoint(toWorld, Cartesian3.fromElements(x, y, z), scratch);
          const at = SceneTransforms.worldToWindowCoordinates(this.#viewer.scene, world, screen);
          if (!at) continue;
          x0 = Math.min(x0, at.x);
          y0 = Math.min(y0, at.y);
          x1 = Math.max(x1, at.x);
          y1 = Math.max(y1, at.y);
        }
    if (!Number.isFinite(x0)) return 0;
    return Math.max(x1 - x0, y1 - y0);
  }

  // ---- Actions ---------------------------------------------------------------------------

  /** The selected instance and its scan, or null. */
  selection(): { assetId: string; id: number; doc: InstancesDoc } | null {
    const state = useSceneSelect.getState();
    const id = selectedId(state);
    const assetId = state.assetId;
    const doc = assetId ? paintedDocOf(assetId) : undefined;
    if (id === null || !assetId || !doc) return null;
    return { assetId, id, doc };
  }

  cycle(step: number): void {
    useSceneSelect.getState().cycle(step);
  }

  clear(): void {
    useSceneSelect.getState().clear();
  }

  /** Hides the selection (with what it contains) and clears it. */
  hide(): void {
    const s = this.selection();
    if (!s) return;
    useInstances.getState().setHidden(s.assetId, [...withDescendants(s.doc, [s.id])], true);
    this.clear();
    // What is visible changed: the brush starts again on what is left.
    this.#paintView = null;
    this.#painted = null;
    this.#drawOverlay();
  }

  /** Hides everything but the selection. */
  showOnly(): void {
    const s = this.selection();
    if (!s) return;
    const store = useInstances.getState();
    store.showAll(s.assetId);
    store.setHidden(s.assetId, hiddenForShowOnly(s.doc, withDescendants(s.doc, [s.id])), true);
  }

  /** Shows everything of the selection's scan again. */
  showAll(): void {
    const assetId = useSceneSelect.getState().assetId;
    if (assetId) useInstances.getState().showAll(assetId);
  }

  flyTo(): void {
    const s = this.selection();
    if (!s) return;
    const sphere = instanceSphere(s.assetId, s.id);
    if (!sphere) return;
    const camera = this.#viewer.camera;
    camera.flyToBoundingSphere(BoundingSphere.clone(sphere), {
      offset: new HeadingPitchRange(camera.heading, CesiumMath.toRadians(-35), 0),
      duration: 1.2,
    });
  }

  /** Whether the selection is an object painted in this browser. */
  selectionIsPainted(): boolean {
    const s = this.selection();
    const base = s ? instancesDocOf(s.assetId) : undefined;
    return s !== null && base !== undefined && isCustomId(base, s.id);
  }

  /** Forgets the selected painted object. */
  deletePainted(): void {
    const s = this.selection();
    const base = s ? instancesDocOf(s.assetId) : undefined;
    if (!s || !base || !isCustomId(base, s.id)) return;
    const store = useSceneSelect.getState();
    const set = store.customOf(s.assetId)[s.id - base.maxId - 1];
    this.clear();
    if (set) store.removeCustom(s.assetId, set.key);
  }

  // ---- Painting --------------------------------------------------------------------------

  setPainting(on: boolean): void {
    const store = useSceneSelect.getState();
    if ((store.mode === "paint") === on) return;
    store.setMode(on ? "paint" : "pick");
  }

  /** Paints a stroke through `points` (CSS px from the canvas's top left), then matches it. */
  paintStroke(
    points: readonly { x: number; y: number }[],
    mode: "replace" | "add" | "subtract",
  ): void {
    const first = points[0];
    if (!first) return;
    this.#beginStroke(first.x, first.y, mode);
    for (const p of points.slice(1)) this.#strokeTo(p.x, p.y);
    this.#endStroke(true);
  }

  /** Makes the last painted area an object of its own and selects it. */
  usePaintedArea(name?: string): number | null {
    const painted = this.#painted;
    const base = painted ? instancesDocOf(painted.assetId) : undefined;
    if (!painted || !base) return null;
    const perTile = new Map<number, number[]>();
    const min: [number, number, number] = [Infinity, Infinity, Infinity];
    const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    const { screen } = painted;
    for (let k = 0; k < screen.count; k++) {
      if (!painted.painted[k]) continue;
      const t = screen.tile[k] ?? 0;
      const i = screen.index[k] ?? 0;
      const tile = painted.tiles[t];
      if (!tile) continue;
      const list = perTile.get(t) ?? [];
      list.push(i);
      perTile.set(t, list);
      for (let a = 0; a < 3; a++) {
        const v = tile.positions[i * 3 + a] ?? 0;
        min[a] = Math.min(min[a] ?? v, v);
        max[a] = Math.max(max[a] ?? v, v);
      }
    }
    const tiles: Record<string, number[]> = {};
    let splats = 0;
    for (const [t, indices] of perTile) {
      const tile = painted.tiles[t];
      if (!tile) continue;
      // A tile drawn twice (a renderer's handover) is listed once.
      tiles[tile.checksum] = encodeRanges([
        ...decodeRanges(tiles[tile.checksum] ?? []),
        ...indices,
      ]);
    }
    for (const ranges of Object.values(tiles)) splats += rangesLength(ranges);
    if (splats === 0) return null;
    const store = useSceneSelect.getState();
    const existing = store.customOf(painted.assetId);
    const set: CustomSet = {
      key: `p${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`,
      name: name ?? `Painted area ${String(existing.length + 1)}`,
      tiles,
      splats,
      bounds: { min, max },
      created: Date.now(),
    };
    store.addCustom(painted.assetId, set);
    const id = customId(base, existing.length);
    store.select(painted.assetId, [id], 1, 0, null);
    store.setPaint(null);
    this.#painted = null;
    this.#paintView?.mask.clear();
    this.#drawOverlay();
    log.info("painted object made", { asset: painted.assetId, id, splats });
    return id;
  }

  // ---- Internals -------------------------------------------------------------------------

  #assets(): string[] {
    const tables = useInstances.getState().assets;
    return pickAssets().filter((assetId) => tables[assetId] !== undefined);
  }

  #pixelAngle(): number {
    const frustum = this.#viewer.camera.frustum as { fovy?: number };
    return (frustum.fovy ?? Math.PI / 3) / Math.max(1, this.#viewer.canvas.clientHeight);
  }

  /** Every instance id not drawn now (the store's hidden set with what it contains). */
  #hiddenOf(assetId: string, doc: InstancesDoc): Set<number> {
    const hidden = useInstances.getState().assets[assetId]?.hidden;
    return hidden && hidden.size > 0 ? withDescendants(doc, hidden) : new Set();
  }

  #local(e: PointerEvent | WheelEvent): { x: number; y: number } {
    const rect = this.#viewer.canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  #onDown(e: PointerEvent): void {
    if (e.button !== 0 || !this.#enabled()) return;
    const p = this.#local(e);
    if (useSceneSelect.getState().mode === "paint") {
      e.preventDefault();
      e.stopPropagation();
      this.#beginStroke(p.x, p.y, e.altKey ? "subtract" : e.shiftKey ? "add" : "replace");
      return;
    }
    this.#down = { ...p, at: performance.now(), id: e.pointerId };
  }

  #onMove(e: PointerEvent): void {
    if (this.#stroke) {
      const p = this.#local(e);
      this.#strokeTo(p.x, p.y);
    }
  }

  #onUp(e: PointerEvent): void {
    if (this.#stroke) {
      this.#endStroke(true);
      return;
    }
    const down = this.#down;
    this.#down = null;
    if (down?.id !== e.pointerId || e.button !== 0 || !this.#enabled()) return;
    const p = this.#local(e);
    if (Math.hypot(p.x - down.x, p.y - down.y) > CLICK_PX) return;
    if (performance.now() - down.at > CLICK_MS) return;
    this.pickAt(p.x, p.y, { x: e.clientX, y: e.clientY });
  }

  #onKey(e: KeyboardEvent): void {
    if (e.defaultPrevented || editable(e.target) || e.ctrlKey || e.metaKey) return;
    const state = useSceneSelect.getState();
    const has = state.index >= 0;
    if ((e.key === "b" || e.key === "B") && !e.altKey) {
      if (this.#assets().length === 0) return;
      this.setPainting(state.mode !== "paint");
      e.preventDefault();
      return;
    }
    if (e.key === "Escape") {
      if (state.mode === "paint") this.setPainting(false);
      else if (has) this.clear();
      else return;
      e.preventDefault();
      return;
    }
    if (!has || state.candidates.length < 2) return;
    let step = 0;
    if (e.key === "]" || (e.key === "Tab" && !e.shiftKey)) step = 1;
    else if (e.key === "[" || (e.key === "Tab" && e.shiftKey)) step = -1;
    if (step === 0) return;
    // Tab moves focus only from the scene: inside a panel it keeps its meaning.
    if (e.key === "Tab" && document.activeElement && document.activeElement !== document.body)
      return;
    e.preventDefault();
    this.cycle(step);
  }

  #onWheel(e: WheelEvent): void {
    const state = useSceneSelect.getState();
    if (!e.altKey || e.deltaY === 0) return;
    if (state.mode === "paint") {
      state.setBrush(state.brush * (e.deltaY > 0 ? 0.85 : 1.18));
    } else if (state.index >= 0 && state.candidates.length > 1) {
      this.cycle(e.deltaY > 0 ? 1 : -1);
    } else {
      return;
    }
    e.preventDefault();
    e.stopPropagation();
  }

  /** Keeps the highlight, the camera's inputs and the brush on what the store says. */
  #follow(
    state: ReturnType<typeof useSceneSelect.getState>,
    previous: ReturnType<typeof useSceneSelect.getState>,
  ): void {
    const id = selectedId(state);
    if (id !== selectedId(previous) || state.assetId !== previous.assetId) {
      const lit = this.#lit;
      if (lit && (lit.assetId !== state.assetId || id === null)) {
        useInstances.getState().highlight(lit.assetId, []);
        this.#lit = null;
      }
      const doc = state.assetId ? paintedDocOf(state.assetId) : undefined;
      if (id !== null && state.assetId && doc) {
        useInstances.getState().highlight(state.assetId, [...withDescendants(doc, [id])]);
        this.#lit = { assetId: state.assetId, id };
      }
      this.#viewer.scene.requestRender();
    }
    if (state.mode !== previous.mode) {
      const painting = state.mode === "paint";
      this.#setCameraInputs(!painting);
      if (!painting) {
        this.#paintView = null;
        this.#painted = null;
        this.#stroke = null;
      }
      this.#drawOverlay();
    }
  }

  #setCameraInputs(on: boolean): void {
    const controller = this.#viewer.scene.screenSpaceCameraController;
    if (!on) {
      this.#cameraInputs ??= controller.enableInputs;
      controller.enableInputs = false;
    } else if (this.#cameraInputs !== null) {
      controller.enableInputs = this.#cameraInputs;
      this.#cameraInputs = null;
    }
  }

  #viewKey(): string {
    const camera = this.#viewer.camera;
    const p = camera.positionWC;
    const d = camera.directionWC;
    const c = this.#viewer.canvas;
    return [p.x, p.y, p.z, d.x, d.y, d.z, c.clientWidth, c.clientHeight]
      .map((v) => v.toFixed(4))
      .join(",");
  }

  /** The scan under the brush, projected from the current view (once per view). */
  #paintViewNow(): PaintView | null {
    const key = this.#viewKey();
    if (this.#paintView?.key === key) return this.#paintView;
    const camera = this.#viewer.camera;
    const width = this.#viewer.canvas.clientWidth;
    const height = this.#viewer.canvas.clientHeight;
    let chosen: PaintView | null = null;
    for (const assetId of this.#assets()) {
      const source = pickSourceOf(assetId);
      const doc = paintedDocOf(assetId);
      const toWorld = source?.toWorld();
      if (!source || !doc || !toWorld) continue;
      const tiles = source.tiles();
      if (tiles.length === 0) continue;
      const viewProj = Matrix4.multiply(
        camera.frustum.projectionMatrix,
        Matrix4.multiply(camera.viewMatrix, toWorld, new Matrix4()),
        new Matrix4(),
      );
      const ids = tiles.map((tile) => this.#ids.of(doc, tile));
      const hidden = this.#hiddenOf(assetId, doc);
      const screen = projectTiles(
        tiles,
        Matrix4.toArray(viewProj),
        width,
        height,
        CELL_PX,
        hidden.size ? (tile, index) => !hidden.has(ids[tile]?.[index] ?? 0) : undefined,
      );
      if (chosen && chosen.screen.count >= screen.count) continue;
      const perSplat = new Uint32Array(screen.count);
      for (let k = 0; k < screen.count; k++) {
        perSplat[k] = ids[screen.tile[k] ?? 0]?.[screen.index[k] ?? 0] ?? 0;
      }
      chosen = {
        assetId,
        doc,
        tiles,
        screen,
        visible: visibleSplats(screen),
        ids: perSplat,
        mask: new BrushMask(screen.cols, screen.rows, CELL_PX),
        key,
      };
    }
    this.#paintView = chosen;
    return chosen;
  }

  #beginStroke(x: number, y: number, mode: "replace" | "add" | "subtract"): void {
    const view = this.#paintViewNow();
    if (!view) return;
    if (mode === "replace") view.mask.clear();
    const value = mode === "subtract" ? 0 : 1;
    this.#stroke = { x, y, value };
    view.mask.stamp(x, y, useSceneSelect.getState().brush, value);
    this.#drawOverlay();
  }

  #strokeTo(x: number, y: number): void {
    const stroke = this.#stroke;
    const view = this.#paintView;
    if (!stroke || !view) return;
    view.mask.line(stroke.x, stroke.y, x, y, useSceneSelect.getState().brush, stroke.value);
    stroke.x = x;
    stroke.y = y;
    this.#drawOverlay();
  }

  #endStroke(evaluate: boolean): void {
    this.#stroke = null;
    const view = this.#paintView;
    if (!evaluate || !view) return;
    const painted = paintedSplats(view.screen, view.visible, view.mask);
    const weights = new Float32Array(view.screen.count);
    let count = 0;
    for (let k = 0; k < view.screen.count; k++) {
      weights[k] = view.visible[k] ? (view.screen.opacity[k] ?? 0) : 0;
      if (painted[k]) count++;
    }
    const best = bestByIoU(view.doc, { ids: view.ids, weights, painted });
    const store = useSceneSelect.getState();
    store.setPaint({ best: best?.id ?? null, iou: best?.iou ?? 0, painted: count });
    this.#painted =
      count > 0 ? { assetId: view.assetId, tiles: view.tiles, screen: view.screen, painted } : null;
    if (best) {
      const chain = chainOf(view.doc, best.id);
      store.select(view.assetId, chain, chain.length, 0, null);
    } else {
      store.clear();
      store.setPaint({ best: null, iou: 0, painted: count });
    }
    log.info("painted", { asset: view.assetId, best: best?.id, iou: best?.iou, splats: count });
  }

  /** Whether the last painted area matched no object well (`PAINT_MIN_IOU`). */
  get paintedAreaOffered(): boolean {
    const paint = useSceneSelect.getState().paint;
    return this.#painted !== null && paint !== null && paint.iou < PAINT_MIN_IOU;
  }

  /** Draws the painted cells over the scene while painting. */
  #drawOverlay(): void {
    const painting = useSceneSelect.getState().mode === "paint";
    const view = this.#paintView;
    if (!painting || !view) {
      this.#overlay?.remove();
      this.#overlay = null;
      return;
    }
    const canvas = this.#viewer.canvas;
    let overlay = this.#overlay;
    if (!overlay) {
      overlay = document.createElement("canvas");
      overlay.dataset.sceneSelectBrush = "";
      Object.assign(overlay.style, {
        position: "absolute",
        inset: "0",
        width: "100%",
        height: "100%",
        pointerEvents: "none",
      });
      (canvas.parentElement ?? document.body).appendChild(overlay);
      this.#overlay = overlay;
    }
    const { cols, rows, data } = view.mask;
    overlay.width = cols;
    overlay.height = rows;
    overlay.style.imageRendering = "pixelated";
    const context = overlay.getContext("2d");
    if (!context) return;
    const image = context.createImageData(cols, rows);
    for (let i = 0; i < data.length; i++) {
      if (!data[i]) continue;
      image.data.set([90, 170, 255, 90], i * 4);
    }
    context.putImageData(image, 0, 0);
  }
}
