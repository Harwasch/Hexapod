/**
 * Selecting a scan's objects in the scene (docs/SCENE_OBJECTS.md, "Selecting in the scene"),
 * under whichever renderer draws the scan: a click picks the splats under the cursor, a brush
 * paints them. Both read the tiles drawn now from the renderer's pick source
 * (pickSources.ts) and test them on the CPU in the scan's frame (lib/splatPick.ts,
 * lib/splatPaint.ts), so CesiumJS, PlayCanvas and Spark behave alike.
 *
 * - **Click**: the ray through the cursor is composited front to back over the splats it
 *   meets; the instance with most of the pixel, its chain up to the top level and the other
 *   instances met near the front are the candidates (lib/sceneSelect.ts). The whole object
 *   (the top of the chain) is chosen first, and each click again on it goes one level finer
 *   toward what was hit; a double-click is one click. `[` (finer) / `]` (coarser, then the
 *   nearby instances), Alt and the wheel, or Tab / Shift+Tab while the map or the selection
 *   card has focus, cycle; Escape clears. A hit gives the keyboard to the map (the canvas),
 *   so Tab cycles straight away; from anywhere else Tab moves focus as it always does. In
 *   the app the click is the map's own (SelectionManager asks `click` first, so a hit on the
 *   scan takes the click from the Location and site cards); standalone it is a press
 *   released where it began.
 * - **Brush** (`B`, or the card's Brush): strokes on screen collect the front-most splats under
 *   them -- Shift (or the card's Add) adds to the painted area, Alt (or Remove) takes away, a
 *   plain stroke starts again -- and the combination of instances (at whatever levels fit)
 *   whose union has the best intersection over union with it is selected: one instance, or
 *   several (both flanges of a spool) when together they match better (lib/sceneSelect.ts
 *   `bestSet`). While a stroke is painted its best match so far is highlighted, held steady
 *   between near-equal answers (`steadySet`), so painting can stop once the right objects
 *   light up. Below `PAINT_MIN_IOU` the painted splats can be kept as an object of their own,
 *   and a combination can be kept as one too (lib/customSets.ts), drawn through the same hide
 *   and highlight pipeline.
 *
 * What is selected (`selectedIds`: the chosen candidate, or the combination) is the objects
 * store's highlight (`state/instances.ts`); the HUD's selection card
 * (features/sites/ObjectCard.tsx) shows it and offers Hide, Show only, Fly to, the brush and
 * Clear. In the app `B` and Escape are the app's keys (`ownKeys: false`): the hotkey registry
 * binds the brush, and Escape steps back through GlobalHotkeys' chain, so one press does one
 * thing.
 */

import {
  BoundingSphere,
  Cartesian2,
  Cartesian3,
  HeadingPitchRange,
  Math as CesiumMath,
  Matrix4,
  type Camera,
  type Scene,
} from "cesium";

import {
  customId,
  decodeRanges,
  encodeRanges,
  isCustomId,
  rangesLength,
  setFromInstances,
  withCustomSets,
  type CustomSet,
} from "@/lib/customSets";
import { tileInstanceIds, withDescendants, type InstancesDoc } from "@/lib/instances";
import { createLogger } from "@/lib/log";
import { supersededIdOf } from "@/lib/supersedes";
import {
  buildCandidates,
  chainOf,
  commonChain,
  drillIndex,
  hiddenForShowOnly,
  PAINT_MIN_IOU,
  paintedCount,
  paintIndex,
  paintSumsIndexed,
  sameMembers,
  splatShares,
  steadySet,
  type PaintIndex,
  type PaintSetMatch,
} from "@/lib/sceneSelect";
import {
  BrushMask,
  paintedSplats,
  projectTiles,
  visibleSplats,
  type ScreenSplats,
} from "@/lib/splatPaint";
import { castRay, hitWeights, labelsNear, type PickTile } from "@/lib/splatPick";
import { useInstances } from "@/state/instances";
import { chosenCombination, selectedId, selectedIds, useSceneSelect } from "@/state/sceneSelect";
import { useSupersedes } from "@/state/supersedes";

import { uniformScale } from "../placement";
import { paintedDocOf } from "../scanView/scanInstances";
import { instanceSphere, instancesDocOf } from "../splatInstances";
import { inverseScaledTransformation } from "../tilesetScale";
import { pickAssets, pickSourceOf } from "./pickSources";

const log = createLogger("scene-select");

/**
 * A press released less than this (CSS px) from where it began is a click, however long it
 * took, as CesiumJS's own click: a frame of a big scan on a slow GPU holds the page for a
 * second or more, so a release is often handled long after it happened.
 */
const CLICK_PX = 5;
/**
 * A click this soon (ms) and this near (CSS px) after the last one is the same click: a
 * double-click is two clicks (and the map's double-click asks once more), and a click drills
 * one level, not two.
 */
const REPEAT_CLICK_MS = 400;
const REPEAT_CLICK_PX = 6;
/**
 * A click whose labelled splats are less than this share of its unlabelled ones is labelled
 * from the splats drawn around it instead (`labelsNear`): within the first of
 * `NEAR_LABEL_PIXELS` pixels' width at its distance that finds any (never less than
 * `NEAR_LABEL_MIN_M`). From 400 m the camp's coarse canopy splats sit 8 to 17 m from the
 * nearest labelled one, some 15 to 30 pixels.
 */
const UNLABELLED_SHARE = 0.25;
const NEAR_LABEL_PIXELS = [6, 24, 64] as const;
const NEAR_LABEL_MIN_M = 0.15;
/** Screen cells the brush and the depth test work in (CSS px). */
const CELL_PX = 3;
/** While a stroke is painted, its match is brought up to date at most this often (ms). */
const PREVIEW_MS = 100;

/** What the controller needs of a CesiumJS viewer or widget. */
export interface SelectViewer {
  readonly scene: Scene;
  readonly camera: Camera;
  readonly canvas: HTMLCanvasElement;
}

export interface SceneSelectOptions {
  /** Whether clicks select now (measuring and exploring take the pointer). */
  enabled?: () => boolean;
  /**
   * Whether the controller tells clicks from drags itself (true, the default), or is handed
   * them through `click` (the app: SelectionManager's click, so one click opens one thing).
   */
  ownClicks?: boolean;
  /**
   * Whether `B` (the brush) and Escape are the controller's own keys (true, the default), or
   * the app's (false): the app binds `B` from its hotkey registry and routes Escape through
   * its step-back chain, so a press is answered once (`togglePainting`, `escape`).
   */
  ownKeys?: boolean;
  /**
   * How Fly to moves the camera: the app's camera controller (`CameraController.flyToObject`,
   * the pace, range and destination prefetch of every fly-to there). Standalone, CesiumJS's
   * own flight.
   */
  fly?: (sphere: BoundingSphere) => void;
}

/** Marks an element whose focus counts as the scene's for Tab (the selection card). */
export const SCENE_FOCUS_ATTRIBUTE = "data-scene-select-focus";

/** One scan's view while painting: its tiles projected, and the brush. */
interface PaintView {
  assetId: string;
  doc: InstancesDoc;
  tiles: readonly PickTile[];
  screen: ScreenSplats;
  visible: Uint8Array;
  /** The visible splats by cell, with their instance ids in `doc`, for the match. */
  index: PaintIndex;
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

/**
 * Whether Tab cycles the candidates: only while the map (the canvas) or the selection card
 * itself has focus. Anywhere else, the page's body included, Tab moves focus as it always does
 * -- cycling from the body trapped the keyboard on the map.
 */
function tabCycles(canvas: HTMLCanvasElement): boolean {
  const active = document.activeElement;
  return (
    active === canvas ||
    (active instanceof HTMLElement && active.hasAttribute(SCENE_FOCUS_ATTRIBUTE))
  );
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
  readonly #ownClicks: boolean;
  readonly #ownKeys: boolean;
  readonly #fly: ((sphere: BoundingSphere) => void) | null;
  readonly #ids = new TileIds();
  readonly #off: (() => void)[] = [];
  #down: { x: number; y: number; id: number } | null = null;
  #stroke: { x: number; y: number; value: 0 | 1 } | null = null;
  #paintView: PaintView | null = null;
  #painted: Painted | null = null;
  #overlay: HTMLCanvasElement | null = null;
  /** The brush overlay's pixels, kept between strokes (one buffer a size, not one a move). */
  #overlayImage: ImageData | null = null;
  /** Removes the wheel listener while one is attached (`#syncWheel`). */
  #wheelOff: (() => void) | null = null;
  /** The highlight this controller set (what is selected): cleared when the selection is. */
  #lit: { assetId: string; ids: readonly number[] } | null = null;
  #cameraInputs: boolean | null = null;
  /** The last click taken, and what it answered (`click`: a repeat is the same click). */
  #lastClick: { x: number; y: number; at: number; hit: boolean } | null = null;
  /** The pending frame that brings a stroke's match up to date, and when it last ran. */
  #previewFrame: number | null = null;
  #previewAt = -Infinity;
  /** What the stroke's match lit (no ids: nothing), until the stroke ends. */
  #previewed: { assetId: string; ids: readonly number[] } | null = null;
  /** What bringing the match up to date cost this stroke. */
  #previewCost = { runs: 0, maxMs: 0 };

  constructor(viewer: SelectViewer, options: SceneSelectOptions = {}) {
    this.#viewer = viewer;
    this.#enabled = options.enabled ?? (() => true);
    this.#ownClicks = options.ownClicks ?? true;
    this.#ownKeys = options.ownKeys ?? true;
    this.#fly = options.fly ?? null;
    const canvas = viewer.canvas;
    // Focusable, out of the tab order: a hit gives the map the keyboard (`#focusScene`), and a
    // click on it takes focus from whatever had it, as CesiumJS's own blur on press intends.
    if (!canvas.hasAttribute("tabindex")) {
      canvas.tabIndex = -1;
      this.#off.push(() => canvas.removeAttribute("tabindex"));
    }
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
    this.#off.push(useSceneSelect.subscribe((state, previous) => this.#follow(state, previous)));
    // A fill's swap changes what is drawn (lib/supersedes.ts): the brush projects anew.
    this.#off.push(
      useSupersedes.subscribe(() => {
        this.#paintView = null;
      }),
    );
    this.#syncWheel(useSceneSelect.getState());
  }

  destroy(): void {
    for (const off of this.#off.splice(0)) off();
    this.#endPreview();
    this.#wheelOff?.();
    this.#wheelOff = null;
    this.#setCameraInputs(true);
    this.#overlay?.remove();
    this.#overlay = null;
    this.#overlayImage = null;
  }

  /** Whether a wheel listener is attached now (tests). */
  get listensToWheel(): boolean {
    return this.#wheelOff !== null;
  }

  /**
   * The wheel (with Alt) sizes the brush or cycles the candidates -- so it is listened to only
   * while there is a brush or more than one candidate. A window listener in the capture phase
   * that may cancel (`passive: false`) makes the browser wait for script on every wheel tick
   * before it scrolls or zooms anything; attached for good, it did so for the globe's own zoom
   * all the time.
   */
  #syncWheel(state: ReturnType<typeof useSceneSelect.getState>): void {
    const wanted = state.mode === "paint" || (state.index >= 0 && state.candidates.length > 1);
    if (wanted && !this.#wheelOff) {
      const listener = (e: WheelEvent): void => this.#onWheel(e);
      window.addEventListener("wheel", listener, { capture: true, passive: false });
      this.#wheelOff = () => window.removeEventListener("wheel", listener, { capture: true });
    } else if (!wanted && this.#wheelOff) {
      this.#wheelOff();
      this.#wheelOff = null;
    }
  }

  // ---- Picking ---------------------------------------------------------------------------

  /**
   * A click at (`x`, `y`), CSS px from the canvas's top left, at time `at` (ms, as
   * `performance.now()`): picks there unless painting or disabled. True when the click is
   * taken: an object of a scan was selected, or the brush is out. A click repeated within
   * `REPEAT_CLICK_MS` and `REPEAT_CLICK_PX` of the last is answered as that one was and picks
   * nothing, so a double-click does not drill two levels.
   */
  click(x: number, y: number, at = performance.now()): boolean {
    if (!this.#enabled()) return false;
    // While painting, a dab of the brush is not a click on the map.
    if (useSceneSelect.getState().mode === "paint") return true;
    const last = this.#lastClick;
    if (
      last &&
      at - last.at < REPEAT_CLICK_MS &&
      Math.hypot(x - last.x, y - last.y) <= REPEAT_CLICK_PX
    ) {
      last.at = at;
      return last.hit;
    }
    const rect = this.#viewer.canvas.getBoundingClientRect();
    const hit = this.pickAt(x, y, { x: x + rect.left, y: y + rect.top });
    this.#lastClick = { x, y, at, hit };
    if (hit) this.#focusScene();
    return hit;
  }

  /**
   * Gives the keyboard to the map after a hit: a search box still focused would take the
   * cycling keys, and on the body Tab moves focus rather than cycling. On the canvas `[`, `]`
   * and Tab cycle, the arrows still pan (KeyboardNavigator counts it as the map), and Escape
   * clears.
   */
  #focusScene(): void {
    const canvas = this.#viewer.canvas;
    if (document.activeElement === canvas) return;
    const active = document.activeElement;
    if (active instanceof HTMLElement && editable(active)) active.blur();
    canvas.focus({ preventScroll: true });
  }

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
      // The ray in the scan's frame, where a unit is `scale` metres under a runtime scale.
      const toLocal = inverseScaledTransformation(toWorld, new Matrix4());
      const scale = uniformScale(toWorld);
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
      const idOf = (tile: number, index: number): number => ids[tile]?.[index] ?? 0;
      let weights = hitWeights(hits, (hit) => idOf(hit.tile, hit.index));
      // Mostly unlabelled (a coarse level of detail's merged splats, far off): the labelled
      // splats drawn around where the ray met the scan say what is there.
      const unlabelled = weights.get(0)?.weight ?? 0;
      let labelled = 0;
      for (const [id, w] of weights) if (id !== 0) labelled += w.weight;
      if (labelled < UNLABELLED_SHARE * unlabelled) {
        const d = Math.hypot(direction.x, direction.y, direction.z) || 1;
        const point: [number, number, number] = [
          origin.x + (direction.x / d) * front.t,
          origin.y + (direction.y / d) * front.t,
          origin.z + (direction.z / d) * front.t,
        ];
        // Widening: far off, a coarse tile's labelled splats can be metres apart.
        let found = false;
        for (const pixels of NEAR_LABEL_PIXELS) {
          const radius = Math.max(NEAR_LABEL_MIN_M / scale, front.t * pixelAngle * pixels);
          const near = labelsNear(
            tiles,
            point,
            radius,
            idOf,
            hidden.size ? (tile, index) => !hidden.has(idOf(tile, index)) : undefined,
          );
          if (near.size === 0) continue;
          for (const entry of near.values()) entry.t = front.t;
          weights = near;
          found = true;
          break;
        }
        // Nothing labelled near where the ray met the scan (the camera inside a coarse
        // canopy's haze): the first labelled object along the ray, the haze let through.
        if (!found) {
          const behind = castRay(
            tiles,
            {
              origin: [origin.x, origin.y, origin.z],
              direction: [direction.x, direction.y, direction.z],
            },
            {
              pixelAngle,
              include: (tile, index) => {
                const id = idOf(tile, index);
                return id !== 0 && !hidden.has(id);
              },
            },
          );
          if (behind.length > 0) weights = hitWeights(behind, (hit) => idOf(hit.tile, hit.index));
        }
      }
      // Scans are compared by how far along the ray they are on the globe.
      const t = front.t * scale;
      if (!best || t < best.t) best = { assetId, doc, t, weights };
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
    const state = useSceneSelect.getState();
    // Drilling goes on only within the scan selected now; on another, or from a combination
    // the brush chose (not a level of any chain), the whole object again.
    const current =
      state.assetId === assetId && !chosenCombination(state) ? selectedId(state) : null;
    const shares = splatShares(doc);
    const index = drillIndex(candidates, current, (id) => shares.get(id) ?? 0);
    state.select(assetId, candidates.ids, candidates.chain, index, anchor ?? { x, y });
    log.info("picked", { asset: assetId, candidates: candidates.ids, index });
    return true;
  }

  // ---- Actions ---------------------------------------------------------------------------

  /**
   * What is selected and its scan, or null: `ids` the instances (a combination's members, else
   * the one chosen), `id` the first of them.
   */
  selection(): { assetId: string; id: number; ids: readonly number[]; doc: InstancesDoc } | null {
    const state = useSceneSelect.getState();
    const ids = selectedIds(state);
    const id = ids[0];
    const assetId = state.assetId;
    const doc = assetId ? paintedDocOf(assetId) : undefined;
    if (id === undefined || !assetId || !doc) return null;
    return { assetId, id, ids, doc };
  }

  cycle(step: number): void {
    useSceneSelect.getState().cycle(step);
  }

  clear(): void {
    useSceneSelect.getState().clear();
  }

  /** Hides the selection (every member of a combination, with what it contains) and clears it. */
  hide(): void {
    const s = this.selection();
    if (!s) return;
    useInstances.getState().setHidden(s.assetId, [...withDescendants(s.doc, s.ids)], true);
    this.clear();
    // What is visible changed: the brush starts again on what is left.
    this.#paintView = null;
    this.#painted = null;
    this.#drawOverlay();
  }

  /** Hides everything but the selection (every member of a combination). */
  showOnly(): void {
    const s = this.selection();
    if (!s) return;
    const store = useInstances.getState();
    store.showAll(s.assetId);
    store.setHidden(s.assetId, hiddenForShowOnly(s.doc, withDescendants(s.doc, s.ids)), true);
  }

  /** Shows everything of the selection's scan again. */
  showAll(): void {
    const assetId = useSceneSelect.getState().assetId;
    if (assetId) useInstances.getState().showAll(assetId);
  }

  /**
   * Flies to the selection. In the app through the camera controller (`fly`), so the flight
   * paces, ranges and prefetches as every other fly-to and a site flight still settling gives
   * the camera up to it; standalone, CesiumJS's own flight.
   */
  flyTo(): void {
    const s = this.selection();
    if (!s) return;
    // A combination: the sphere around its members'.
    const spheres = s.ids
      .map((id) => instanceSphere(s.assetId, id))
      .filter((each): each is BoundingSphere => each !== undefined);
    const sphere = spheres.length > 1 ? BoundingSphere.fromBoundingSpheres(spheres) : spheres[0];
    if (!sphere) return;
    if (this.#fly) {
      this.#fly(BoundingSphere.clone(sphere));
      return;
    }
    const camera = this.#viewer.camera;
    camera.flyToBoundingSphere(BoundingSphere.clone(sphere), {
      offset: new HeadingPitchRange(camera.heading, CesiumMath.toRadians(-35), 0),
      duration: 1.2,
    });
  }

  /** Whether the selection is one object painted in this browser. */
  selectionIsPainted(): boolean {
    const s = this.selection();
    const base = s ? instancesDocOf(s.assetId) : undefined;
    return s?.ids.length === 1 && base !== undefined && isCustomId(base, s.id);
  }

  /** Forgets the selected painted object. */
  deletePainted(): void {
    const s = this.selection();
    const base = s ? instancesDocOf(s.assetId) : undefined;
    if (s?.ids.length !== 1 || !base || !isCustomId(base, s.id)) return;
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

  /** Whether there is a scan with objects to paint over (the brush means nothing without). */
  get canPaint(): boolean {
    return this.#assets().length > 0;
  }

  /**
   * The brush on or off (`B`); true when it changed. It is not taken up without a scan with
   * objects, or while measuring or exploring has the pointer.
   */
  togglePainting(): boolean {
    const painting = useSceneSelect.getState().mode === "paint";
    if (!painting && (!this.#enabled() || !this.canPaint)) return false;
    this.setPainting(!painting);
    return true;
  }

  /**
   * One step back (Escape): the brush away first, else the selection cleared. True when it did
   * something, so the app's chain stops there.
   */
  escape(): boolean {
    const state = useSceneSelect.getState();
    if (state.mode === "paint") this.setPainting(false);
    else if (state.index >= 0) this.clear();
    else return false;
    return true;
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
    this.#afterNewObject();
    log.info("painted object made", { asset: painted.assetId, id, splats });
    return id;
  }

  /**
   * Keeps the selected combination as an object of its own (lib/customSets.ts
   * `setFromInstances`: its members' splats in every tile of the scan) and selects it.
   */
  saveCombination(name?: string): number | null {
    const state = useSceneSelect.getState();
    const combination = chosenCombination(state);
    const assetId = state.assetId;
    const base = assetId ? instancesDocOf(assetId) : undefined;
    if (!combination || !assetId || !base) return null;
    const existing = state.customOf(assetId);
    // Not the drawn document: a kept combination holds every splat of its members, whichever
    // fill is shown and whatever it supersedes now (lib/supersedes.ts).
    const doc = withCustomSets(base, existing);
    const set = setFromInstances(
      doc,
      combination.ids,
      name ?? `Painted area ${String(existing.length + 1)}`,
      `p${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`,
      Date.now(),
    );
    if (!set) return null;
    state.addCustom(assetId, set);
    const id = customId(base, existing.length);
    state.select(assetId, [id], 1, 0, null);
    this.#afterNewObject();
    log.info("combination kept", {
      asset: assetId,
      id,
      members: combination.ids,
      splats: set.splats,
    });
    return id;
  }

  /**
   * After a painted object is made: the brush's match and area are done with, and the view's
   * index is stale (the object's splats carry its id now), so the next stroke projects anew.
   */
  #afterNewObject(): void {
    useSceneSelect.getState().setPaint(null);
    this.#painted = null;
    this.#paintView = null;
    this.#drawOverlay();
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

  /**
   * Every instance id not drawn now (the store's hidden set with what it contains, and the
   * splats a fill supersedes, lib/supersedes.ts).
   */
  #hiddenOf(assetId: string, doc: InstancesDoc): Set<number> {
    const hidden = useInstances.getState().assets[assetId]?.hidden;
    const out = hidden && hidden.size > 0 ? withDescendants(doc, hidden) : new Set<number>();
    const superseded = supersededIdOf(doc);
    if (superseded !== null) out.add(superseded);
    return out;
  }

  #local(e: PointerEvent | WheelEvent): { x: number; y: number } {
    const rect = this.#viewer.canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  #onDown(e: PointerEvent): void {
    if (e.button !== 0 || !this.#enabled()) return;
    const p = this.#local(e);
    const state = useSceneSelect.getState();
    if (state.mode === "paint") {
      e.preventDefault();
      e.stopPropagation();
      // Shift and Alt where there is a keyboard; the card's New / Add / Remove on a touch screen.
      this.#beginStroke(p.x, p.y, e.altKey ? "subtract" : e.shiftKey ? "add" : state.strokeMode);
      return;
    }
    if (this.#ownClicks) this.#down = { ...p, id: e.pointerId };
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
    // When the release happened, not when a slow frame let it be handled.
    this.click(p.x, p.y, e.timeStamp);
  }

  #onKey(e: KeyboardEvent): void {
    if (e.defaultPrevented || editable(e.target) || e.ctrlKey || e.metaKey) return;
    const state = useSceneSelect.getState();
    if (this.#ownKeys) {
      if ((e.key === "b" || e.key === "B") && !e.altKey) {
        if (this.togglePainting()) e.preventDefault();
        return;
      }
      if (e.key === "Escape") {
        if (this.escape()) e.preventDefault();
        return;
      }
    }
    if (state.index < 0 || state.candidates.length < 2) return;
    let step = 0;
    if (e.key === "]" || (e.key === "Tab" && !e.shiftKey)) step = 1;
    else if (e.key === "[" || (e.key === "Tab" && e.shiftKey)) step = -1;
    if (step === 0) return;
    // Tab cycles only from the map or the card; anywhere else it moves focus.
    if (e.key === "Tab" && !tabCycles(this.#viewer.canvas)) return;
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
    this.#syncWheel(state);
    const ids = selectedIds(state);
    if (!sameMembers(ids, selectedIds(previous)) || state.assetId !== previous.assetId) {
      const lit = this.#lit;
      if (lit && (lit.assetId !== state.assetId || ids.length === 0)) {
        useInstances.getState().highlight(lit.assetId, []);
        this.#lit = null;
      }
      const doc = state.assetId ? paintedDocOf(state.assetId) : undefined;
      if (ids.length > 0 && state.assetId && doc) {
        useInstances.getState().highlight(state.assetId, [...withDescendants(doc, ids)]);
        this.#lit = { assetId: state.assetId, ids };
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
        this.#endPreview();
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
      const visible = visibleSplats(screen);
      chosen = {
        assetId,
        doc,
        tiles,
        screen,
        visible,
        index: paintIndex(doc, screen, visible, perSplat),
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
    this.#previewCost = { runs: 0, maxMs: 0 };
    view.mask.stamp(x, y, useSceneSelect.getState().brush, value);
    this.#drawOverlay();
    this.#schedulePreview();
  }

  #strokeTo(x: number, y: number): void {
    const stroke = this.#stroke;
    const view = this.#paintView;
    if (!stroke || !view) return;
    view.mask.line(stroke.x, stroke.y, x, y, useSceneSelect.getState().brush, stroke.value);
    stroke.x = x;
    stroke.y = y;
    this.#drawOverlay();
    this.#schedulePreview();
  }

  /**
   * Brings the stroke's match up to date on a coming frame, at most every `PREVIEW_MS`: moves
   * come far more often than a frame, and the match is wanted while the stroke goes on.
   */
  #schedulePreview(): void {
    if (this.#previewFrame !== null) return;
    const run = (): void => {
      this.#previewFrame = null;
      if (!this.#stroke) return;
      if (performance.now() - this.#previewAt < PREVIEW_MS) {
        this.#previewFrame = requestAnimationFrame(run);
        return;
      }
      this.#preview();
    };
    this.#previewFrame = requestAnimationFrame(run);
  }

  /**
   * The stroke's best match now (lib/sceneSelect.ts `bestSet`), held on what the preview shows
   * while that is still about as good (`steadySet`).
   */
  #match(view: PaintView): PaintSetMatch | null {
    const shown = this.#previewed;
    return steadySet(
      paintSumsIndexed(view.index, view.mask),
      shown?.assetId === view.assetId ? shown.ids : null,
    );
  }

  /**
   * Highlights the stroke's best match so far, through the objects store as the selection is
   * (only the instances' state table changes, which every renderer applies cheaply), and says
   * it on the card (`PaintResult.live`). Nothing is selected until the stroke ends.
   */
  #preview(): void {
    const view = this.#paintView;
    if (!this.#stroke || !view) return;
    const started = performance.now();
    const best = this.#match(view);
    const ids = best?.ids ?? [];
    const shown = this.#previewed;
    if (shown?.assetId !== view.assetId || !sameMembers(shown.ids, ids)) {
      // The store lights what is below it too: its table has the same hierarchy, and a
      // painted object has nothing below it, so the match's ids alone are the selection's
      // highlight without walking the hierarchy twice.
      useInstances.getState().highlight(view.assetId, ids);
      this.#previewed = { assetId: view.assetId, ids };
      this.#viewer.scene.requestRender();
    }
    const painted = paintedCount(view.index, view.mask);
    useSceneSelect
      .getState()
      .setPaint({ ids, best: ids[0] ?? null, iou: best?.iou ?? 0, painted, live: true });
    this.#previewAt = performance.now();
    const ms = this.#previewAt - started;
    this.#previewCost.runs++;
    this.#previewCost.maxMs = Math.max(this.#previewCost.maxMs, ms);
    log.debug("paint preview", { ms, ids: ids.slice(0, 8), iou: best?.iou, splats: painted });
  }

  /**
   * Stops bringing the match up to date, and gives the highlight back to the selection (or
   * clears it) if the stroke's match had taken it.
   */
  #endPreview(): void {
    if (this.#previewFrame !== null) cancelAnimationFrame(this.#previewFrame);
    this.#previewFrame = null;
    this.#previewAt = -Infinity;
    const shown = this.#previewed;
    this.#previewed = null;
    if (!shown) return;
    const state = useSceneSelect.getState();
    const ids = state.assetId === shown.assetId ? selectedIds(state) : [];
    if (sameMembers(ids, shown.ids)) return;
    const doc = paintedDocOf(shown.assetId);
    useInstances
      .getState()
      .highlight(shown.assetId, ids.length > 0 && doc ? [...withDescendants(doc, ids)] : []);
    this.#viewer.scene.requestRender();
  }

  #endStroke(evaluate: boolean): void {
    this.#stroke = null;
    const view = this.#paintView;
    if (!evaluate || !view) {
      this.#endPreview();
      // A match said while the stroke went on is not one any more.
      const store = useSceneSelect.getState();
      if (store.paint?.live) store.setPaint(null);
      return;
    }
    const painted = paintedSplats(view.screen, view.visible, view.mask);
    let count = 0;
    for (let k = 0; k < view.screen.count; k++) if (painted[k]) count++;
    // What was lit as the stroke ended, if it is still about as good: what you see is selected.
    const best = this.#match(view);
    const ids = best?.ids ?? [];
    const first = ids[0];
    const store = useSceneSelect.getState();
    store.setPaint({ ids, best: first ?? null, iou: best?.iou ?? 0, painted: count });
    this.#painted =
      count > 0 ? { assetId: view.assetId, tiles: view.tiles, screen: view.screen, painted } : null;
    if (best && first !== undefined && ids.length > 1) {
      store.selectSet(view.assetId, best, commonChain(view.doc, ids), null);
      this.#focusScene();
    } else if (first !== undefined) {
      const chain = chainOf(view.doc, first);
      store.select(view.assetId, chain, chain.length, 0, null);
      this.#focusScene();
    } else {
      store.clear();
      store.setPaint({ ids: [], best: null, iou: 0, painted: count });
    }
    this.#endPreview();
    log.info("painted", {
      asset: view.assetId,
      ids: ids.slice(0, 8),
      members: ids.length,
      iou: best?.iou,
      splats: count,
      preview: this.#previewCost,
    });
  }

  /** Whether the last painted area matched no object well (`PAINT_MIN_IOU`). */
  get paintedAreaOffered(): boolean {
    const paint = useSceneSelect.getState().paint;
    return this.#painted !== null && paint !== null && !paint.live && paint.iou < PAINT_MIN_IOU;
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
    // Resized only when the view's grid changes: setting a canvas's size reallocates and
    // clears it, and this runs on every move of a stroke.
    if (overlay.width !== cols) overlay.width = cols;
    if (overlay.height !== rows) overlay.height = rows;
    overlay.style.imageRendering = "pixelated";
    const context = overlay.getContext("2d");
    if (!context) return;
    let image = this.#overlayImage;
    if (image?.width !== cols || image.height !== rows) {
      image = context.createImageData(cols, rows);
      this.#overlayImage = image;
    }
    const pixels = image.data;
    pixels.fill(0);
    for (let i = 0; i < data.length; i++) {
      if (!data[i]) continue;
      const at = i * 4;
      pixels[at] = 90;
      pixels[at + 1] = 170;
      pixels[at + 2] = 255;
      pixels[at + 3] = 90;
    }
    context.putImageData(image, 0, 0);
  }
}
