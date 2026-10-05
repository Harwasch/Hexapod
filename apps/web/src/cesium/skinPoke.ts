/**
 * Poke and drag scene objects that move by skins (docs/SCENE_OBJECTS.md §9, "The poke
 * driver"), under whichever renderer draws the scan: press on a skinned object and drag -- a
 * spring at the grabbed splat pulls its handles (`@twin/world` `SkinPoke`) -- let go and it
 * rings down at its own frequencies.
 *
 * - **On and off.** Only while the poke tool is on (`state/skinPoke.ts`: `K`, or Settings'
 *   switch under the wind) and selection would take the pointer (`enabled`: not while
 *   measuring, exploring or painting with the brush).
 * - **The camera stays yours.** A press is taken only when it lands on a skinned splat: then
 *   the camera's inputs are held (`enableInputs`, as the brush holds them) until the release,
 *   and the press never reaches CesiumJS's handlers, so it neither turns the camera nor
 *   clicks. A press on the sky, the ground or an object without a skin goes on to the camera.
 * - **Picking** is the scene selection's (`lib/splatPick.ts` on the renderer's pick source):
 *   the front-most skinned splat under the cursor, its skin and weight row read from the
 *   skin's own tile binding (`tileSkin`, by the tile's checksum), so it works under
 *   PlayCanvas, Spark and CesiumJS alike, with or without `instances.json`.
 * - **Dragging** moves the cursor's point on the plane through the grabbed splat facing the
 *   camera; its offset from the splat's rest position is the pull.
 * - **With the wind.** The poke's handles are laid over the skin's (`setInstanceOverlay`), so
 *   the wind goes on swaying what is held, and calm leaves a ringing object ringing.
 * - **Material**: the wind's (`materialPrior` from the instance's or the skin's own traits,
 *   `materials.json` over it). A movable object moves whole (its constant handle) and springs
 *   home; a rooted one bends in its anchored modes. What the wind sways is rooted, whatever its
 *   behaviour says (a plant). A rigid rooted object (one handle) does not move.
 *
 * Time is the page's clock (`performance.now`), not the scene's: a poke is the viewer's own
 * hand. The physics runs on a fixed grid, so a frame's look does not depend on the frame rate.
 */

import {
  materialPrior,
  mergeMaterial,
  SkinPoke,
  skinPokeModel,
  type SkinPokeModel,
} from "@twin/world";
import { Cartesian2, Cartesian3, Matrix4, type Camera, type Scene } from "cesium";

import { createLogger } from "@/lib/log";
import {
  HANDLE_FLOATS,
  rowWeight,
  tileSkin,
  type SkinDoc,
  type SkinEntry,
  type TileSkin,
} from "@/lib/skin";
import { castRay, type PickTile } from "@/lib/splatPick";
import { useInstances } from "@/state/instances";
import { useSceneSelect } from "@/state/sceneSelect";
import { useSkinPoke } from "@/state/skinPoke";

import { uniformScale } from "./placement";
import { pickSourceOf } from "./sceneSelect/pickSources";
import { attachedSkins, type SplatSkinning } from "./splatSkin";
import { inverseScaledTransformation } from "./tilesetScale";

const log = createLogger("skin-poke");

/** What the controller needs of a CesiumJS viewer or widget. */
export interface PokeViewer {
  readonly scene: Scene;
  readonly camera: Camera;
  readonly canvas: HTMLCanvasElement;
}

export interface SkinPokeOptions {
  /** Whether presses may poke now, beyond the tool being on (measuring, exploring...). */
  enabled?: () => boolean;
  /** Seconds, only moving forward (default: the page's clock). */
  clock?: () => number;
}

/** One object being poked (or still ringing). */
interface Poked {
  readonly assetId: string;
  readonly part: SplatSkinning;
  readonly skin: SkinEntry;
  readonly poke: SkinPoke;
  readonly buffer: Float64Array;
}

/** What a press took hold of. */
interface Hold {
  readonly poked: Poked;
  /** The grabbed splat's rest position, and the plane's normal (the ray), scan frame. */
  readonly rest: [number, number, number];
  readonly normal: [number, number, number];
  readonly toLocal: Matrix4;
  readonly pointerId: number;
}

/** A splat a ray met that has a skin: where, which skin, its weights (`w_0 = 1` first). */
export interface SkinHit {
  readonly assetId: string;
  readonly part: SplatSkinning;
  readonly skin: SkinEntry;
  readonly weights: Float64Array;
  readonly point: [number, number, number];
  readonly direction: [number, number, number];
  readonly toLocal: Matrix4;
  /** Distance along the ray on the globe (metres), to compare scans. */
  readonly t: number;
}

/** A splat's weights `[1, w_1, ..., w_{m−1}]` from its tile's skin binding. */
export function splatWeights(
  tile: TileSkin,
  index: number,
  skin: SkinEntry,
  doc: SkinDoc,
): Float64Array {
  const out = new Float64Array(skin.handles);
  out[0] = 1;
  for (let k = 0; k < skin.handles - 1; k += 1)
    out[k + 1] = rowWeight(tile.words, index, k, doc.scale, tile.words2);
  return out;
}

export class SkinPokeController {
  readonly #viewer: PokeViewer;
  readonly #enabled: () => boolean;
  readonly #clock: () => number;
  readonly #off: (() => void)[] = [];
  /** Per skin part and skin id, what is being poked. */
  readonly #poked = new Map<SplatSkinning, Map<number, Poked>>();
  /** Decoded tile bindings, per skin document and tile checksum. */
  readonly #tiles = new WeakMap<SkinDoc, Map<string, TileSkin | null>>();
  #hold: Hold | null = null;
  #cameraInputs: boolean | null = null;

  constructor(viewer: PokeViewer, options: SkinPokeOptions = {}) {
    this.#viewer = viewer;
    this.#enabled = options.enabled ?? (() => true);
    this.#clock = options.clock ?? (() => performance.now() / 1000);
    const canvas = viewer.canvas;
    const down = (e: PointerEvent): void => this.#onDown(e);
    const move = (e: PointerEvent): void => this.#onMove(e);
    const up = (e: PointerEvent): void => this.#onUp(e);
    // Capture on the canvas: ahead of CesiumJS's own handler on it, so a taken press is never
    // seen by the camera or the map's click.
    canvas.addEventListener("pointerdown", down, { capture: true });
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
    this.#off.push(
      () => canvas.removeEventListener("pointerdown", down, { capture: true }),
      () => window.removeEventListener("pointermove", move),
      () => window.removeEventListener("pointerup", up),
      () => window.removeEventListener("pointercancel", up),
      viewer.scene.preUpdate.addEventListener(() => this.tick()),
      // The tool turned off lets go of what is held.
      useSkinPoke.subscribe((state, previous) => {
        if (previous.active && !state.active) this.#letGo();
      }),
    );
  }

  /** Whether something is held now. */
  get holding(): boolean {
    return this.#hold !== null;
  }

  /** Objects held or still ringing. */
  get active(): number {
    let n = 0;
    for (const skins of this.#poked.values())
      for (const p of skins.values()) if (p.poke.active) n += 1;
    return n;
  }

  destroy(): void {
    this.#letGo();
    for (const off of this.#off.splice(0)) off();
    for (const [part, skins] of this.#poked)
      for (const p of skins.values()) part.setOverlay(p.skin.id, null);
    this.#poked.clear();
  }

  /**
   * The front-most skinned splat under (`x`, `y`) (CSS px from the canvas's top left) over
   * every scan with a skin attached, or null.
   */
  hitAt(x: number, y: number): SkinHit | null {
    const ray = this.#viewer.camera.getPickRay(new Cartesian2(x, y));
    if (!ray) return null;
    const frustum = this.#viewer.camera.frustum as { fovy?: number };
    const pixelAngle =
      (frustum.fovy ?? Math.PI / 3) / Math.max(1, this.#viewer.canvas.clientHeight);
    let best: SkinHit | null = null;
    for (const [assetId, part] of attachedSkins()) {
      const source = pickSourceOf(assetId);
      const toWorld = source?.toWorld();
      if (!source || !toWorld) continue;
      const tiles = source.tiles();
      if (tiles.length === 0) continue;
      const toLocal = inverseScaledTransformation(toWorld, new Matrix4());
      const scale = uniformScale(toWorld);
      const origin = Matrix4.multiplyByPoint(toLocal, ray.origin, new Cartesian3());
      const direction = Matrix4.multiplyByPointAsVector(toLocal, ray.direction, new Cartesian3());
      const length = Math.hypot(direction.x, direction.y, direction.z) || 1;
      const unit: [number, number, number] = [
        direction.x / length,
        direction.y / length,
        direction.z / length,
      ];
      const hits = castRay(
        tiles,
        { origin: [origin.x, origin.y, origin.z], direction: unit },
        { pixelAngle },
      );
      // The splat that is most of the pixel among the front ones, if it has a skin.
      const front = hits.slice(0, 64).sort((a, b) => b.weight - a.weight);
      for (const hit of front) {
        const tile = tiles[hit.tile];
        if (!tile) continue;
        const bound = this.#tileOf(part.doc, tile);
        const id = bound?.skins[hit.index] ?? 0;
        const skin = id ? part.doc.byId.get(id) : undefined;
        if (!bound || !skin) continue;
        const point: [number, number, number] = [
          tile.positions[hit.index * 3] ?? 0,
          tile.positions[hit.index * 3 + 1] ?? 0,
          tile.positions[hit.index * 3 + 2] ?? 0,
        ];
        const t = hit.t * scale;
        if (!best || t < best.t) {
          best = {
            assetId,
            part,
            skin,
            weights: splatWeights(bound, hit.index, skin, part.doc),
            point,
            direction: unit,
            toLocal,
            t,
          };
        }
        break;
      }
    }
    return best;
  }

  /**
   * Takes hold at (`x`, `y`) when a skinned object that can move is there: true when taken.
   * (The pointer handler's work; public for harnesses.)
   */
  grabAt(x: number, y: number, pointerId = -1): boolean {
    const hit = this.hitAt(x, y);
    if (!hit) return false;
    const poked = this.#pokedFor(hit);
    if (!poked) return false;
    poked.poke.advance(this.#clock());
    poked.poke.grab({ weights: hit.weights });
    this.#hold = {
      poked,
      rest: hit.point,
      normal: hit.direction,
      toLocal: hit.toLocal,
      pointerId,
    };
    const model = poked.poke.model;
    const slowest = model.modes > 0 ? Math.min(...model.omega) / (2 * Math.PI) : null;
    useSkinPoke.getState().setGrabbed({
      assetId: hit.assetId,
      instance: hit.skin.instance,
      label: this.#labelOf(hit.assetId, hit.skin),
      hz: slowest,
      movable: model.movable,
    });
    this.#holdCamera(true);
    log.info("poke", {
      asset: hit.assetId,
      instance: hit.skin.instance,
      handles: hit.skin.handles,
      modes: model.modes,
      movable: model.movable,
    });
    this.#viewer.scene.requestRender();
    return true;
  }

  /** Pulls what is held toward (`x`, `y`). */
  dragTo(x: number, y: number): void {
    const hold = this.#hold;
    if (!hold) return;
    const ray = this.#viewer.camera.getPickRay(new Cartesian2(x, y));
    if (!ray) return;
    const o = Matrix4.multiplyByPoint(hold.toLocal, ray.origin, new Cartesian3());
    const d = Matrix4.multiplyByPointAsVector(hold.toLocal, ray.direction, new Cartesian3());
    const [nx, ny, nz] = hold.normal;
    const denom = d.x * nx + d.y * ny + d.z * nz;
    if (Math.abs(denom) < 1e-9) return;
    const [px, py, pz] = hold.rest;
    const t = ((px - o.x) * nx + (py - o.y) * ny + (pz - o.z) * nz) / denom;
    if (!(t > 0)) return;
    hold.poked.poke.pull([o.x + d.x * t - px, o.y + d.y * t - py, o.z + d.z * t - pz]);
    this.#viewer.scene.requestRender();
  }

  /** Lets go of what is held: it rings down. */
  release(): void {
    this.#letGo();
  }

  /** Advances every poked object and hands its skin part the overlay. Once a frame. */
  tick(): void {
    if (this.#poked.size === 0) return;
    const now = this.#clock();
    let moving = false;
    for (const [part, skins] of this.#poked) {
      const first = skins.values().next().value;
      if (!first || attachedSkins().get(first.assetId) !== part) {
        // The part went (another skin chosen, the scan unloaded): drop its pokes.
        if (this.#hold && this.#hold.poked.part === part) this.#letGo();
        this.#poked.delete(part);
        continue;
      }
      for (const [id, p] of skins) {
        p.poke.advance(now);
        const handles = p.poke.handles(p.buffer);
        part.setOverlay(id, handles);
        if (handles) moving = true;
        else if (this.#hold?.poked !== p) skins.delete(id);
      }
      if (skins.size === 0) this.#poked.delete(part);
    }
    if (moving) this.#viewer.scene.requestRender();
    else if (this.#hold === null) useSkinPoke.getState().setGrabbed(null);
  }

  #pokedFor(hit: SkinHit): Poked | null {
    let skins = this.#poked.get(hit.part);
    const found = skins?.get(hit.skin.id);
    if (found) return found;
    const traits = this.#traitsOf(hit.assetId, hit.skin);
    const material = mergeMaterial(
      materialPrior(traits.properties, traits.behaviour),
      hit.part.materials.get(hit.skin.instance),
    );
    // Movable by its behaviour, unless its material says the wind sways it: a plant is rooted.
    const movable = traits.behaviour === "movable" && !material.wind;
    const dynamics = hit.skin.dynamics;
    const model: SkinPokeModel | undefined =
      hit.skin.handles === 1
        ? skinPokeModel(
            {
              handles: 1,
              origin: hit.skin.origin,
              scale: hit.skin.scale,
              eigenvalues: [],
              support: [],
              mass: [1],
              anchorGram: [1],
            },
            material,
            movable,
          )
        : dynamics
          ? skinPokeModel(
              {
                handles: hit.skin.handles,
                origin: hit.skin.origin,
                scale: hit.skin.scale,
                eigenvalues: hit.skin.eigenvalues,
                support: hit.skin.support,
                mass: dynamics.mass,
                anchorGram: dynamics.anchorGram,
              },
              material,
              movable,
            )
          : undefined;
    if (!model) return null;
    const poked: Poked = {
      assetId: hit.assetId,
      part: hit.part,
      skin: hit.skin,
      poke: new SkinPoke(model),
      buffer: new Float64Array(hit.skin.handles * HANDLE_FLOATS),
    };
    if (!skins) {
      skins = new Map();
      this.#poked.set(hit.part, skins);
    }
    skins.set(hit.skin.id, poked);
    return poked;
  }

  #traitsOf(
    assetId: string,
    skin: SkinEntry,
  ): { properties: Record<string, number> | undefined; behaviour: string | undefined } {
    const instance = useInstances
      .getState()
      .assets[assetId]?.instances.find((i) => i.id === skin.instance);
    if (instance) return { properties: instance.properties, behaviour: instance.behaviour };
    return { properties: skin.traits?.properties, behaviour: skin.traits?.behaviour ?? "in-place" };
  }

  #labelOf(assetId: string, skin: SkinEntry): string {
    const instance = useInstances
      .getState()
      .assets[assetId]?.instances.find((i) => i.id === skin.instance);
    const names = [
      instance?.tags[0]?.label,
      instance?.category,
      skin.traits?.label,
      skin.traits?.category,
    ];
    return (
      names.find((name): name is string => typeof name === "string" && name !== "") ?? "object"
    );
  }

  #tileOf(doc: SkinDoc, tile: PickTile): TileSkin | null {
    let tiles = this.#tiles.get(doc);
    if (!tiles) {
      tiles = new Map();
      this.#tiles.set(doc, tiles);
    }
    let bound = tiles.get(tile.checksum);
    if (bound === undefined) {
      const decoded = tileSkin(doc, tile.checksum);
      bound = decoded?.skins.length === tile.count ? decoded : null;
      tiles.set(tile.checksum, bound);
    }
    return bound;
  }

  #local(e: PointerEvent): { x: number; y: number } {
    const rect = this.#viewer.canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  #onDown(e: PointerEvent): void {
    if (e.button !== 0 || this.#hold !== null) return;
    if (!useSkinPoke.getState().active || !this.#enabled()) return;
    // The brush owns every press while it paints.
    if (useSceneSelect.getState().mode === "paint") return;
    const { x, y } = this.#local(e);
    if (!this.grabAt(x, y, e.pointerId)) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    try {
      this.#viewer.canvas.setPointerCapture(e.pointerId);
    } catch {
      // A synthetic or already-released pointer: the window listeners still see it.
    }
  }

  #onMove(e: PointerEvent): void {
    const hold = this.#hold;
    if (!hold || (hold.pointerId !== -1 && e.pointerId !== hold.pointerId)) return;
    const { x, y } = this.#local(e);
    this.dragTo(x, y);
  }

  #onUp(e: PointerEvent): void {
    const hold = this.#hold;
    if (!hold || (hold.pointerId !== -1 && e.pointerId !== hold.pointerId)) return;
    this.#letGo();
  }

  #letGo(): void {
    const hold = this.#hold;
    if (!hold) return;
    hold.poked.poke.advance(this.#clock());
    hold.poked.poke.release();
    this.#hold = null;
    this.#holdCamera(false);
    this.#viewer.scene.requestRender();
  }

  /** The camera's inputs held while something is held, as they were before afterwards. */
  #holdCamera(hold: boolean): void {
    const controller = this.#viewer.scene.screenSpaceCameraController;
    if (hold) {
      this.#cameraInputs ??= controller.enableInputs;
      controller.enableInputs = false;
    } else if (this.#cameraInputs !== null) {
      controller.enableInputs = this.#cameraInputs;
      this.#cameraInputs = null;
    }
  }
}
