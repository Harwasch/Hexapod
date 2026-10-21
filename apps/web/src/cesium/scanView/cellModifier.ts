/**
 * Confidence and Data age (lib/cellLook.ts) as the dedicated splat renderers draw them: in
 * PlayCanvas's work-buffer modifier and Spark's world modifier, which change a splat's colour
 * once, as the renderer copies (PlayCanvas) or generates (Spark) a tile -- not every frame, and
 * with no eye at hand.
 *
 * So the level of the grid's pyramid is picked per tile, not per splat as CesiumJS does: the
 * level CesiumJS's rule gives the tile's centre (`cellLevelIndex`), handed to the tile as a
 * uniform. As the eye moves the host re-picks every tile's level each frame it draws
 * (`CellTiles.update`), and only a tile whose level steps is copied or generated again -- a
 * pass over its splats, as when an object it holds is hidden. Within a tile every splat reads
 * the same level, so a tile's far side draws finer cells than CesiumJS would and its near side
 * coarser ones; the colours and cues are the same (the coarser cells hold the weakest of their
 * children, so a weak patch is never hidden), only the cell size can step at a tile's edge.
 * `cellLevelAgreement` measures the difference (the developer readouts and the e2e tests).
 *
 * The scan's frame, where both renderers draw a splat, is the cell grid's frame: the tileset's
 * root frame (pose.ts, splatCellView.ts), in the grid's own metres even under a runtime scale.
 *
 * The splats of an inferred layer -- separate tiles here, with their own modifier
 * (layerLook.ts) -- have no capture evidence: drawn grey under either view, never a measured
 * colour, as CesiumJS draws them (`hexapodCellGrey`).
 */

import {
  CELL_LOOK_GLSL,
  cellLevelAgreement,
  cellLevelIndex,
  cellRuleOf,
  cellTileUniforms,
  pxPerRadianOf,
  type CellLookView,
  type CellTileUniforms,
} from "@/lib/cellLook";

import type { ScanCellViewReading } from "./scanCellView";
import type { ScanPose } from "./types";

/**
 * The look as one function of its inputs, for both renderers (PlayCanvas's uniforms, Spark's
 * dyno inputs): a splat at `p` (the scan's frame) of colour `color` on a tile reading `level`
 * and `tile` (`cellTileUniforms`) of `grid`, under `rule` (`cellRuleOf`). Mode 0: as it was.
 */
export const CELL_TILE_GLSL = `${CELL_LOOK_GLSL}
vec4 hexapodCellTile(vec3 p, vec4 color, highp sampler2D grid, vec4 rule, vec4 level, vec4 tile) {
    int mode = int(tile.x + 0.5);
    if (mode == 0) {
        return color;
    }
    ivec2 t = hexapodCellTexel(grid, p.xy, tile.y, level, tile.z);
    return hexapodCellLook(mode, t, p.xy, tile.y, rule, color);
}
`;

/** `(1, 0, 0, 0)` while a view is drawn: an inferred layer's splats then grey. */
export function cellFillOf(view: CellLookView | null, out = new Float32Array(4)): Float32Array {
  out.fill(0);
  out[0] = view ? 1 : 0;
  return out;
}

/** Device pixels a radian spans in `pose`, as CesiumJS measures a cell on screen. */
export function posePxPerRadian(pose: ScanPose): number {
  return pxPerRadianOf(pose.height * (pose.globePixelRatio ?? pose.pixelRatio), pose.fovy);
}

/** One tile's sampled splats, for `CellTiles.reading`: count, and position `i` into `out`. */
export interface CellTileSample {
  count: number;
  at(i: number, out: [number, number, number]): void;
}

/** Splats sampled a tile for the reading: enough to say a share, few enough to ask often. */
const SAMPLES_A_TILE = 64;

interface CellTile {
  centre: readonly [number, number, number];
  /** The level index read now, -1 before the first. */
  level: number;
  readonly uniforms: CellTileUniforms;
}

/**
 * A renderer's tiles under the cell view: which view is drawn, the level each tile reads, and
 * the uniforms it reads it through. The renderer tells it which tiles are the scan's
 * (`add`, `delete`), hands it the view (`setView`) and the eye (`update`), and puts a tile's
 * uniforms on it when told it stepped.
 */
export class CellTiles<M> {
  view: CellLookView | null = null;
  /** The grid's rule (`cellRuleOf`); one array, so a renderer binds it once. */
  readonly rule = new Float32Array(4);
  /** Times a tile's level stepped, so the tile was copied or generated again. */
  steps = 0;
  readonly #tiles = new Map<M, CellTile>();
  #eye: readonly [number, number, number] | null = null;
  #pxPerRadian = 1000;

  get drawing(): boolean {
    return this.view !== null;
  }

  /** The scan's tiles (not a layer's), with where each one's centre is. */
  meshes(): IterableIterator<M> {
    return this.#tiles.keys();
  }

  has(mesh: M): boolean {
    return this.#tiles.has(mesh);
  }

  /** A scan tile, centred at `centre` (the scan's frame); its level picked from the last eye. */
  add(mesh: M, centre: readonly [number, number, number]): CellTileUniforms {
    const tile: CellTile = { centre, level: -1, uniforms: cellTileUniforms(null, 0) };
    this.#tiles.set(mesh, tile);
    this.#pick(tile);
    return tile.uniforms;
  }

  delete(mesh: M): void {
    this.#tiles.delete(mesh);
  }

  /** The uniforms `mesh` reads its level through (undefined for a tile not the scan's). */
  uniforms(mesh: M): CellTileUniforms | undefined {
    return this.#tiles.get(mesh)?.uniforms;
  }

  /**
   * The view to draw now, or null for the photo. Every tile's level is picked again; the caller
   * re-applies every tile (on or off, a new texture). Returns whether anything changed.
   */
  setView(view: CellLookView | null): boolean {
    if (view === this.view) return false;
    this.view = view;
    if (view) this.rule.set(cellRuleOf(view.meta));
    for (const tile of this.#tiles.values()) {
      tile.level = -1;
      this.#pick(tile);
    }
    return true;
  }

  /**
   * The eye moved (the scan's frame) or the view's size did: every tile's level is picked
   * again, and `stepped` is called for each one whose level changed, with its new uniforms.
   */
  update(pose: ScanPose, stepped: (mesh: M, uniforms: CellTileUniforms) => void): void {
    this.#eye = pose.eye;
    this.#pxPerRadian = posePxPerRadian(pose);
    if (!this.view) return;
    for (const [mesh, tile] of this.#tiles) {
      if (this.#pick(tile)) {
        this.steps += 1;
        stepped(mesh, tile.uniforms);
      }
    }
  }

  /** Picks `tile`'s level for the last eye; true when it changed. */
  #pick(tile: CellTile): boolean {
    const view = this.view;
    const eye = this.#eye;
    const k =
      view && eye
        ? cellLevelIndex(
            view.meta,
            Math.hypot(tile.centre[0] - eye[0], tile.centre[1] - eye[1], tile.centre[2] - eye[2]),
            this.#pxPerRadian,
          )
        : 0;
    if (k === tile.level) return false;
    tile.level = k;
    cellTileUniforms(view, k, tile.uniforms);
    return true;
  }

  /**
   * What is drawn, for the readouts: the tiles tinted, the level steps so far, and of up to 64
   * splats sampled a tile (`sample`), how many read the level CesiumJS would give them.
   */
  reading(renderer: string, sample: (mesh: M) => CellTileSample | null): ScanCellViewReading {
    let sampled = 0;
    let agree = 0;
    const view = this.view;
    const eye = this.#eye;
    if (view && eye) {
      const at: [number, number, number] = [0, 0, 0];
      for (const [mesh, tile] of this.#tiles) {
        const splats = sample(mesh);
        if (!splats || splats.count === 0) continue;
        const step = Math.max(1, Math.floor(splats.count / SAMPLES_A_TILE));
        const positions: number[] = [];
        for (let i = 0; i < splats.count; i += step) {
          splats.at(i, at);
          positions.push(at[0], at[1], at[2]);
        }
        const found = cellLevelAgreement(
          view.meta,
          positions,
          [0, 0, 0],
          eye,
          this.#pxPerRadian,
          tile.level,
        );
        sampled += found.splats;
        agree += found.agree;
      }
    }
    return {
      renderer,
      mode: view?.mode ?? null,
      tiles: view ? this.#tiles.size : 0,
      steps: this.steps,
      sampled,
      agree,
    };
  }
}
