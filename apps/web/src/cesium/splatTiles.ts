/**
 * Multi-tile splat snapshots: which tile each aggregated splat came from, and what each tile's
 * gaussians are bound to. Pure: no Cesium, no GPU.
 *
 * A level-of-detail splat tileset is drawn by **one** `GaussianSplatPrimitive` that
 * concatenates every selected tile's baked positions into one array, one texture and one sort
 * (`GaussianSplatPrimitive.js` `update`). Tiles come and go with the camera, and under REPLACE
 * refinement a merged parent gives way to its children and back. So a splat's index in the
 * aggregate means nothing on its own — the same gaussian sits at a different index every time
 * the selection changes, and a parent's merged gaussians are not in any leaf at all.
 *
 * What does not change is a gaussian's **canonical position**. So everything that decides how
 * a gaussian moves is computed per tile, from that tile's own positions, once per tile load:
 *
 * - **identity** — the tile's un-baked positions digest into `rig.tileChecksums` (or, for a
 *   single-tile rig, `canonicalChecksum`), bit-exactly, or the deformer refuses;
 * - **binding** — each gaussian's four nearest rig nodes and their blend weights
 *   (`skinSplatsToNodes`), leaves and merged parents alike, so a parent rides the limb its own
 *   centre sits on and neighbouring gaussians bound to different joints move continuously;
 * - **flutter identity** — a key from the snapped position (`positionKeys`), so the same
 *   gaussian shimmers the same whichever tile carries it.
 *
 * and a snapshot's arrays are those per-tile results concatenated in the snapshot's own tile
 * order. A tile that stays loaded across selections is bound once (`TileBindingCache`).
 *
 * **A forest rig** (many plants, `rig.plants`) adds one thing a position cannot say: which
 * plant a gaussian belongs to, or none. That comes from the rig's plant binding
 * (`plants.json`, `plantBinding.ts` in `@twin/world`), keyed by the same per-tile checksum
 * that proves the tile's identity: a tile whose digest has no entry is refused, a gaussian
 * labelled static is bound to the rig's pinned anchor and never moves, and a plant's gaussian
 * is skinned to that plant's joints only (`skinSplatsToPlants`).
 *
 * Why at load time and not packaged per tile: see `docs/LIVING_SURVEY.md`, "Multi-tile
 * tilesets" — binding measured 2–3 µs a gaussian here, which a sidecar would trade for bytes
 * on the wire, a second format to keep in step with the rig, and a binding that could no
 * longer follow a re-extracted rig without re-packaging every tile.
 */

import {
  checksumPositions,
  concatSkins,
  plantLabels,
  positionKeys,
  skinPrimary,
  skinSplatsToNodes,
  skinSplatsToPlants,
  type MotionRig,
  type PlantBinding,
  type SplatSkin,
} from "@twin/world";

import {
  invertAffine,
  maxAbsDifference,
  transformPositions,
  unbakePositions,
  type Mat4,
} from "./splatFrames";
import type {
  SplatPrimitive,
  SplatTile,
  SplatTileContent,
  SplatTilesetLike,
} from "./splatInternals";

/** One tile's place in a snapshot: its splats are `positions[start*3 .. (start+count)*3)`. */
export interface SnapshotTile {
  /** Identity key for the binding cache. Replaced by the engine when the tile reloads. */
  readonly content: SplatTileContent | SplatTile;
  readonly start: number;
  readonly count: number;
  /** The tile's own bake matrix `B`. */
  readonly bake: Mat4;
}

/** Why a snapshot's tiles cannot be read yet. Always a wait: the next snapshot may resolve it. */
export type SnapshotTilesWait =
  /** A rebuild is in flight, so `_selectedTileSet` names its tiles, not the committed ones. */
  | "pending"
  /** The tile list does not account for the snapshot: counts or sampled positions disagree. */
  | "tiles"
  /** A selected tile has not been baked yet. */
  | "no-bake-transform";

export type SnapshotTilesResult =
  | { readonly kind: "tiles"; readonly tiles: readonly SnapshotTile[] }
  | { readonly kind: "wait"; readonly reason: SnapshotTilesWait; readonly detail: string };

/** Positions compared per tile when checking the tile list against the snapshot. */
const SAMPLED_SPLATS = 16;

/** `Object.is` over raw bits, so `-0`/`+0` and NaN payloads count as different. */
function sameBits(a: Float32Array, aIndex: number, b: Float32Array, bIndex: number): boolean {
  return Object.is(a[aIndex], b[bIndex]);
}

function bakeOf(content: SplatTileContent | undefined): Mat4 | undefined {
  const transform = content?._lastSplatTransform;
  return transform?.length === 16 ? transform : undefined;
}

/**
 * The tiles the committed snapshot aggregated, in aggregation order, each with its splat range.
 *
 * On the engine this reads `_selectedTileSet`. A primitive without one — a test double, or an
 * engine that stops exposing it — falls back to the root when the root is the only tile, which
 * is what the single-tile deformer always assumed.
 *
 * Every answer is checked against the snapshot rather than trusted: the counts must sum to
 * `numSplats` exactly, and a sample of each tile's own baked positions must sit, bit for bit,
 * at its range in `_positions`. A disagreement is a wait, not a refusal — it means the
 * bookkeeping is between two snapshots, not that the splats are the wrong ones.
 */
export function snapshotTiles(
  tileset: SplatTilesetLike,
  primitive: SplatPrimitive,
  positions: Float32Array,
  numSplats: number,
): SnapshotTilesResult {
  const selected = primitive._selectedTileSet;
  let list: SplatTile[];
  if (selected !== undefined) {
    if (primitive._pendingSnapshot !== undefined && primitive._pendingSnapshot !== null) {
      return { kind: "wait", reason: "pending", detail: "a snapshot rebuild is in flight" };
    }
    list = [...selected];
  } else {
    const root = tileset.root;
    if (root === undefined || (root.children?.length ?? 0) > 0) {
      return { kind: "wait", reason: "tiles", detail: "no selected-tile list on this engine" };
    }
    list = [root];
  }
  if (list.length === 0) return { kind: "wait", reason: "tiles", detail: "no tiles selected" };

  const tiles: SnapshotTile[] = [];
  let start = 0;
  for (const tile of list) {
    const content = tile.content;
    const bake = bakeOf(content);
    if (bake === undefined) {
      return { kind: "wait", reason: "no-bake-transform", detail: "a tile is not baked yet" };
    }
    const own = content?.positions;
    const count =
      content?.pointsLength ??
      (own === undefined ? (list.length === 1 ? numSplats : undefined) : own.length / 3);
    if (count === undefined || !Number.isInteger(count) || count < 0) {
      return { kind: "wait", reason: "tiles", detail: "a tile does not report its splat count" };
    }
    if (own !== undefined) {
      if (own.length !== count * 3) {
        return { kind: "wait", reason: "tiles", detail: "a tile's positions and count disagree" };
      }
      const step = Math.max(1, Math.floor(count / SAMPLED_SPLATS));
      for (let i = 0; i < count; i += step) {
        for (let k = 0; k < 3; k += 1) {
          if (!sameBits(own, i * 3 + k, positions, (start + i) * 3 + k)) {
            return {
              kind: "wait",
              reason: "tiles",
              detail: "a tile's positions are not where the snapshot has them",
            };
          }
        }
      }
    }
    tiles.push({ content: content ?? tile, start, count, bake });
    start += count;
  }
  if (start !== numSplats) {
    return {
      kind: "wait",
      reason: "tiles",
      detail: `tiles hold ${String(start)} splats, the snapshot ${String(numSplats)}`,
    };
  }
  return { kind: "tiles", tiles };
}

/** Everything derived from one tile under one bake matrix. Immutable once built. */
export interface TileBinding {
  /** The bake matrix this was derived under, copied. */
  readonly bake: readonly number[];
  /** `checksumPositions` over the un-baked positions: the tile's identity. */
  readonly checksum: string;
  /** Immutable. The rig's frame, recovered from the baked positions. */
  readonly canonicalLocal: Float32Array;
  /** Nearest rig node per gaussian, from its own position: slot 0 of `skin`. */
  readonly assignment: Uint16Array;
  /** Four nodes and blend weights per gaussian, from its own position. */
  readonly skin: SplatSkin;
  /** Flutter identity per gaussian, from its own position. */
  readonly flutterKeys: Uint32Array;
  /** Largest disagreement between our re-bake and the engine's baked positions, metres. */
  readonly bakeResidualM: number;
  /**
   * For a forest rig: each gaussian's plant label from the binding — 0 static, `k` plant
   * `k - 1` of `rig.plants`. Absent for a single-plant rig.
   */
  readonly plantLabels?: Uint16Array;
  /** Milliseconds this tile took to prove and bind: the per-tile binding cost. */
  readonly bindMs: number;
}

/** Why a tile cannot be bound. Both are permanent: the splats or the placement are wrong. */
export type TileRefusal = "checksum" | "bake";

export type BindResult =
  | { readonly kind: "bound"; readonly binding: TileBinding }
  | {
      readonly kind: "refused";
      readonly reason: TileRefusal;
      readonly detail: string;
      readonly checksum?: string;
    };

/**
 * Proves one tile's identity and binds its gaussians.
 *
 * `baked` is the tile's range of `_positions` — read, never written. The un-bake snaps onto
 * SPZ's grid, which recovers the tiler's own float32s exactly (`splatFrames.ts`), so the
 * digest is bit-exact against the one `rig_tiles.py` took from the SPZ block.
 */
export function bindTile(
  baked: Float32Array,
  bake: Mat4,
  rig: MotionRig,
  accepted: ReadonlySet<string>,
  residualLimitM: number,
  plants?: PlantBinding,
): BindResult {
  const started = performance.now();
  const inverse = invertAffine(bake);
  if (inverse === undefined) {
    return { kind: "refused", reason: "bake", detail: "the tile's bake transform is singular" };
  }
  const canonicalLocal = unbakePositions(baked, inverse);
  const checksum = checksumPositions(canonicalLocal);
  if (!accepted.has(checksum)) {
    return {
      kind: "refused",
      reason: "checksum",
      detail: `un-baked tile digests ${checksum}, which the rig does not list`,
      checksum,
    };
  }
  // Proves in one number that the bake matrix is right, that the un-bake recovered the true
  // source, and that our arithmetic matches the engine's. Exactly 0 on the rigid fast path
  // `transformTile` takes for a placed capture.
  const rebaked = transformPositions(canonicalLocal, bake, new Float32Array(baked.length));
  const bakeResidualM = maxAbsDifference(rebaked, baked);
  if (!(bakeResidualM <= residualLimitM)) {
    return {
      kind: "refused",
      reason: "bake",
      detail: `re-baking disagrees with the engine by ${String(bakeResidualM)} m`,
      checksum,
    };
  }
  let labels: Uint16Array | undefined;
  if (rig.plants !== undefined) {
    labels = plants === undefined ? undefined : plantLabels(plants, checksum);
    if (labels?.length !== canonicalLocal.length / 3) {
      return {
        kind: "refused",
        reason: "checksum",
        detail: `the plant binding has no entry for tile ${checksum}`,
        checksum,
      };
    }
  }
  const skin =
    labels === undefined
      ? skinSplatsToNodes(canonicalLocal, rig)
      : skinSplatsToPlants(canonicalLocal, rig, labels);
  const flutterKeys = positionKeys(canonicalLocal);
  return {
    kind: "bound",
    binding: {
      bake: Array.from(bake),
      checksum,
      canonicalLocal,
      assignment: skinPrimary(skin),
      skin,
      flutterKeys,
      bakeResidualM,
      ...(labels === undefined ? {} : { plantLabels: labels }),
      bindMs: performance.now() - started,
    },
  };
}

/** Whether two matrices are the same sixteen numbers. Exact: a re-bake is a new derivation. */
export function sameMatrix(a: Mat4, b: Mat4): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) if (a[i] !== b[i]) return false;
  return true;
}

/**
 * Tile bindings kept across snapshots, keyed by the engine's content object.
 *
 * A tile that stays selected while its neighbours come and go is bound once. Keyed weakly, so
 * a tile the engine unloads takes its binding with it; and checked against the bake matrix, so
 * a re-bake (clamp-to-ground moving the model) is re-proven rather than assumed.
 */
export class TileBindingCache {
  #entries = new WeakMap<object, TileBinding>();
  /** Tiles bound since construction, for diagnostics. */
  bound = 0;

  get(tile: SnapshotTile): TileBinding | undefined {
    const entry = this.#entries.get(tile.content);
    return entry !== undefined && sameMatrix(entry.bake, tile.bake) ? entry : undefined;
  }

  set(tile: SnapshotTile, binding: TileBinding): void {
    this.#entries.set(tile.content, binding);
    this.bound += 1;
  }

  clear(): void {
    this.#entries = new WeakMap();
  }
}

/** One snapshot's per-splat arrays: the tiles' bindings concatenated in snapshot order. */
export interface SnapshotBinding {
  readonly numSplats: number;
  readonly tiles: readonly (SnapshotTile & { readonly binding: TileBinding })[];
  /** Immutable. Rig-frame positions of every splat. */
  readonly canonicalLocal: Float32Array;
  readonly assignment: Uint16Array;
  readonly skin: SplatSkin;
  readonly flutterKeys: Uint32Array;
  /** A forest rig's plant label per splat (0 static); absent for a single-plant rig. */
  readonly plantLabels: Uint16Array | undefined;
  /** The bake matrix every tile shares, or `undefined` when they differ. */
  readonly commonBake: Mat4 | undefined;
  /** The worst tile's residual. */
  readonly bakeResidualM: number;
}

/**
 * Concatenates tile bindings in snapshot order. A single tile's arrays are used as they are —
 * no copy — which keeps the single-tile path exactly what it was.
 */
export function aggregateBindings(
  tiles: readonly SnapshotTile[],
  bindings: readonly TileBinding[],
  numSplats: number,
): SnapshotBinding {
  const joined = tiles.map((tile, index) => {
    const binding = bindings[index];
    if (binding === undefined) throw new Error("aggregateBindings: a tile has no binding");
    return { ...tile, binding };
  });
  let bakeResidualM = 0;
  let commonBake: Mat4 | undefined = joined[0]?.bake;
  for (const tile of joined) {
    if (tile.binding.bakeResidualM > bakeResidualM) bakeResidualM = tile.binding.bakeResidualM;
    if (commonBake !== undefined && !sameMatrix(commonBake, tile.bake)) commonBake = undefined;
  }
  const only = joined.length === 1 ? joined[0] : undefined;
  if (only !== undefined) {
    return {
      numSplats,
      tiles: joined,
      canonicalLocal: only.binding.canonicalLocal,
      assignment: only.binding.assignment,
      skin: only.binding.skin,
      flutterKeys: only.binding.flutterKeys,
      plantLabels: only.binding.plantLabels,
      commonBake,
      bakeResidualM,
    };
  }
  const canonicalLocal = new Float32Array(numSplats * 3);
  const assignment = new Uint16Array(numSplats);
  const flutterKeys = new Uint32Array(numSplats);
  const labelled = joined.some((tile) => tile.binding.plantLabels !== undefined);
  const plantLabelsAll = labelled ? new Uint16Array(numSplats) : undefined;
  for (const tile of joined) {
    canonicalLocal.set(tile.binding.canonicalLocal, tile.start * 3);
    assignment.set(tile.binding.assignment, tile.start);
    flutterKeys.set(tile.binding.flutterKeys, tile.start);
    if (plantLabelsAll !== undefined && tile.binding.plantLabels !== undefined)
      plantLabelsAll.set(tile.binding.plantLabels, tile.start);
  }
  const skin = concatSkins(
    joined.map((tile) => tile.binding.skin),
    joined.map((tile) => tile.start),
    numSplats,
  );
  return {
    numSplats,
    tiles: joined,
    canonicalLocal,
    assignment,
    skin,
    flutterKeys,
    plantLabels: plantLabelsAll,
    commonBake,
    bakeResidualM,
  };
}
