/**
 * Which tiles of a scan's level-of-detail tileset the scan viewer downloads, and in what order.
 *
 * `tools/captures/splat_tiles.py` packs every gaussian of a scan into an octree of tiles with
 * REPLACE refinement: the leaves hold every gaussian once, as trained, and each parent holds
 * its subtree *merged* (Hierarchical 3DGS moment matching, one gaussian per occupied cell), so
 * any cut through the tree -- a set of tiles meeting every root-to-leaf path exactly once -- is
 * the whole scan at some resolution, and the cut of all leaves is the scan itself.
 *
 * Deciding what to *draw* is no longer this module's job. Spark 2.2 has level of detail built
 * in (`SparkRenderer.lodSplatCount`, `SplatMesh({ lod: true })`): it builds a merged LoD tree
 * over each loaded mesh in a worker and, every frame, draws at most its splat budget, chosen
 * by screen size across every mesh at once (SparkRenderer.ts `driveLod`, one
 * `traverseLodTrees` call over all instances). The phone's Detail choice is that budget
 * (main.ts). What is left is what to *download*: this module reads the tree (each tile's
 * gaussian count is in its `extras.gaussians` and its bounds in its `boundingVolume`, so
 * nothing is fetched to decide), and stream.ts picks the cut from where the camera is.
 *
 * Tilesets packed before merged parents are ADD (a parent holds a thinned subset, drawn under
 * its children): there, tiles are only ever added, in the static order `planAdditive` gives.
 * A single-tile tileset -- anything packed before the hierarchy, and the committed tree -- is
 * just its root either way.
 */

import { collisionMetaOf, type CollisionMeta } from "@/lib/collision";

export type Refine = "ADD" | "REPLACE";

/** A tile's bounding sphere in the tileset's own frame (the root transform's, east/north/up). */
export interface Sphere {
  center: [number, number, number];
  radius: number;
}

export interface TileNode {
  uri: string;
  gaussians: number;
  geometricError: number;
  children: TileNode[];
  /** Absent when the tileset gives none this page reads (a `region`): such a tile is always
   *  treated as in view and at distance zero, so it is refined first rather than never. */
  bounds?: Sphere | null;
  /** Its `box` bounding volume (centre, then three half-axis vectors), when it has one: how
   *  far the camera is from the tile, where the sphere around a box is far too loose. */
  box?: number[] | null;
}

export interface TileTree {
  refine: Refine;
  root: TileNode;
  /** The packaged collision grid the root declares (hexapod.collision), if any. */
  collision?: CollisionMeta | null;
}

/** One step of loading an ADD tileset: fetch `add` and show it (`remove` stays empty). */
export interface LoadStep {
  add: TileNode[];
  remove: TileNode[];
}

/**
 * How many times the Detail budget the viewer may download. Spark draws at most the budget
 * (its `lodSplatCount`) from whatever is loaded, so loading more than it draws is what lets a
 * close look sharpen instead of stopping at the overview's detail. 4x is the ratio Spark
 * itself ships on iOS (a 6.3M-splat page pool against a 1.5M draw budget,
 * SparkRenderer.ts `maxPagedSplats` / `defaultSplatTarget`); Standard (400k) then downloads up
 * to 1.6M gaussians, about 23 MB of SPZ, and holds ~36 MB of Spark's 16-byte splats with
 * their LoD tree.
 */
export const LOAD_FACTOR = 4;

interface RawTile {
  boundingVolume?: { box?: unknown; sphere?: unknown };
  content?: { uri?: unknown };
  geometricError?: unknown;
  extras?: { gaussians?: unknown };
  children?: unknown;
  refine?: unknown;
}

/**
 * The tile tree of a `tileset.json`. A tile without a gaussian count (a tileset packed before
 * the hierarchy, or from elsewhere) counts as unaffordable -- except the root, which is always
 * loaded: without it there is nothing to show. Refinement is read from the root, where the
 * packer states it; a root that does not say is REPLACE, 3D Tiles' own default for content
 * that stands for its children.
 */
export function parseTileset(document: unknown): TileTree {
  const root = (document as { root?: RawTile } | null)?.root;
  if (!root) throw new Error("The scan's tileset has no root tile.");
  const node = (raw: RawTile): TileNode => {
    const uri = raw.content?.uri;
    const gaussians = raw.extras?.gaussians;
    const error = raw.geometricError;
    return {
      uri: typeof uri === "string" ? uri : "",
      gaussians: typeof gaussians === "number" ? gaussians : Number.POSITIVE_INFINITY,
      geometricError: typeof error === "number" ? error : 0,
      children: Array.isArray(raw.children) ? (raw.children as RawTile[]).map(node) : [],
      bounds: sphereOf(raw.boundingVolume),
      box: numbers(raw.boundingVolume?.box, 12),
    };
  };
  const parsed = node(root);
  if (!parsed.uri) throw new Error("The scan's tileset names no content.");
  return {
    refine: root.refine === "ADD" ? "ADD" : "REPLACE",
    root: parsed,
    collision: collisionMetaOf((root as { extras?: unknown }).extras),
  };
}

function numbers(value: unknown, length: number): number[] | null {
  if (!Array.isArray(value) || value.length !== length) return null;
  return value.every((item) => typeof item === "number" && Number.isFinite(item))
    ? (value as number[])
    : null;
}

/**
 * Distance from `point` to an oriented 3D Tiles `box` (centre, three half-axis vectors), 0
 * inside it: along each axis, how far the point is past the face.
 */
export function boxDistance(box: number[], point: readonly [number, number, number]): number {
  const dx = point[0] - (box[0] ?? 0);
  const dy = point[1] - (box[1] ?? 0);
  const dz = point[2] - (box[2] ?? 0);
  let squared = 0;
  for (let axis = 0; axis < 3; axis++) {
    const ux = box[3 + axis * 3] ?? 0;
    const uy = box[4 + axis * 3] ?? 0;
    const uz = box[5 + axis * 3] ?? 0;
    const half = Math.hypot(ux, uy, uz);
    if (half === 0) continue;
    const along = Math.abs((dx * ux + dy * uy + dz * uz) / half);
    const past = along - half;
    if (past > 0) squared += past * past;
  }
  return Math.sqrt(squared);
}

/**
 * A 3D Tiles bounding volume as a sphere: a `sphere` as it is; a `box` (centre, then three
 * half-axis vectors) as the sphere through its corners, whose radius is the length of the
 * three half-axes added -- sqrt of the sum of their squared lengths when they are orthogonal,
 * as 3D Tiles boxes are. A `region` (radians and metres on the ellipsoid) is left out: the
 * pipeline never writes one for a splat.
 */
export function sphereOf(volume: RawTile["boundingVolume"]): Sphere | null {
  const sphere = numbers(volume?.sphere, 4);
  if (sphere) {
    const [x, y, z, r] = sphere as [number, number, number, number];
    return { center: [x, y, z], radius: Math.abs(r) };
  }
  const box = numbers(volume?.box, 12);
  if (!box) return null;
  let squared = 0;
  for (let i = 3; i < 12; i++) squared += (box[i] ?? 0) ** 2;
  return { center: [box[0] ?? 0, box[1] ?? 0, box[2] ?? 0], radius: Math.sqrt(squared) };
}

/** Every tile of the tree, and the gaussians of the scan itself: its leaves under REPLACE
 *  (the parents are merged stand-ins for them), every tile under ADD. */
export function countTiles(tree: TileTree): { tiles: number; gaussians: number } {
  let tiles = 0;
  let gaussians = 0;
  const visit = (tile: TileNode): void => {
    tiles += 1;
    const own = Number.isFinite(tile.gaussians) ? tile.gaussians : 0;
    if (tree.refine === "ADD" || tile.children.length === 0) gaussians += own;
    tile.children.forEach(visit);
  };
  visit(tree.root);
  return { tiles, gaussians };
}

/**
 * The downloads of an ADD tileset (packed before merged parents: a parent holds a thinned
 * subset, drawn under its children), in order, holding at most `budget` gaussians apart from
 * the root, which is loaded whatever it holds: root first, then coarsest region first -- a
 * tile's priority is its parent's error, so a parent always comes before its children. A
 * REPLACE tileset streams with the view instead (stream.ts).
 */
export function planAdditive(tree: TileTree, budget: number): LoadStep[] {
  return planAdd(tree.root, budget);
}

function planAdd(root: TileNode, budget: number): LoadStep[] {
  interface Candidate {
    tile: TileNode;
    parent: TileNode | null;
    priority: number;
    depth: number;
    order: number;
  }
  const candidates: Candidate[] = [];
  const visit = (tile: TileNode, parent: TileNode | null, depth: number): void => {
    candidates.push({
      tile,
      parent,
      priority: parent ? parent.geometricError : Number.POSITIVE_INFINITY,
      depth,
      order: candidates.length,
    });
    for (const child of tile.children) visit(child, tile, depth + 1);
  };
  visit(root, null, 0);
  // A child's priority (its parent's error) is never above its parent's (the grandparent's
  // error), because error does not grow with depth; shallower first breaks the ties, so a
  // parent always comes before its children in this order.
  candidates.sort((a, b) => b.priority - a.priority || a.depth - b.depth || a.order - b.order);
  const chosen = new Set<TileNode>();
  const steps: LoadStep[] = [];
  let spent = 0;
  for (const { tile, parent } of candidates) {
    if (parent && !chosen.has(parent)) continue;
    if (parent && spent + tile.gaussians > budget) continue;
    chosen.add(tile);
    steps.push({ add: [tile], remove: [] });
    spent += Number.isFinite(tile.gaussians) ? tile.gaussians : 0;
  }
  return steps;
}
