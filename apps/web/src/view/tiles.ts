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
 * (main.ts). What is left here is what to *download*: all of it when it fits, since then a
 * close look shows everything the scan has; otherwise the finest cut that fits `loadBudget`,
 * refined where the error is largest first. Each tile's gaussian count is in its
 * `extras.gaussians`, so nothing is fetched to decide.
 *
 * Tilesets packed before merged parents are ADD (a parent holds a thinned subset, drawn under
 * its children): there, tiles are only ever added. A single-tile tileset -- anything packed
 * before the hierarchy, and the committed tree -- is one step either way.
 */

export type Refine = "ADD" | "REPLACE";

export interface TileNode {
  uri: string;
  gaussians: number;
  geometricError: number;
  children: TileNode[];
}

export interface TileTree {
  refine: Refine;
  root: TileNode;
}

/** One step of loading: fetch `add`, show it, then drop `remove` (the tile it replaces). */
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
    };
  };
  const parsed = node(root);
  if (!parsed.uri) throw new Error("The scan's tileset names no content.");
  return { refine: root.refine === "ADD" ? "ADD" : "REPLACE", root: parsed };
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
 * The downloads, in order, holding at most `budget` gaussians once each step is done -- apart
 * from the root, which is loaded whatever it holds.
 *
 * REPLACE: start from the root and repeatedly replace the loaded tile with the largest
 * geometric error by its children, while the swap keeps the total within the budget; a swap
 * that does not fit is skipped, and smaller ones after it may still fill the gap. Each step's
 * children are shown together and only then is their parent dropped -- the rule CesiumJS's
 * base traversal applies (Cesium3DTilesetBaseTraversal.js: a REPLACE tile refines only once
 * all its children are loaded), so the scan never has a hole while it streams.
 *
 * ADD: the tiles themselves, root first, then coarsest-region-first: a tile's priority is its
 * parent's error, so a parent always comes before its children.
 */
export function planLoads(tree: TileTree, budget: number): LoadStep[] {
  return tree.refine === "ADD" ? planAdd(tree.root, budget) : planReplace(tree.root, budget);
}

function planReplace(root: TileNode, budget: number): LoadStep[] {
  const steps: LoadStep[] = [{ add: [root], remove: [] }];
  let spent = Number.isFinite(root.gaussians) ? root.gaussians : 0;
  // The loaded tiles that could still be refined, largest error first; ties to the earlier.
  const open: { tile: TileNode; order: number }[] = [{ tile: root, order: 0 }];
  let order = 1;
  while (open.length > 0) {
    open.sort((a, b) => b.tile.geometricError - a.tile.geometricError || a.order - b.order);
    const next = open.shift();
    if (!next || next.tile.children.length === 0) continue;
    const { tile } = next;
    const children = tile.children.reduce((sum, child) => sum + child.gaussians, 0);
    const after = spent - (Number.isFinite(tile.gaussians) ? tile.gaussians : 0) + children;
    if (!Number.isFinite(children) || after > budget) continue;
    spent = after;
    steps.push({ add: tile.children, remove: [tile] });
    for (const child of tile.children) open.push({ tile: child, order: order++ });
  }
  return steps;
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

/** The gaussians loaded once every step has run: what the plan costs the phone. */
export function plannedGaussians(steps: LoadStep[]): number {
  const loaded = new Set<TileNode>();
  for (const step of steps) {
    step.remove.forEach((tile) => loaded.delete(tile));
    step.add.forEach((tile) => loaded.add(tile));
  }
  let total = 0;
  loaded.forEach((tile) => (total += Number.isFinite(tile.gaussians) ? tile.gaussians : 0));
  return total;
}
