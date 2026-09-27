/**
 * Which tiles of a scan's level-of-detail tileset the viewer loads, within a gaussian budget.
 *
 * `tools/captures/splat_tiles.py` packs every gaussian of a scan into an octree of tiles
 * with ADD refinement: each tile holds different gaussians, a parent an even subset of what
 * is under it, so any set of tiles closed under "parent first" is a valid, seamless view of
 * the scan and the whole set is the scan itself. A scan of 5M gaussians does not fit a phone;
 * a few hundred thousand of them, spread evenly, does.
 *
 * Spark has no 3D Tiles traversal, and the viewer is an object on a turntable rather than a
 * landscape to fly through, so the choice is made once, view-independently: refine where the
 * error is largest first. A tile's priority is its *parent's* geometric error -- what its
 * region looks like until it arrives -- so the root's coarse subset is refined everywhere
 * before anything is refined twice. Tiles are taken in that order while they fit the budget;
 * one that does not fit is skipped with everything under it, and smaller ones after it may
 * still fill the gap. Each tile's gaussian count is in its `extras.gaussians`, so nothing is
 * fetched to decide.
 */

export interface TileNode {
  uri: string;
  gaussians: number;
  geometricError: number;
  children: TileNode[];
}

interface RawTile {
  content?: { uri?: unknown };
  geometricError?: unknown;
  extras?: { gaussians?: unknown };
  children?: unknown;
}

/**
 * The tile tree of a `tileset.json`. A tile without a gaussian count (a tileset packed before
 * the hierarchy, or from elsewhere) counts as unaffordable -- except the root, which is always
 * loaded: without it there is nothing to show.
 */
export function parseTileset(document: unknown): TileNode {
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
  return parsed;
}

/** Every tile of the tree. */
export function countTiles(root: TileNode): { tiles: number; gaussians: number } {
  let tiles = 0;
  let gaussians = 0;
  const visit = (tile: TileNode): void => {
    tiles += 1;
    gaussians += Number.isFinite(tile.gaussians) ? tile.gaussians : 0;
    tile.children.forEach(visit);
  };
  visit(root);
  return { tiles, gaussians };
}

/**
 * The tiles to load, in load order (root first, then coarsest-region-first), holding at most
 * `budget` gaussians between them -- apart from the root, which is loaded whatever it holds.
 */
export function chooseTiles(root: TileNode, budget: number): TileNode[] {
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
  const out: TileNode[] = [];
  let spent = 0;
  for (const { tile, parent } of candidates) {
    if (parent && !chosen.has(parent)) continue;
    if (parent && spent + tile.gaussians > budget) continue;
    chosen.add(tile);
    out.push(tile);
    spent += Number.isFinite(tile.gaussians) ? tile.gaussians : 0;
  }
  return out;
}
