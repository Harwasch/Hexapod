/**
 * View-dependent streaming of a scan's REPLACE tileset in the scan viewer: which tiles to
 * draw from where the camera is, and how to get from what is on screen to that.
 *
 * The viewer used to plan one static cut: the finest cut that fit `LOAD_FACTOR` times the
 * Detail budget, refined wherever the geometric error was largest, whatever the camera did.
 * A small scan fits whole, so that was fine; a 22.6M-gaussian site gets 1.6M spread evenly
 * over 150 m, and walking up to one bench shows the same coarse splats as the overview.
 *
 * Now the cut follows the camera, the way CesiumJS traverses 3D Tiles
 * (Cesium3DTilesetTraversal: a tile refines while its screen-space error is above the
 * maximum), with two differences a phone needs:
 *
 * - **A gaussian budget, spent on the largest error first.** The same greedy as the static
 *   plan, ordered by *screen* error instead of geometric error, so the budget goes where the
 *   camera is looking and close by; refinement stops early once every tile is under
 *   `TARGET_ERROR_PX`, since there is nothing left to see.
 * - **Full coverage, coarse where unseen.** Cesium draws only tiles in the frustum; turning
 *   the camera then shows a hole until the new tiles arrive. Here a tile out of view stays
 *   in the cut at a coarse level (its error weighted down by `OFFSCREEN_WEIGHT`, not
 *   dropped), so every direction always shows the scene -- blurry at first, then sharp: the
 *   Google Maps behaviour of motion first, detail after.
 *
 * Getting there never leaves a hole: a tile is swapped for its children only once all of
 * them have arrived, and descendants are swapped back for an ancestor only once it has
 * (Cesium's base-traversal rule, as the static plan used). Loaded tiles no longer wanted are
 * kept, least recently used first out, up to a cache budget, so looking back is instant; the
 * ancestors of what is drawn are never let go, so zooming out never waits.
 */

import { boxDistance, type Sphere, type TileNode, type TileTree } from "./tiles";

/** Where the camera is and what it sees, in the tileset's own frame. */
export interface View {
  eye: [number, number, number];
  /** Screen height in pixels over 2 tan(fovy / 2): pixels per metre at one metre away. */
  projection: number;
  /** Whether a tile's sphere is at least partly inside the view frustum. */
  visible(bounds: Sphere): boolean;
}

/** Refinement stops once every tile's error on screen is under this many CSS pixels. */
export const TARGET_ERROR_PX = 2;
/** A tile out of view refines as if its error were this fraction of what it is: the view
 *  gets the budget first, but what is around the camera stays close to as sharp, so turning
 *  round finds it loaded. At 0.15 every turn in place re-fetched what the last turn had just
 *  dropped (up to 7.4M gaussians a turn in the Fort Clatsop scan); at 0.5, nothing. */
export const OFFSCREEN_WEIGHT = 0.5;

/** Updates in a row a fetch may go unwanted before it is aborted (~0.3 s at the replan rate). */
export const ABANDON_AFTER_UPDATES = 2;
/** A tile that failed is tried again after this long, up to `MAX_LOAD_FAILURES` times: one
 *  dropped request used to leave its region coarse for the rest of the visit. */
export const RETRY_FAILED_MS = 5000;
export const MAX_LOAD_FAILURES = 3;

/** Nearer than this (metres, or a twentieth of a small tile) a tile's error stops growing:
 *  the camera is at or in it, and a closer look shows it no worse. */
export const NEAR_FLOOR_M = 0.5;

/**
 * The screen-space error of drawing `tile` instead of what is under it, in pixels, from the
 * camera's distance to the tile's box (Cesium's measure). The sphere around a box is far
 * looser: standing in a scan put the camera inside dozens of tiles' spheres, behind it
 * included, each then scored as if touching the lens -- and they spent the budget before the
 * tiles in front of it.
 */
export function screenError(tile: TileNode, view: View): number {
  if (tile.geometricError <= 0) return 0;
  const bounds = tile.bounds;
  if (!bounds) return Number.MAX_VALUE;
  const [x, y, z] = view.eye;
  const [cx, cy, cz] = bounds.center;
  const outside = tile.box
    ? boxDistance(tile.box, view.eye)
    : Math.hypot(x - cx, y - cy, z - cz) - bounds.radius;
  const floor = Math.max(Math.min(NEAR_FLOOR_M, bounds.radius / 20), 1e-6);
  return (tile.geometricError * view.projection) / Math.max(outside, floor);
}

/** How much refining `tile` is worth now: its screen error, weighted down when unseen. */
export function priority(tile: TileNode, view: View): number {
  const error = screenError(tile, view);
  return !tile.bounds || view.visible(tile.bounds) ? error : error * OFFSCREEN_WEIGHT;
}

/**
 * The cut to draw from `view`: the root, then repeatedly the tile with the largest priority
 * swapped for its children while that keeps the cut within `budget` gaussians (a swap that
 * does not fit is skipped, and smaller ones may still), until every tile left is under
 * `targetErrorPx`. The root is in it whatever it holds.
 */
export function chooseCut(
  tree: TileTree,
  view: View,
  budget: number,
  targetErrorPx = TARGET_ERROR_PX,
): { tiles: Set<TileNode>; gaussians: number } {
  const { root } = tree;
  const cut = new Set<TileNode>([root]);
  let spent = Number.isFinite(root.gaussians) ? root.gaussians : 0;
  const open: { tile: TileNode; value: number }[] = [{ tile: root, value: priority(root, view) }];
  while (open.length > 0) {
    let best = 0;
    for (let i = 1; i < open.length; i++) {
      if ((open[i]?.value ?? 0) > (open[best]?.value ?? 0)) best = i;
    }
    const [next] = open.splice(best, 1);
    if (!next) break;
    if (next.value <= targetErrorPx) break;
    const { tile } = next;
    if (tile.children.length === 0) continue;
    const children = tile.children.reduce((sum, child) => sum + child.gaussians, 0);
    const own = Number.isFinite(tile.gaussians) ? tile.gaussians : 0;
    if (!Number.isFinite(children) || spent - own + children > budget) continue;
    spent += children - own;
    cut.delete(tile);
    for (const child of tile.children) {
      cut.add(child);
      open.push({ tile: child, value: priority(child, view) });
    }
  }
  return { tiles: cut, gaussians: spent };
}

/** One change to what is drawn: `add` goes on screen in the same frame `remove` comes off. */
export interface Swap {
  add: TileNode[];
  remove: TileNode[];
}

/** Each tile's parent, the root's null. */
export function parents(tree: TileTree): Map<TileNode, TileNode | null> {
  const parent = new Map<TileNode, TileNode | null>([[tree.root, null]]);
  const visit = (tile: TileNode): void => {
    for (const child of tile.children) {
      parent.set(child, tile);
      visit(child);
    }
  };
  visit(tree.root);
  return parent;
}

/**
 * From `shown` towards `desired` without a hole: the swaps whose tiles are all loaded, and
 * the tiles still missing, in no particular order. A shown tile above the desired cut swaps
 * for its children once all of them are loaded (one level a step; the next step goes on); a
 * shown tile below it swaps back, with every shown tile under the same desired ancestor, once
 * that ancestor is loaded. Each root-to-leaf path meets both cuts exactly once, so every
 * shown tile is one or the other, or in `desired` already.
 */
export function nextSwaps(
  parentOf: Map<TileNode, TileNode | null>,
  shown: ReadonlySet<TileNode>,
  desired: ReadonlySet<TileNode>,
  isLoaded: (tile: TileNode) => boolean,
): { swaps: Swap[]; missing: TileNode[] } {
  const above = new Set<TileNode>();
  for (const tile of desired) {
    let up = parentOf.get(tile) ?? null;
    while (up && !above.has(up)) {
      above.add(up);
      up = parentOf.get(up) ?? null;
    }
  }
  const swaps: Swap[] = [];
  const missing = new Set<TileNode>();
  const coarsened = new Map<TileNode, TileNode[]>();
  for (const tile of shown) {
    if (desired.has(tile)) continue;
    if (above.has(tile)) {
      const absent = tile.children.filter((child) => !isLoaded(child));
      if (absent.length === 0) swaps.push({ add: tile.children, remove: [tile] });
      else absent.forEach((child) => missing.add(child));
      continue;
    }
    let up = parentOf.get(tile) ?? null;
    while (up && !desired.has(up)) up = parentOf.get(up) ?? null;
    if (!up) continue;
    const under = coarsened.get(up);
    if (under) under.push(tile);
    else coarsened.set(up, [tile]);
  }
  for (const [ancestor, under] of coarsened) {
    if (isLoaded(ancestor)) swaps.push({ add: [ancestor], remove: under });
    else missing.add(ancestor);
  }
  return { swaps, missing: [...missing] };
}

/** What the streamer asks of the page: fetch a tile, put it on screen or take it off. */
export interface StreamHost<M> {
  /** Fetches and decodes `tile`; `signal` aborts it once the view no longer wants it. */
  load(tile: TileNode, signal?: AbortSignal): Promise<M>;
  show(tile: TileNode, mesh: M): void;
  hide(tile: TileNode, mesh: M): void;
  dispose(mesh: M): void;
  /** A tile failed to load; it is not asked for again, and what covers it stays. */
  failed?(tile: TileNode, error: unknown): void;
}

export interface StreamOptions {
  /** Most gaussians the desired cut may hold. */
  budget: number;
  /** Most gaussians kept loaded, drawn or not; at least `budget`. */
  cacheBudget: number;
  /** Tiles fetched at once. */
  concurrency: number;
  targetErrorPx?: number;
  /**
   * Most gaussians put on screen in one update. Tiles that land together otherwise all go up
   * in one frame (each a GPU upload and a re-sort): a hitch. The rest wait for the next
   * update, which `onArrival` asks for at once. Unlimited when absent.
   */
  maxShownPerUpdate?: number;
  /**
   * Prefetch: once nothing the view wants is left to fetch, the next level of detail of the
   * cut within this many metres of the camera is fetched into the cache, so turning or
   * stepping there finds it ready -- the ring of a game world loaded around the player.
   * Off when absent.
   */
  prefetchRadiusM?: number;
}

/**
 * Drives a REPLACE tileset towards the cut the current view wants: call `update(view)` as the
 * camera moves (it is cheap: a few hundred tiles), and again when `onArrival` says a tile came.
 */
export class TileStreamer<M> {
  private readonly parentOf: Map<TileNode, TileNode | null>;
  private readonly loaded = new Map<TileNode, M>();
  private readonly used = new Map<TileNode, number>();
  /** Fetches under way, how many updates in a row each has not been wanted, and whether it
   *  is a prefetch (fetched for the cache, not for the view). */
  private readonly inFlight = new Map<
    TileNode,
    { abort: AbortController; unwanted: number; prefetch: boolean }
  >();
  /** Tiles that failed to load: how often, and when to try again. */
  private readonly broken = new Map<TileNode, { failures: number; retryAt: number }>();
  private readonly shown = new Set<TileNode>();
  private desired = new Set<TileNode>();
  private clock = 0;
  /** Where the camera was at the last update, for eviction: what is near stays. */
  private eye: [number, number, number] | null = null;
  private stopped = false;
  /** Called when a fetch finishes, so the page can run `update` again. */
  onArrival: (() => void) | null = null;

  constructor(
    private readonly tree: TileTree,
    private readonly host: StreamHost<M>,
    private options: StreamOptions,
  ) {
    this.parentOf = parents(tree);
  }

  /** A tile already loaded and drawn by the page (the root, used to frame the camera). */
  adopt(tile: TileNode, mesh: M): void {
    this.loaded.set(tile, mesh);
    this.shown.add(tile);
    this.used.set(tile, ++this.clock);
  }

  /** The tiles on screen. */
  get drawn(): TileNode[] {
    return [...this.shown];
  }

  /** Gaussians on screen. */
  get drawnGaussians(): number {
    let total = 0;
    this.shown.forEach((tile) => (total += Number.isFinite(tile.gaussians) ? tile.gaussians : 0));
    return total;
  }

  /** A new budget (and cache budget), from the next `update` on. */
  setBudget(budget: number, cacheBudget: number): void {
    this.options.budget = budget;
    this.options.cacheBudget = cacheBudget;
  }

  /** Tiles being fetched now. */
  get loading(): number {
    return this.inFlight.size;
  }

  /** Gaussians loaded, drawn or kept for later. */
  get loadedGaussians(): number {
    let total = 0;
    this.loaded.forEach((_mesh, tile) => (total += gaussiansOf(tile)));
    return total;
  }

  /** Whether anything is still to fetch for the last view. */
  get busy(): boolean {
    return this.inFlight.size > 0;
  }

  /** Moves the drawn tiles towards what `view` wants; returns whether anything changed. */
  update(view: View): boolean {
    if (this.stopped) return false;
    const { budget, targetErrorPx } = this.options;
    this.eye = view.eye;
    this.desired = chooseCut(this.tree, view, budget, targetErrorPx).tiles;
    const isLoaded = (tile: TileNode): boolean => this.loaded.has(tile);
    let changed = false;
    let missing: TileNode[] = [];
    const limit = this.options.maxShownPerUpdate ?? Number.POSITIVE_INFINITY;
    let room = limit;
    let deferred = false;
    // A swap can make the next one possible (children already cached from an earlier look),
    // so swap until nothing more can; the depth of the tree bounds it.
    for (let pass = 0; pass < 32; pass++) {
      if (this.shown.size === 0) {
        const { root } = this.tree;
        if (isLoaded(root)) this.apply({ add: [root], remove: [] });
        else missing = [root];
        if (!isLoaded(root)) break;
      }
      const next = nextSwaps(this.parentOf, this.shown, this.desired, isLoaded);
      missing = next.missing;
      if (next.swaps.length === 0) break;
      for (const swap of next.swaps) {
        const adding = swap.add.reduce(
          (sum, tile) => sum + (this.shown.has(tile) ? 0 : gaussiansOf(tile)),
          0,
        );
        // A swap bigger than the whole allowance still goes, alone: it cannot be split.
        if (adding > room && room < limit) {
          deferred = true;
          continue;
        }
        room -= adding;
        this.apply(swap);
        changed = true;
      }
      if (deferred) break;
    }
    if (deferred) setTimeout(() => this.onArrival?.(), 0);
    for (const tile of this.shown) this.used.set(tile, ++this.clock);
    for (const tile of this.desired) if (this.loaded.has(tile)) this.used.set(tile, this.clock);
    this.abandon(missing);
    this.fetch(missing, view);
    this.evict();
    return changed;
  }

  /** Disposes everything and ignores fetches still on their way. */
  stop(): void {
    this.stopped = true;
    for (const { abort } of this.inFlight.values()) abort.abort();
    this.inFlight.clear();
    for (const [tile, mesh] of this.loaded) {
      if (this.shown.has(tile)) this.host.hide(tile, mesh);
      this.host.dispose(mesh);
    }
    this.loaded.clear();
    this.shown.clear();
  }

  private apply(swap: Swap): void {
    for (const tile of swap.add) {
      const mesh = this.loaded.get(tile);
      if (mesh === undefined) continue;
      this.host.show(tile, mesh);
      this.shown.add(tile);
    }
    for (const tile of swap.remove) {
      const mesh = this.loaded.get(tile);
      if (mesh !== undefined) this.host.hide(tile, mesh);
      this.shown.delete(tile);
    }
  }

  /**
   * Aborts fetches the view has not wanted for `ABANDON_AFTER_UPDATES` updates in a row: a
   * camera on the move otherwise kept downloading tiles for where it had been, at the cost of
   * those for where it is (a game streamer's first rule: cancel what is no longer needed).
   */
  private abandon(missing: TileNode[]): void {
    const needed = new Set(missing);
    const radius = this.options.prefetchRadiusM ?? 0;
    for (const [tile, fetch] of this.inFlight) {
      if (needed.has(tile)) fetch.prefetch = false;
      fetch.unwanted = needed.has(tile) ? 0 : fetch.unwanted + 1;
      // A prefetch is never "wanted"; it goes only once the camera has left its ring.
      const stale = fetch.prefetch
        ? this.distance(tile) > 2 * radius
        : fetch.unwanted >= ABANDON_AFTER_UPDATES;
      if (stale) {
        fetch.abort.abort();
        this.inFlight.delete(tile);
      }
    }
  }

  /** Starts the most needed fetches: by their parent's priority (what a refine buys). */
  private fetch(missing: TileNode[], view: View): void {
    const now = performance.now();
    const wanted = missing.filter((tile) => {
      if (this.inFlight.has(tile)) return false;
      const broken = this.broken.get(tile);
      return !broken || (broken.failures < MAX_LOAD_FAILURES && now >= broken.retryAt);
    });
    const worth = (tile: TileNode): number => {
      const parent = this.parentOf.get(tile);
      return parent ? priority(parent, view) : Number.MAX_VALUE;
    };
    wanted.sort((a, b) => worth(b) - worth(a));
    for (const tile of wanted) {
      if (this.inFlight.size >= this.options.concurrency) break;
      this.start(tile, false);
    }
    if (
      this.inFlight.size < this.options.concurrency &&
      wanted.every((t) => this.inFlight.has(t))
    ) {
      this.prefetch();
    }
  }

  /** The next level of the cut near the camera, into the cache, nearest first. */
  private prefetch(): void {
    const radius = this.options.prefetchRadiusM;
    if (!radius) return;
    let room = this.options.cacheBudget - this.loadedGaussians;
    for (const fetch of this.inFlight.keys()) room -= gaussiansOf(fetch);
    const now = performance.now();
    const candidates: TileNode[] = [];
    for (const tile of this.desired) {
      if (tile.children.length === 0 || this.distance(tile) > radius) continue;
      for (const child of tile.children) {
        if (this.loaded.has(child) || this.inFlight.has(child)) continue;
        const broken = this.broken.get(child);
        if (broken && (broken.failures >= MAX_LOAD_FAILURES || now < broken.retryAt)) continue;
        candidates.push(child);
      }
    }
    candidates.sort((a, b) => this.distance(a) - this.distance(b));
    for (const tile of candidates) {
      if (this.inFlight.size >= this.options.concurrency) break;
      if (gaussiansOf(tile) > room) break;
      room -= gaussiansOf(tile);
      this.start(tile, true);
    }
  }

  private start(tile: TileNode, prefetch: boolean): void {
    const abort = new AbortController();
    this.inFlight.set(tile, { abort, unwanted: 0, prefetch });
    this.host.load(tile, abort.signal).then(
      (mesh) => {
        if (abort.signal.aborted || this.stopped) {
          this.host.dispose(mesh);
          return;
        }
        this.inFlight.delete(tile);
        this.broken.delete(tile);
        this.loaded.set(tile, mesh);
        this.used.set(tile, ++this.clock);
        this.onArrival?.();
      },
      (error: unknown) => {
        if (abort.signal.aborted) return;
        this.inFlight.delete(tile);
        const failures = (this.broken.get(tile)?.failures ?? 0) + 1;
        this.broken.set(tile, { failures, retryAt: performance.now() + RETRY_FAILED_MS });
        if (!this.stopped) {
          this.host.failed?.(tile, error);
          this.onArrival?.();
        }
      },
    );
  }

  /** How far the camera is from `tile` (its box, else its sphere); 0 with no camera yet. */
  private distance(tile: TileNode): number {
    const eye = this.eye;
    if (!eye) return 0;
    if (tile.box) return boxDistance(tile.box, eye);
    const bounds = tile.bounds;
    if (!bounds) return 0;
    const [cx, cy, cz] = bounds.center;
    return Math.max(0, Math.hypot(eye[0] - cx, eye[1] - cy, eye[2] - cz) - bounds.radius);
  }

  /**
   * Farthest from the camera first, then least recently used, until the cache fits. Never a
   * drawn or wanted tile, nor an ancestor of a drawn one: those are what a step back or a zoom
   * out swaps to, and merged parents are an eighth or less of what they stand for.
   *
   * It used to drop the finest tiles first -- but the finest tiles loaded are the ones right
   * around the camera, which turning away takes off screen and turning back wants again:
   * spinning in place in the Fort Clatsop scan re-downloaded up to 7.4M gaussians a turn.
   * Distance keeps what surrounds the camera; it is what any direction will show next.
   */
  private evict(): void {
    let total = 0;
    this.loaded.forEach((_mesh, tile) => (total += gaussiansOf(tile)));
    if (total <= this.options.cacheBudget) return;
    const kept = new Set<TileNode>();
    for (const tile of this.shown) {
      let up = this.parentOf.get(tile) ?? null;
      while (up && !kept.has(up)) {
        kept.add(up);
        up = this.parentOf.get(up) ?? null;
      }
    }
    const spare = [...this.loaded.keys()]
      .filter((tile) => !this.shown.has(tile) && !this.desired.has(tile) && !kept.has(tile))
      .sort(
        (a, b) =>
          this.distance(b) - this.distance(a) || (this.used.get(a) ?? 0) - (this.used.get(b) ?? 0),
      );
    for (const tile of spare) {
      if (total <= this.options.cacheBudget) break;
      const mesh = this.loaded.get(tile);
      if (mesh !== undefined) this.host.dispose(mesh);
      this.loaded.delete(tile);
      this.used.delete(tile);
      total -= gaussiansOf(tile);
    }
  }
}

function gaussiansOf(tile: TileNode): number {
  return Number.isFinite(tile.gaussians) ? tile.gaussians : 0;
}
