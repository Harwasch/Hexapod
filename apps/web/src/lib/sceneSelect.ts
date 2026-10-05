/**
 * Selecting a scan's objects in the scene (docs/SCENE_OBJECTS.md, "Selecting in the scene"):
 * what a click offers, which of it is chosen first, how the choice cycles, and which object a
 * painted area is. Pure; the picking itself is `splatPick.ts`, the wiring
 * `cesium/sceneSelect/`.
 *
 * A click hits splats, and each splat carries one id: its leaf instance. The candidates are
 * that leaf's chain up to the top level (leaf → … → top), then the other instances whose
 * splats the ray also met near its front, each with the part of its own chain not already
 * offered. A click chooses the whole object first, the top of the chain, and each click again
 * on it one level finer (`drillIndex`): a cable spool, then one of its planks.
 *
 * A painted area is matched against every instance, at every level, by intersection over union
 * in splats weighted by opacity, counting only what is visible from the camera (an instance's
 * hidden back does not count against it): the best match is selected, and below
 * `PAINT_MIN_IOU` the painted splats themselves can become an object of their own
 * (`customSets.ts`). While a stroke is painted the match is kept up to date from an index of
 * the view (`paintIndex`), which walks only the painted cells.
 */

import { categoryById } from "./categories";
import type { Instance, InstancesDoc } from "./instances";
import type { BrushMask, ScreenSplats } from "./splatPaint";

/** A candidate's share of the clicked pixel, below which another instance is not offered. */
export const NEAR_SHARE = 0.15;
/** A candidate must be this close behind the first hit (share of its distance) to be offered. */
export const NEAR_DEPTH = 0.25;
/**
 * A top-level instance with more than this share of a scan's splats is the scene itself, not
 * an object in it: a click starts one level below it.
 */
export const WHOLE_SCENE_SHARE = 0.5;
/** Below this, a painted area is offered as an object of its own. */
export const PAINT_MIN_IOU = 0.5;

/** The instance, its parent, … up to the top level; cycles and unknown ids end the chain. */
export function chainOf(doc: Pick<InstancesDoc, "byId">, id: number): number[] {
  const out: number[] = [];
  const seen = new Set<number>();
  for (let at: number | null = id; at !== null && !seen.has(at);) {
    const instance = doc.byId.get(at);
    if (!instance) break;
    seen.add(at);
    out.push(at);
    at = instance.parent;
  }
  return out;
}

/** What a click offers. */
export interface Candidates {
  /** Leaf → top of the main hit, then the other hits' chains (what is not already offered). */
  ids: number[];
  /** How many of `ids` are the main hit's chain. */
  chain: number;
}

/**
 * The candidates from the instances a ray hit (`splatPick.hitWeights`: per leaf id, its share
 * of the pixel and nearest distance). Id 0 (no instance) and ids the file does not list are
 * skipped.
 */
export function buildCandidates(
  doc: Pick<InstancesDoc, "byId">,
  weights: ReadonlyMap<number, { weight: number; t: number }>,
  { share = NEAR_SHARE, depth = NEAR_DEPTH }: { share?: number; depth?: number } = {},
): Candidates {
  const hits = [...weights]
    .filter(([id]) => id !== 0 && doc.byId.has(id))
    .sort((a, b) => b[1].weight - a[1].weight || a[1].t - b[1].t || a[0] - b[0]);
  const first = hits[0];
  if (!first) return { ids: [], chain: 0 };
  const ids = chainOf(doc, first[0]);
  const chain = ids.length;
  const offered = new Set(ids);
  const front = first[1].t;
  for (const [id, hit] of hits.slice(1)) {
    if (hit.weight < share * first[1].weight) continue;
    if (hit.t > front * (1 + depth) + 0.05) continue;
    for (const up of chainOf(doc, id)) {
      if (offered.has(up)) break;
      offered.add(up);
      ids.push(up);
    }
  }
  return { ids, chain };
}

const shares = new WeakMap<object, ReadonlyMap<number, number>>();

/**
 * Per instance, its share of the scan's splats with everything below it (an instance's
 * `splats` are its own leaf splats only), once per document. The splats without an instance
 * are not in the file, so it is a share of those that have one.
 */
export function splatShares(
  doc: Pick<InstancesDoc, "instances" | "byId">,
): ReadonlyMap<number, number> {
  const known = shares.get(doc);
  if (known) return known;
  const out = new Map<number, number>();
  let total = 0;
  for (const instance of doc.instances) {
    total += instance.splats;
    for (const up of chainOf(doc, instance.id)) out.set(up, (out.get(up) ?? 0) + instance.splats);
  }
  if (total > 0) for (const [id, splats] of out) out.set(id, splats / total);
  shares.set(doc, out);
  return out;
}

/**
 * Which of `candidates` a click chooses, given the selection `current` (on the same scan, else
 * null): the whole object first, then one level finer per click. With nothing selected, or a
 * selection the hit's chain does not hold (another object, a nearby one), the top of the chain
 * -- below any top that is most of the scan (`shareOf` over `maxShare`), down the hit's chain.
 * With the selection in the chain, the level below it toward the hit leaf; at the leaf, the
 * leaf again. -1 when there is nothing.
 */
export function drillIndex(
  candidates: Candidates,
  current: number | null,
  shareOf: (id: number) => number = () => 0,
  maxShare = WHOLE_SCENE_SHARE,
): number {
  if (candidates.ids.length === 0) return -1;
  const chain = Math.max(1, Math.min(candidates.chain, candidates.ids.length));
  const at = current === null ? -1 : candidates.ids.indexOf(current);
  if (at >= 0 && at < chain) return Math.max(0, at - 1);
  let top = chain - 1;
  while (top > 0 && shareOf(candidates.ids[top] ?? 0) > maxShare) top--;
  return top;
}

/** `index` moved by `step` around `count` candidates. */
export function cycleIndex(index: number, count: number, step: number): number {
  if (count <= 0) return -1;
  return (((index + step) % count) + count) % count;
}

/** "Some words" from an id-like name: `tall_grass` → "Tall grass". */
function humanise(name: string): string {
  const words = name.replace(/[_-]+/g, " ").trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/**
 * What an instance is called, as the objects panel names it (lib/categories.ts): its top tag,
 * else its broad category's name (`category`, a category id, else the file's own), never an id.
 */
export function selectionLabel(
  instance: Instance | undefined,
  id: number,
  category?: string,
): string {
  const tag = instance?.tags[0]?.label;
  if (tag) return humanise(tag);
  const known = category ?? instance?.category;
  if (typeof known === "string" && known !== "") {
    const named = categoryById(known);
    return named.id === known ? named.name : humanise(known);
  }
  return id > 0 ? "Unnamed object" : "Nothing";
}

/**
 * Where candidate `index` is among `count` a click offered, the first `chain` of them the hit's
 * chain: in the chain its level counted from the top, "1 of 3" the whole object and "3 of 3"
 * its finest part, with "+2 nearby" when other instances were offered too; past the chain,
 * which of those it is, "Nearby 1 of 2". Empty for a lone candidate.
 */
export function levelText(index: number, chain: number, count: number): string {
  if (count <= 1 || index < 0) return "";
  const levels = Math.max(1, Math.min(chain, count));
  const nearby = count - levels;
  if (index >= levels) return `Nearby ${String(index - levels + 1)} of ${String(nearby)}`;
  const level = levels > 1 ? `${String(levels - index)} of ${String(levels)}` : "";
  return [level, nearby > 0 ? `+${String(nearby)} nearby` : ""].filter(Boolean).join(" · ");
}

/** A selection with its place among the candidates, "Spool · 1 of 3": the card's name. */
export function chipText(label: string, index: number, chain: number, count: number): string {
  const level = levelText(index, chain, count);
  return level ? `${label} · ${level}` : label;
}

/** What is visible on screen, per splat: its leaf id, its weight, and whether it was painted. */
export interface PaintSample {
  ids: ArrayLike<number>;
  weights: ArrayLike<number>;
  painted: ArrayLike<number>;
}

export interface PaintMatch {
  id: number;
  iou: number;
}

/**
 * The instance (any level) whose visible splats best match the painted ones, by intersection
 * over union in weight; null when nothing painted carries an instance.
 */
export function bestByIoU(doc: Pick<InstancesDoc, "byId">, sample: PaintSample): PaintMatch | null {
  const chains = new Map<number, number[]>();
  const chain = (id: number): number[] => {
    let c = chains.get(id);
    if (!c) {
      c = chainOf(doc, id);
      chains.set(id, c);
    }
    return c;
  };
  const visible = new Map<number, number>();
  const inter = new Map<number, number>();
  let painted = 0;
  const n = sample.ids.length;
  for (let i = 0; i < n; i++) {
    const w = sample.weights[i] ?? 0;
    if (w <= 0) continue;
    const isPainted = (sample.painted[i] ?? 0) !== 0;
    if (isPainted) painted += w;
    const id = sample.ids[i] ?? 0;
    if (id === 0) continue;
    for (const up of chain(id)) {
      visible.set(up, (visible.get(up) ?? 0) + w);
      if (isPainted) inter.set(up, (inter.get(up) ?? 0) + w);
    }
  }
  const ids = [...inter.keys()].sort((x, y) => x - y);
  return bestOf(
    ids,
    (id) => inter.get(id) ?? 0,
    (id) => visible.get(id) ?? 0,
    painted,
  );
}

/**
 * The best of the instances a painted area met (`ids`, ascending), by intersection over union
 * of their painted weight (`interOf`) and visible weight (`visibleOf`) with `painted`. In id
 * order, so the same sums give the same choice however they were gathered; of two as good, the
 * smaller id.
 */
function bestOf(
  ids: Iterable<number>,
  interOf: (id: number) => number,
  visibleOf: (id: number) => number,
  painted: number,
): PaintMatch | null {
  let best: PaintMatch | null = null;
  for (const id of ids) {
    const i = interOf(id);
    const union = painted + visibleOf(id) - i;
    const iou = union > 0 ? i / union : 0;
    if (!best || iou > best.iou + 1e-9) best = { id, iou };
  }
  return best;
}

/**
 * A view's visible splats by screen cell, to match a painted area while it is painted
 * (`bestByIoUIndexed`): `bestByIoU` over every splat on screen is too slow to run as a stroke
 * moves (the camp has 22.6 M). Built once per view; a match then costs the painted cells and
 * the instances they hold.
 */
export interface PaintIndex {
  cols: number;
  rows: number;
  /** Per cell, where its entries start (`cols × rows + 1` offsets into the entries). */
  start: Uint32Array;
  /** Per entry: a leaf id, and the weight of the cell's visible splats carrying it. */
  ids: Uint32Array;
  weights: Float64Array;
  /** Per cell, the weight and the number of its visible splats. */
  cellWeights: Float64Array;
  cellSplats: Uint32Array;
  /** Per instance id (any level), the weight of its visible splats with everything below it. */
  visible: Float64Array;
  /** Per leaf id on screen that the file lists, its chain (`chainOf`). */
  chains: ReadonlyMap<number, readonly number[]>;
}

/** A cell's splats of one id share an entry, looked for among its last this many. */
const MERGE_RECENT = 8;

/**
 * The index of a projected view (`splatPaint.projectTiles`): its splats visible from the camera
 * (`visible`, as `splatPaint.visibleSplats`) and their leaf ids (`ids`), weighted by opacity
 * as `bestByIoU` weighs them. The weights are summed in doubles: a sum of float32 opacities is
 * exact there (to 2^25 splats), so it is the same however it is grouped, and the match is
 * `bestByIoU`'s to the last bit.
 */
export function paintIndex(
  doc: Pick<InstancesDoc, "byId" | "maxId">,
  screen: Pick<ScreenSplats, "cols" | "rows" | "cell" | "opacity" | "count">,
  visible: ArrayLike<number>,
  ids: ArrayLike<number>,
): PaintIndex {
  const { cell, opacity, count } = screen;
  const cells = screen.cols * screen.rows;
  const cellSplats = new Uint32Array(cells);
  for (let k = 0; k < count; k++) {
    if (!visible[k] || !((opacity[k] ?? 0) > 0)) continue;
    const c = cell[k] ?? 0;
    cellSplats[c] = (cellSplats[c] ?? 0) + 1;
  }
  // As many slots a cell as it has splats; its splats of one id (a tile's are in runs) share one.
  const offset = new Uint32Array(cells + 1);
  for (let c = 0; c < cells; c++) offset[c + 1] = (offset[c] ?? 0) + (cellSplats[c] ?? 0);
  const fill = offset.slice(0, cells);
  const slotIds = new Uint32Array(offset[cells] ?? 0);
  const slotWeights = new Float64Array(offset[cells] ?? 0);
  for (let k = 0; k < count; k++) {
    const w = opacity[k] ?? 0;
    if (!visible[k] || !(w > 0)) continue;
    const c = cell[k] ?? 0;
    const id = ids[k] ?? 0;
    const end = fill[c] ?? 0;
    const low = Math.max(offset[c] ?? 0, end - MERGE_RECENT);
    let at = end - 1;
    while (at >= low && slotIds[at] !== id) at--;
    if (at < low) {
      at = end;
      fill[c] = end + 1;
      slotIds[at] = id;
    }
    slotWeights[at] = (slotWeights[at] ?? 0) + w;
  }
  // Packed, cell after cell, with each listed leaf's visible weight.
  let entries = 0;
  for (let c = 0; c < cells; c++) entries += (fill[c] ?? 0) - (offset[c] ?? 0);
  const start = new Uint32Array(cells + 1);
  const packedIds = new Uint32Array(entries);
  const packedWeights = new Float64Array(entries);
  const cellWeights = new Float64Array(cells);
  const span = doc.maxId + 1;
  const leaves = new Float64Array(span);
  let e = 0;
  for (let c = 0; c < cells; c++) {
    start[c] = e;
    let total = 0;
    for (let j = offset[c] ?? 0; j < (fill[c] ?? 0); j++, e++) {
      const id = slotIds[j] ?? 0;
      const w = slotWeights[j] ?? 0;
      packedIds[e] = id;
      packedWeights[e] = w;
      total += w;
      if (id !== 0 && id < span) leaves[id] = (leaves[id] ?? 0) + w;
    }
    cellWeights[c] = total;
  }
  start[cells] = e;
  // Rolled up each leaf's chain.
  const chains = new Map<number, number[]>();
  const rolled = new Float64Array(span);
  for (let leaf = 1; leaf < span; leaf++) {
    const w = leaves[leaf] ?? 0;
    if (w === 0) continue;
    const chain = chainOf(doc, leaf);
    if (chain.length === 0) continue;
    chains.set(leaf, chain);
    for (const up of chain) rolled[up] = (rolled[up] ?? 0) + w;
  }
  return {
    cols: screen.cols,
    rows: screen.rows,
    start,
    ids: packedIds,
    weights: packedWeights,
    cellWeights,
    cellSplats,
    visible: rolled,
    chains,
  };
}

/**
 * `bestByIoU` of the splats under `mask` (the view's brush) through the view's index: the same
 * match, walking only the painted cells and the instances they hold.
 */
export function bestByIoUIndexed(
  index: PaintIndex,
  mask: Pick<BrushMask, "data">,
): PaintMatch | null {
  const { start, ids, weights, cellWeights, visible } = index;
  const span = visible.length;
  // Per leaf id, its painted weight; ids are dense, so arrays beat maps here by far.
  const leaves = new Float64Array(span);
  const touched: number[] = [];
  let painted = 0;
  const cells = Math.min(mask.data.length, index.cols * index.rows);
  for (let c = 0; c < cells; c++) {
    if (!mask.data[c]) continue;
    painted += cellWeights[c] ?? 0;
    const end = start[c + 1] ?? 0;
    for (let e = start[c] ?? 0; e < end; e++) {
      const id = ids[e] ?? 0;
      if (id === 0 || id >= span) continue;
      const before = leaves[id] ?? 0;
      if (before === 0) touched.push(id);
      leaves[id] = before + (weights[e] ?? 0);
    }
  }
  const inter = new Float64Array(span);
  const met: number[] = [];
  for (const leaf of touched) {
    const w = leaves[leaf] ?? 0;
    for (const up of index.chains.get(leaf) ?? []) {
      const before = inter[up] ?? 0;
      if (before === 0) met.push(up);
      inter[up] = before + w;
    }
  }
  return bestOf(
    Uint32Array.from(met).sort(),
    (id) => inter[id] ?? 0,
    (id) => visible[id] ?? 0,
    painted,
  );
}

/** How many visible splats are under `mask`, through the view's index. */
export function paintedCount(index: PaintIndex, mask: Pick<BrushMask, "data">): number {
  let count = 0;
  const cells = Math.min(mask.data.length, index.cols * index.rows);
  for (let c = 0; c < cells; c++) if (mask.data[c]) count += index.cellSplats[c] ?? 0;
  return count;
}

/**
 * What to hide so only `ids` (with what they contain) stay drawn: every other instance but
 * their ancestors -- hiding an ancestor hides what is under it in a renderer that expands to
 * descendants, so they stay, and their other children are hidden one by one.
 */
export function hiddenForShowOnly(
  doc: Pick<InstancesDoc, "instances" | "byId">,
  keep: ReadonlySet<number>,
): number[] {
  const spared = new Set(keep);
  for (const id of keep) for (const up of chainOf(doc, id)) spared.add(up);
  return doc.instances.filter((i) => !spared.has(i.id)).map((i) => i.id);
}
