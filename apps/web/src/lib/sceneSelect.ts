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
 * A painted area is matched by intersection over union in splats weighted by opacity, counting
 * only what is visible from the camera (an instance's hidden back does not count against it),
 * against combinations of instances at whatever levels fit (`bestSet`): the set of instances
 * from disjoint subtrees whose union best matches the painted area -- both flanges of a spool,
 * or the spool itself when that is as good. Below `PAINT_MIN_IOU` the painted splats
 * themselves can become an object of their own (`customSets.ts`). While a stroke is painted
 * the match is kept up to date from an index of the view (`paintIndex`), which walks only the
 * painted cells, and held steady between near-equal answers (`steadySet`).
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
/**
 * A painted area's best set drops members that add little, the least first, while its IoU
 * stays within this share of the best's: a sliver of a neighbour under the brush's edge is not
 * a part. A share, not a difference, so a member that is half of a loosely painted set stays.
 */
export const PAINT_SET_GAIN = 0.02;
/**
 * Then members that share an ancestor become that ancestor, the deepest first, while the IoU
 * stays within this share of what it was: the whole spool rather than its three parts when the
 * spool is as good, its parts when the spool also holds ground nobody painted.
 */
export const PAINT_PARENT_SLACK = 0.03;
/**
 * While a stroke is painted, the match shown is kept until another's IoU is better by more
 * than this share, so two near-equal answers do not take turns at every preview (`steadySet`).
 */
export const PAINT_STEADY = 0.03;

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
 * What a painted area met, per instance at every level: the sums every match is made from,
 * the same whether gathered over every visible splat (`paintSums`) or through the view's index
 * (`paintSumsIndexed`). Weights are opacities summed in doubles, exact however grouped.
 */
export interface PaintSums {
  /** Weight of the painted splats (visible and under the brush), with an instance or not. */
  painted: number;
  /** The instances the painted splats carry, at every level (each with its chain), ascending. */
  ids: Uint32Array;
  /** Per entry of `ids`: its painted weight and its visible weight, with everything below it. */
  inter: Float64Array;
  visible: Float64Array;
  /** Per entry, its parent's entry, or -1 at the top of its chain. */
  parent: Int32Array;
  /** The visible weight of any instance, painted or not (0 off screen). */
  visibleOf: (id: number) => number;
}

/** `id`'s entry in `ids` (ascending), or -1. */
function entryIn(ids: Uint32Array, id: number): number {
  let lo = 0;
  let hi = ids.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    const at = ids[mid] ?? 0;
    if (at === id) return mid;
    if (at < id) lo = mid + 1;
    else hi = mid - 1;
  }
  return -1;
}

/** The sums of the instances `ids` (ascending) met, from per-id lookups. */
function sumsOf(
  ids: Uint32Array,
  interOf: (id: number) => number,
  visibleOf: (id: number) => number,
  parentOf: (id: number) => number,
  painted: number,
): PaintSums {
  const n = ids.length;
  const inter = new Float64Array(n);
  const visible = new Float64Array(n);
  const parent = new Int32Array(n);
  for (let k = 0; k < n; k++) {
    const id = ids[k] ?? 0;
    inter[k] = interOf(id);
    visible[k] = visibleOf(id);
    const up = parentOf(id);
    parent[k] = up > 0 ? entryIn(ids, up) : -1;
  }
  return { painted, ids, inter, visible, parent, visibleOf };
}

/**
 * The sums of a painted area over every visible splat (`sample`). The hierarchy is the one the
 * chains on screen give, leaf by leaf in id order: each instance's parent is the next on the
 * first chain that holds it (`paintIndex` reads it so too).
 */
export function paintSums(doc: Pick<InstancesDoc, "byId">, sample: PaintSample): PaintSums {
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
  const parents = new Map<number, number>();
  for (const leaf of [...chains.keys()].sort((a, b) => a - b)) {
    const c = chains.get(leaf) ?? [];
    for (let k = 0; k < c.length; k++) {
      const id = c[k] ?? 0;
      if (!parents.has(id)) parents.set(id, c[k + 1] ?? 0);
    }
  }
  return sumsOf(
    Uint32Array.from(inter.keys()).sort(),
    (id) => inter.get(id) ?? 0,
    (id) => visible.get(id) ?? 0,
    (id) => parents.get(id) ?? 0,
    painted,
  );
}

/**
 * The single instance (any level) whose visible splats best match the painted ones, by
 * intersection over union in weight; null when nothing painted carries an instance. In id
 * order, so the same sums give the same choice however they were gathered; of two as good, the
 * smaller id.
 */
export function bestSingle(sums: PaintSums): PaintMatch | null {
  let best: PaintMatch | null = null;
  for (let k = 0; k < sums.ids.length; k++) {
    const i = sums.inter[k] ?? 0;
    const union = sums.painted + (sums.visible[k] ?? 0) - i;
    const iou = union > 0 ? i / union : 0;
    if (!best || iou > best.iou + 1e-9) best = { id: sums.ids[k] ?? 0, iou };
  }
  return best;
}

/** `bestSingle` of the painted splats of `sample`. */
export function bestByIoU(doc: Pick<InstancesDoc, "byId">, sample: PaintSample): PaintMatch | null {
  return bestSingle(paintSums(doc, sample));
}

/** A painted area's best combination of instances (`bestSet`). */
export interface PaintSetMatch {
  /**
   * The members, from disjoint subtrees (none holds another), the largest on screen first
   * (ascending ids among equals): one id when a single instance is the best match.
   */
  ids: number[];
  iou: number;
}

/** Rounds of the search (`bestSet`) at most; it settles in a handful. */
const SET_ROUNDS = 64;

/** The entries of `parent` by depth, deepest first (ascending within a depth): children first. */
function byDepth(parent: Int32Array): Int32Array {
  const n = parent.length;
  const depth = new Int32Array(n).fill(-1);
  let deepest = 0;
  const path: number[] = [];
  for (let k = 0; k < n; k++) {
    path.length = 0;
    let at = k;
    while (at >= 0 && depth[at] === -1 && path.length <= n) {
      path.push(at);
      at = parent[at] ?? -1;
    }
    let d = at >= 0 ? (depth[at] ?? 0) : -1;
    for (let q = path.length - 1; q >= 0; q--) depth[path[q] ?? 0] = ++d;
    if (d > deepest) deepest = d;
  }
  const starts = new Int32Array(deepest + 2);
  for (let k = 0; k < n; k++) {
    const slot = deepest - (depth[k] ?? 0) + 1;
    starts[slot] = (starts[slot] ?? 0) + 1;
  }
  for (let d = 1; d < starts.length; d++) starts[d] = (starts[d] ?? 0) + (starts[d - 1] ?? 0);
  const order = new Int32Array(n);
  for (let k = 0; k < n; k++) {
    const slot = deepest - (depth[k] ?? 0);
    const at = starts[slot] ?? 0;
    order[at] = k;
    starts[slot] = at + 1;
  }
  return order;
}

/**
 * The combination of instances whose union best matches the painted area, by intersection over
 * union: a set from disjoint subtrees, at whatever levels fit. With disjoint members the union's
 * painted and visible weights are the members' sums, so a set's IoU is
 * `Σinter / (painted + Σvisible − Σinter)`.
 *
 * Found exactly, not greedily: a greedy search that takes the best single instance first can
 * never trade a parent for its children (a spool that also holds ground nobody painted, against
 * the two flanges that were). The best ratio is found by Dinkelbach's method from the best
 * single instance's IoU λ: the antichain maximising `Σ (inter − λ·(visible − inter))` is read
 * off the hierarchy bottom-up (an instance, or the best of its children, whichever is more),
 * and its IoU is the next λ, until it stops rising -- a few passes over the instances met.
 *
 * Then the answer is made as simple as it can be at little cost: members that add little are
 * dropped (`gain`), and members that share an ancestor become that ancestor when it is about
 * as good (`slack`), each a share of the IoU; both 0 give the best set itself. Deterministic: the same sums give the
 * same set.
 */
export function bestSet(
  sums: PaintSums,
  { gain = PAINT_SET_GAIN, slack = PAINT_PARENT_SLACK }: { gain?: number; slack?: number } = {},
): PaintSetMatch | null {
  const { inter, visible, parent, painted } = sums;
  const n = sums.ids.length;
  if (n === 0) return null;
  const iouOf = (i: number, v: number): number => {
    const union = painted + v - i;
    return union > 0 ? i / union : 0;
  };
  const order = byDepth(parent);
  // From the best single instance (`bestSingle`'s).
  let members: number[] = [];
  let best = -1;
  for (let k = 0; k < n; k++) {
    const j = iouOf(inter[k] ?? 0, visible[k] ?? 0);
    if (j > best + 1e-9) {
      best = j;
      members = [k];
    }
  }
  const value = new Float64Array(n);
  const below = new Float64Array(n);
  const whole = new Uint8Array(n);
  const open = new Uint8Array(n);
  for (let round = 0; round < SET_ROUNDS; round++) {
    const lambda = best;
    below.fill(0);
    for (let o = 0; o < n; o++) {
      const k = order[o] ?? 0;
      const i = inter[k] ?? 0;
      const own = i - lambda * ((visible[k] ?? 0) - i);
      const parts = below[k] ?? 0;
      // As good as its parts: the instance itself, one member instead of several.
      const self = own >= parts - 1e-12 * (Math.abs(own) + Math.abs(parts));
      whole[k] = self ? 1 : 0;
      const v = self ? own : parts;
      value[k] = v;
      const up = parent[k] ?? -1;
      if (up >= 0 && v > 0) below[up] = (below[up] ?? 0) + v;
    }
    // Read from the top: an instance taken whole, or its children's best.
    const next: number[] = [];
    let i = 0;
    let v = 0;
    for (let o = n - 1; o >= 0; o--) {
      const k = order[o] ?? 0;
      const up = parent[k] ?? -1;
      const reached = (up < 0 || (open[up] === 1 && whole[up] === 0)) && (value[k] ?? 0) > 0;
      open[k] = reached ? 1 : 0;
      if (reached && whole[k] === 1) {
        next.push(k);
        i += inter[k] ?? 0;
        v += visible[k] ?? 0;
      }
    }
    const j = iouOf(i, v);
    if (!(j > best + 1e-12)) break;
    best = j;
    members = next.sort((a, b) => a - b);
  }
  let i = 0;
  let v = 0;
  for (const k of members) {
    i += inter[k] ?? 0;
    v += visible[k] ?? 0;
  }
  // Members that add little go, the least first (as each adds to the best set), while the set
  // stays within `gain` of the best.
  if (members.length > 1) {
    const without = new Float64Array(n);
    for (const k of members) without[k] = iouOf(i - (inter[k] ?? 0), v - (visible[k] ?? 0));
    const least = members
      .filter((k) => (without[k] ?? 0) >= best * (1 - gain))
      .sort((a, b) => (without[b] ?? 0) - (without[a] ?? 0) || a - b);
    const gone = new Set<number>();
    for (const k of least) {
      if (gone.size === members.length - 1) break;
      const i2 = i - (inter[k] ?? 0);
      const v2 = v - (visible[k] ?? 0);
      if (iouOf(i2, v2) < best * (1 - gain)) break;
      gone.add(k);
      i = i2;
      v = v2;
    }
    if (gone.size > 0) members = members.filter((k) => !gone.has(k));
  }
  // Members that share an ancestor become it, the deepest first in one pass, while the set
  // stays within `slack` of what it was. Per ancestor, the members it holds and their sums.
  const floor = iouOf(i, v) * (1 - slack);
  const held = new Int32Array(n);
  const heldInter = new Float64Array(n);
  const heldVisible = new Float64Array(n);
  for (const m of members) {
    for (let up = parent[m] ?? -1; up >= 0; up = parent[up] ?? -1) {
      held[up] = (held[up] ?? 0) + 1;
      heldInter[up] = (heldInter[up] ?? 0) + (inter[m] ?? 0);
      heldVisible[up] = (heldVisible[up] ?? 0) + (visible[m] ?? 0);
    }
  }
  const merged = new Uint8Array(n);
  let merges = 0;
  for (let o = 0; o < n && members.length > 1; o++) {
    const up = order[o] ?? 0;
    if ((held[up] ?? 0) < 2) continue;
    const i2 = i - (heldInter[up] ?? 0) + (inter[up] ?? 0);
    const v2 = v - (heldVisible[up] ?? 0) + (visible[up] ?? 0);
    if (iouOf(i2, v2) < floor) continue;
    merged[up] = 1;
    merges++;
    // What holds it now holds it in place of its members.
    const count = 1 - (held[up] ?? 0);
    const di = (inter[up] ?? 0) - (heldInter[up] ?? 0);
    const dv = (visible[up] ?? 0) - (heldVisible[up] ?? 0);
    for (let above = parent[up] ?? -1; above >= 0; above = parent[above] ?? -1) {
      held[above] = (held[above] ?? 0) + count;
      heldInter[above] = (heldInter[above] ?? 0) + di;
      heldVisible[above] = (heldVisible[above] ?? 0) + dv;
    }
    i = i2;
    v = v2;
  }
  if (merges > 0) {
    // Each member, or the highest ancestor it was merged into.
    const out = new Set<number>();
    for (const m of members) {
      let top = m;
      for (let up = parent[m] ?? -1; up >= 0; up = parent[up] ?? -1) if (merged[up]) top = up;
      out.add(top);
    }
    members = [...out].sort((a, b) => a - b);
  }
  i = 0;
  v = 0;
  for (const k of members) {
    i += inter[k] ?? 0;
    v += visible[k] ?? 0;
  }
  return {
    ids: largestFirst(
      sums,
      members.map((k) => sums.ids[k] ?? 0),
    ),
    iou: iouOf(i, v),
  };
}

/**
 * Members in a combination's order: the largest on screen first, ascending ids among equals.
 * The view does not change while a stroke is painted, so neither does the order.
 */
function largestFirst(sums: PaintSums, ids: readonly number[]): number[] {
  return [...ids].sort((a, b) => sums.visibleOf(b) - sums.visibleOf(a) || a - b);
}

/** `bestSet` of the painted splats of `sample`. */
export function bestSetByIoU(
  doc: Pick<InstancesDoc, "byId">,
  sample: PaintSample,
): PaintSetMatch | null {
  return bestSet(paintSums(doc, sample));
}

/** The IoU of the union of `ids` (disjoint instances, met or not) with the painted area. */
export function setIoU(sums: PaintSums, ids: readonly number[]): number {
  let i = 0;
  let v = 0;
  for (const id of ids) {
    const k = entryIn(sums.ids, id);
    if (k >= 0) i += sums.inter[k] ?? 0;
    v += sums.visibleOf(id);
  }
  const union = sums.painted + v - i;
  return union > 0 ? i / union : 0;
}

/**
 * The match to show while a stroke goes on: `bestSet`, unless what was shown (`shown`, its
 * members) is still within `slack` (a share) of it, which is then kept with its IoU now -- two
 * near-equal answers do not take turns at every preview, and what is lit when the stroke ends
 * is what it selects.
 */
export function steadySet(
  sums: PaintSums,
  shown: readonly number[] | null,
  slack = PAINT_STEADY,
): PaintSetMatch | null {
  const next = bestSet(sums);
  if (!next || !shown || shown.length === 0 || sameMembers(next.ids, shown)) return next;
  const kept = setIoU(sums, shown);
  return kept < next.iou * (1 - slack) ? next : { ids: largestFirst(sums, shown), iou: kept };
}

/** Whether two lists hold the same ids, in any order. */
export function sameMembers(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  const set = new Set(a);
  return b.every((id) => set.has(id));
}

/**
 * The instances above every one of `ids`, the lowest first: the chain of what they are parts
 * of together ("of Spool"). Empty when they share none.
 */
export function commonChain(doc: Pick<InstancesDoc, "byId">, ids: readonly number[]): number[] {
  let common: number[] | null = null;
  for (const id of ids) {
    const above = chainOf(doc, id).slice(1);
    common = common === null ? above : common.filter((up) => above.includes(up));
  }
  return common ?? [];
}

/** At most this many members are named in a combination's name, and in this many characters. */
const COMBINATION_NAMES = 3;
const COMBINATION_CHARS = 48;

/**
 * What a combination is called (its members' `names`, in its order; `whole`, what they are
 * parts of together, or null): "Top flange + drum + Bottom flange (of Spool)", or when that
 * would be too long or says a name twice, "4 parts of Spool" or "4 objects".
 */
export function combinationLabel(names: readonly string[], whole: string | null): string {
  const joined = names.join(" + ");
  if (
    names.length <= COMBINATION_NAMES &&
    new Set(names).size === names.length &&
    joined.length <= COMBINATION_CHARS
  )
    return whole ? `${joined} (of ${whole})` : joined;
  const count = String(names.length);
  return whole ? `${count} parts of ${whole}` : `${count} objects`;
}

/**
 * A view's visible splats by screen cell, to match a painted area while it is painted
 * (`paintSumsIndexed`): `paintSums` over every splat on screen is too slow to run as a stroke
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
  /**
   * Per instance id on a chain on screen, its parent there: the next on the first chain (in
   * leaf id order) that holds it, 0 at the top (`paintSums` reads the hierarchy so too).
   */
  parents: Uint32Array;
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
  // Rolled up each leaf's chain, and each instance's parent on the first chain that holds it.
  const chains = new Map<number, number[]>();
  const rolled = new Float64Array(span);
  const parents = new Uint32Array(span);
  const placed = new Uint8Array(span);
  for (let leaf = 1; leaf < span; leaf++) {
    const w = leaves[leaf] ?? 0;
    if (w === 0) continue;
    const chain = chainOf(doc, leaf);
    if (chain.length === 0) continue;
    chains.set(leaf, chain);
    for (let k = 0; k < chain.length; k++) {
      const up = chain[k] ?? 0;
      rolled[up] = (rolled[up] ?? 0) + w;
      if (placed[up]) continue;
      placed[up] = 1;
      parents[up] = chain[k + 1] ?? 0;
    }
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
    parents,
    chains,
  };
}

/**
 * `paintSums` of the splats under `mask` (the view's brush) through the view's index: the same
 * sums, walking only the painted cells and the instances they hold.
 */
export function paintSumsIndexed(index: PaintIndex, mask: Pick<BrushMask, "data">): PaintSums {
  const { start, ids, weights, cellWeights, visible, parents } = index;
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
  return sumsOf(
    Uint32Array.from(met).sort(),
    (id) => inter[id] ?? 0,
    (id) => (id > 0 && id < span ? (visible[id] ?? 0) : 0),
    (id) => parents[id] ?? 0,
    painted,
  );
}

/** `bestByIoU` of the splats under `mask`, through the view's index. */
export function bestByIoUIndexed(
  index: PaintIndex,
  mask: Pick<BrushMask, "data">,
): PaintMatch | null {
  return bestSingle(paintSumsIndexed(index, mask));
}

/** `bestSetByIoU` of the splats under `mask`, through the view's index. */
export function bestSetByIoUIndexed(
  index: PaintIndex,
  mask: Pick<BrushMask, "data">,
): PaintSetMatch | null {
  return bestSet(paintSumsIndexed(index, mask));
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
