/**
 * Selecting a scan's objects in the scene (docs/SCENE_OBJECTS.md, "Selecting in the scene"):
 * what a click offers, which of it is chosen first, how the choice cycles, and which object a
 * painted area is. Pure; the picking itself is `splatPick.ts`, the wiring
 * `cesium/sceneSelect/`.
 *
 * A click hits splats, and each splat carries one id: its leaf instance. The candidates are
 * that leaf's chain up to the top level (leaf → … → top), then the other instances whose
 * splats the ray also met near its front, each with the part of its own chain not already
 * offered. The first choice is the smallest instance of the main chain that is at least
 * `minPixels` across on screen (a single leaf fragment is rarely what was meant), else the
 * top of the chain.
 *
 * A painted area is matched against every instance, at every level, by intersection over union
 * in splats weighted by opacity, counting only what is visible from the camera (an instance's
 * hidden back does not count against it): the best match is selected, and below
 * `PAINT_MIN_IOU` the painted splats themselves can become an object of their own
 * (`customSets.ts`).
 */

import { categoryById } from "./categories";
import type { Instance, InstancesDoc } from "./instances";

/** A candidate's share of the clicked pixel, below which another instance is not offered. */
export const NEAR_SHARE = 0.15;
/** A candidate must be this close behind the first hit (share of its distance) to be offered. */
export const NEAR_DEPTH = 0.25;
/** The least size on screen (CSS px) for the first choice. */
export const MIN_PIXELS = 48;
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

/**
 * The first choice among `candidates`: the smallest of the main chain at least `minPixels`
 * across on screen (`pixelsOf`), else its top. -1 when there is nothing.
 */
export function defaultIndex(
  candidates: Candidates,
  pixelsOf: (id: number) => number,
  minPixels = MIN_PIXELS,
): number {
  if (candidates.ids.length === 0) return -1;
  const chain = Math.max(1, Math.min(candidates.chain, candidates.ids.length));
  for (let i = 0; i < chain; i++) {
    const id = candidates.ids[i];
    if (id !== undefined && pixelsOf(id) >= minPixels) return i;
  }
  return chain - 1;
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

/** A selection with its place among the candidates, "Tree · 2 of 4": the card's name. */
export function chipText(label: string, index: number, count: number): string {
  return count > 1 ? `${label} · ${String(index + 1)} of ${String(count)}` : label;
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
  let best: PaintMatch | null = null;
  for (const [id, i] of inter) {
    const union = painted + (visible.get(id) ?? 0) - i;
    const iou = union > 0 ? i / union : 0;
    if (!best || iou > best.iou + 1e-9 || (Math.abs(iou - best.iou) <= 1e-9 && id < best.id))
      best = { id, iou };
  }
  return best;
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
