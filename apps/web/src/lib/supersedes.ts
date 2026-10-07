/**
 * The measured splats a fill method replaces ("swap, don't stack"): a fill variant may name a
 * `supersedes` file (lib/variants.ts) listing the scan's splats its rebuilt surface stands in
 * for -- a thin, see-through patch of the scan under an opaque one the method made. While that
 * method is drawn they are hidden, so the scan and the fill are not drawn over each other.
 *
 *     { "superseded": 1234, "tiles": { "<checksum>": [flag, count, flag, count, ...] } }
 *
 * The addressing is `instances.json`'s (packages/world/src/tileRuns.ts): per tile checksum,
 * run-length over the tile's own splat order, flag 1 for a superseded splat and 0 for one kept;
 * coarse tiles carry their own runs, so every level of detail hides the same patch.
 *
 * Drawn as `lib/customSets.ts` draws a painted object: `withSupersedes` re-labels the
 * superseded splats to one reserved id past the document's largest, and every place that
 * honours the hidden set adds that id to it (`withSuperseded`) -- the renderers' state texels,
 * picking and the brush. The user's own hidden set (state/instances.ts) is never touched, so
 * undo, the objects panel's counts and Hide all stay as they were. No new shader code.
 */

import { runsLength, tileRunsIssue } from "@twin/world";

import { resolveBeside, type InstancesDoc } from "./instances";

export interface SupersedesDoc {
  /** Splats flagged, as the file says (for logs). */
  superseded: number;
  /** Per tile checksum, run-length `[flag, count, …]`; tiles without a flagged splat dropped. */
  tiles: ReadonlyMap<string, Int32Array>;
  /** What was skipped while reading. */
  issues: string[];
}

/** Reads a `supersedes.json`; null when it is not one (no `tiles` object). */
export function parseSupersedes(raw: unknown): SupersedesDoc | null {
  const doc = raw as { superseded?: unknown; tiles?: unknown } | null | undefined;
  if (typeof doc?.tiles !== "object" || doc.tiles === null || Array.isArray(doc.tiles)) {
    return null;
  }
  const tiles = new Map<string, Int32Array>();
  const issues: string[] = [];
  for (const [key, value] of Object.entries(doc.tiles as Record<string, unknown>)) {
    const issue = tileRunsIssue(key, value, 1, "1 (superseded)");
    if (issue !== undefined) {
      issues.push(issue);
      continue;
    }
    const runs = value as number[];
    let flagged = false;
    for (let k = 0; k < runs.length && !flagged; k += 2) flagged = runs[k] === 1;
    if (flagged) tiles.set(key, Int32Array.from(runs));
  }
  const superseded =
    typeof doc.superseded === "number" && Number.isFinite(doc.superseded) ? doc.superseded : 0;
  return { superseded, tiles, issues };
}

/** Fetches and reads a fill's `supersedes.json` beside the measured tileset. */
export async function loadSupersedes(tilesetUrl: string, uri: string): Promise<SupersedesDoc> {
  const response = await fetch(resolveBeside(tilesetUrl, uri));
  if (!response.ok) throw new Error(`supersedes answered ${String(response.status)}`);
  const doc = parseSupersedes(await response.json());
  if (!doc) throw new Error("supersedes: no tiles");
  return doc;
}

/** Documents from `withSupersedes`, and the id their superseded splats carry. */
const RESERVED = new WeakMap<InstancesDoc, number>();
const MERGED = new WeakMap<InstancesDoc, { sup: SupersedesDoc; doc: InstancesDoc }>();

/**
 * `doc` with `sup`'s splats re-labelled to one reserved id, `doc.maxId + 1` (past any painted
 * object's too, when `doc` already carries them). The same document when `sup` is null or
 * flags nothing it lists; the same result for the same pair (memoised), so a renderer keyed on
 * the document's identity rebinds only when the swap changes. A tile `doc` does not list gets
 * runs of its own; one whose splat count disagrees with `doc`'s is left as it is.
 */
export function withSupersedes(doc: InstancesDoc, sup: SupersedesDoc | null): InstancesDoc {
  if (sup === null || sup.tiles.size === 0) return doc;
  const cached = MERGED.get(doc);
  if (cached?.sup === sup) return cached.doc;
  const id = doc.maxId + 1;
  const tiles = new Map(doc.tiles);
  for (const [checksum, flags] of sup.tiles) {
    const n = runsLength(flags);
    const runs = doc.tiles.get(checksum);
    if (runs !== undefined && runsLength(runs) !== n) continue;
    tiles.set(checksum, relabel(runs ?? Int32Array.of(0, n), flags, id));
  }
  const merged: InstancesDoc = { ...doc, tiles, maxId: id };
  RESERVED.set(merged, id);
  MERGED.set(doc, { sup, doc: merged });
  return merged;
}

/** `runs` (ids) with every splat `flags` marks 1 carrying `id`, run-length again. */
function relabel(runs: Int32Array, flags: Int32Array, id: number): Int32Array {
  const out: number[] = [];
  const push = (label: number, count: number): void => {
    const last = out.length - 2;
    if (last >= 0 && out[last] === label) out[last + 1] = (out[last + 1] ?? 0) + count;
    else out.push(label, count);
  };
  // Walk both run lists at once: each step the shorter of the two current runs.
  let r = 0;
  let f = 0;
  let leftR = runs[1] ?? 0;
  let leftF = flags[1] ?? 0;
  while (r < runs.length && f < flags.length) {
    const step = Math.min(leftR, leftF);
    if (step > 0) push(flags[f] === 1 ? id : (runs[r] ?? 0), step);
    leftR -= step;
    leftF -= step;
    if (leftR <= 0) {
      r += 2;
      leftR = runs[r + 1] ?? 0;
    }
    if (leftF <= 0) {
      f += 2;
      leftF = flags[f + 1] ?? 0;
    }
  }
  return Int32Array.from(out);
}

/** The id `doc`'s superseded splats carry, or null when it hides none. */
export function supersededIdOf(doc: InstancesDoc | undefined): number | null {
  return doc ? (RESERVED.get(doc) ?? null) : null;
}

/**
 * `hidden` with `doc`'s superseded splats' id: what a renderer, picking and the brush hide.
 * The same set when `doc` supersedes nothing.
 */
export function withSuperseded(
  doc: InstancesDoc | undefined,
  hidden: ReadonlySet<number>,
): ReadonlySet<number> {
  const id = supersededIdOf(doc);
  if (id === null || hidden.has(id)) return hidden;
  return new Set([...hidden, id]);
}
