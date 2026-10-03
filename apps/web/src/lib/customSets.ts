/**
 * Objects a viewer painted (`sceneSelect.ts`): an exact set of splats, named and kept per scan
 * in this browser, drawn hidden or highlighted through the same pipeline as the segmentation's
 * instances.
 *
 * A set is stored as the scan's own splat addresses: per tile checksum (the key
 * `instances.json` uses), the splat indices in the tile's own order as `[start, length, …]`
 * ranges. Nothing renderer-specific: every renderer finds a tile's splats by its checksum.
 *
 * Drawn by giving each set an id past the file's (`maxId + 1`, `+ 2`, … in the order the sets
 * were made): `withCustomSets` returns the scan's document with the sets' splats re-labelled
 * to their set's id in the tile runs and the sets appended as top-level instances, so the
 * renderers -- which read ids from the document by checksum -- hide and highlight a set as
 * any instance. A splat in a set no longer carries its segmented id while the set exists.
 */

import type { Instance, InstancesDoc, Vec3 } from "./instances";

export const CUSTOM_SETS_FORMAT = "hexapod.custom-objects";
export const CUSTOM_SETS_VERSION = 1;

/** One painted object. */
export interface CustomSet {
  /** Stable key (made once). */
  key: string;
  name: string;
  /** Per tile checksum, `[start, length, …]` over the tile's own splat order, ascending. */
  tiles: Record<string, number[]>;
  splats: number;
  bounds: { min: Vec3; max: Vec3 };
  /** ms since the epoch. */
  created: number;
}

/** Ascending, de-duplicated indices as `[start, length, …]`. */
export function encodeRanges(indices: Iterable<number>): number[] {
  const sorted = [...new Set(indices)].filter((i) => Number.isInteger(i) && i >= 0);
  sorted.sort((a, b) => a - b);
  const out: number[] = [];
  for (const i of sorted) {
    const last = out.length - 2;
    if (last >= 0 && (out[last] ?? 0) + (out[last + 1] ?? 0) === i)
      out[last + 1] = (out[last + 1] ?? 0) + 1;
    else out.push(i, 1);
  }
  return out;
}

/** The indices `[start, length, …]` covers. */
export function decodeRanges(ranges: readonly number[]): number[] {
  const out: number[] = [];
  for (let k = 0; k + 1 < ranges.length; k += 2) {
    const start = ranges[k] ?? 0;
    const length = ranges[k + 1] ?? 0;
    for (let i = 0; i < length; i++) out.push(start + i);
  }
  return out;
}

/** Splats the ranges cover. */
export function rangesLength(ranges: readonly number[]): number {
  let n = 0;
  for (let k = 1; k < ranges.length; k += 2) n += ranges[k] ?? 0;
  return n;
}

function validRanges(value: unknown): value is number[] {
  if (!Array.isArray(value) || value.length % 2 !== 0) return false;
  let end = -1;
  for (let k = 0; k < value.length; k += 2) {
    const start = value[k] as unknown;
    const length = value[k + 1] as unknown;
    if (!Number.isInteger(start) || !Number.isInteger(length)) return false;
    if ((start as number) <= end || (length as number) < 1) return false;
    end = (start as number) + (length as number) - 1;
  }
  return true;
}

function vec3(value: unknown): Vec3 | null {
  if (!Array.isArray(value) || value.length !== 3) return null;
  if (!value.every((v) => typeof v === "number" && Number.isFinite(v))) return null;
  return [value[0] as number, value[1] as number, value[2] as number];
}

/** The sets as stored. */
export function serializeCustomSets(sets: readonly CustomSet[]): string {
  return JSON.stringify({ format: CUSTOM_SETS_FORMAT, version: CUSTOM_SETS_VERSION, sets });
}

/** Stored sets read back; a malformed set is dropped, a malformed document gives none. */
export function parseCustomSets(text: string | null | undefined): CustomSet[] {
  if (!text) return [];
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return [];
  }
  const doc = raw as { format?: unknown; version?: unknown; sets?: unknown } | null;
  if (doc?.format !== CUSTOM_SETS_FORMAT || doc.version !== CUSTOM_SETS_VERSION) return [];
  if (!Array.isArray(doc.sets)) return [];
  const out: CustomSet[] = [];
  const keys = new Set<string>();
  for (const entry of doc.sets as unknown[]) {
    const s = entry as Record<string, unknown> | null;
    if (typeof s?.key !== "string" || keys.has(s.key) || typeof s.name !== "string") continue;
    if (typeof s.tiles !== "object" || s.tiles === null || Array.isArray(s.tiles)) continue;
    const tiles: Record<string, number[]> = {};
    let splats = 0;
    let ok = true;
    for (const [checksum, ranges] of Object.entries(s.tiles as Record<string, unknown>)) {
      if (!validRanges(ranges)) {
        ok = false;
        break;
      }
      tiles[checksum] = ranges;
      splats += rangesLength(ranges);
    }
    const b = s.bounds as { min?: unknown; max?: unknown } | undefined;
    const min = vec3(b?.min);
    const max = vec3(b?.max);
    if (!ok || splats === 0 || !min || !max) continue;
    keys.add(s.key);
    out.push({
      key: s.key,
      name: s.name,
      tiles,
      splats,
      bounds: { min, max },
      created: typeof s.created === "number" && Number.isFinite(s.created) ? s.created : 0,
    });
  }
  return out;
}

/** Run-length `[id, count, …]` of one label per splat. */
function encodeRuns(ids: Uint32Array): Int32Array {
  const out: number[] = [];
  for (let i = 0; i < ids.length;) {
    const id = ids[i] ?? 0;
    let j = i + 1;
    while (j < ids.length && ids[j] === id) j++;
    out.push(id, j - i);
    i = j;
  }
  return Int32Array.from(out);
}

function decodeRunsTo(runs: Int32Array): Uint32Array {
  let n = 0;
  for (let k = 1; k < runs.length; k += 2) n += runs[k] ?? 0;
  const out = new Uint32Array(n);
  let at = 0;
  for (let k = 0; k + 1 < runs.length; k += 2) {
    const count = runs[k + 1] ?? 0;
    out.fill(runs[k] ?? 0, at, at + count);
    at += count;
  }
  return out;
}

/** The id set `index` (0-based, in the order given) is drawn under in `doc`. */
export function customId(doc: Pick<InstancesDoc, "maxId">, index: number): number {
  return doc.maxId + 1 + index;
}

/** Whether `id` is a painted object's in a document from `withCustomSets(doc, …)`. */
export function isCustomId(base: Pick<InstancesDoc, "maxId">, id: number): boolean {
  return id > base.maxId;
}

const MERGED = new WeakMap<InstancesDoc, { sets: readonly CustomSet[]; doc: InstancesDoc }>();

/**
 * `doc` with `sets` drawn as instances of their own (see the module comment). The same
 * document when there are none; the same result for the same `sets` array (memoised), so a
 * renderer that keys on the document's identity rebinds only when the sets change.
 */
export function withCustomSets(doc: InstancesDoc, sets: readonly CustomSet[]): InstancesDoc {
  if (sets.length === 0) return doc;
  const cached = MERGED.get(doc);
  if (cached?.sets === sets) return cached.doc;
  const tiles = new Map(doc.tiles);
  const instances: Instance[] = [...doc.instances];
  const byId = new Map(doc.byId);
  const relabelled = new Map<string, Uint32Array>();
  sets.forEach((set, k) => {
    const id = customId(doc, k);
    const { min, max } = set.bounds;
    const instance: Instance = {
      id,
      parent: null,
      level: 0,
      splats: set.splats,
      bounds: { min, max },
      centroid: [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2],
      tags: [{ label: set.name, score: 1 }],
      properties: {},
      behaviour: "static",
      views: 0,
    };
    instances.push(instance);
    byId.set(id, instance);
    for (const [checksum, ranges] of Object.entries(set.tiles)) {
      let ids = relabelled.get(checksum);
      if (!ids) {
        const runs = doc.tiles.get(checksum);
        if (!runs) continue;
        ids = decodeRunsTo(runs);
        relabelled.set(checksum, ids);
      }
      for (let r = 0; r + 1 < ranges.length; r += 2) {
        const start = ranges[r] ?? 0;
        const end = Math.min(ids.length, start + (ranges[r + 1] ?? 0));
        if (start < end) ids.fill(id, start, end);
      }
    }
  });
  for (const [checksum, ids] of relabelled) tiles.set(checksum, encodeRuns(ids));
  const merged: InstancesDoc = {
    ...doc,
    instances,
    byId,
    tiles,
    maxId: doc.maxId + sets.length,
  };
  MERGED.set(doc, { sets, doc: merged });
  return merged;
}

/** Where a scan's painted objects are kept in this browser. */
export function customSetsStorageKey(assetId: string): string {
  return `hexapod.customObjects.${assetId}`;
}

/** The scan's painted objects in this browser; none when storage is blocked or empty. */
export function loadCustomSets(assetId: string): CustomSet[] {
  try {
    return parseCustomSets(globalThis.localStorage?.getItem(customSetsStorageKey(assetId)));
  } catch {
    return [];
  }
}

/** Keeps the scan's painted objects in this browser; false when storage is blocked. */
export function saveCustomSets(assetId: string, sets: readonly CustomSet[]): boolean {
  try {
    const storage = globalThis.localStorage;
    if (sets.length === 0) storage.removeItem(customSetsStorageKey(assetId));
    else storage.setItem(customSetsStorageKey(assetId), serializeCustomSets(sets));
    return true;
  } catch {
    return false;
  }
}
