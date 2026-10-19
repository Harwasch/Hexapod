/**
 * Scene instances (hexapod.instances v1, docs/SCENE_OBJECTS.md §4): which object each splat
 * of a scan belongs to, and what each object is, written by the segmentation step beside the
 * measured tiles and declared on their root as `extras.instances = { uri, count }`.
 *
 * Nothing here knows a class. An instance carries open-vocabulary `tags` (label + score,
 * descending) and a short set of numeric `properties` (movable, vegetation, ...) whose names
 * come from the file; search ranks instances by how well a query matches their tags (and
 * property names), and filters by property values, so a new vocabulary or a new property
 * needs no change here.
 *
 * Per splat, `tiles` maps each tile checksum to run-length `[id, count, ...]` pairs in the
 * tile's own gaussian order, exactly as `plants.json` does (`tileRuns.ts` in `@twin/world`);
 * id 0 is "no instance". Ids are leaf-level; the hierarchy is walked through `parent`.
 *
 * `instances.emb` (float16, `count × dim`, row `k` is id `k + 1`, L2-normalised) is each
 * object's image embedding. "More like this" reads it (`loadEmbeddings`, `describedMatrix`,
 * `rankSimilar`): the objects whose views look like a chosen one's. Text search by meaning
 * will read it too once there is a text encoder at runtime: `rankByEmbedding` is that seam.
 * v1 search is by tags.
 */

import { decodeRuns, runsLength, tileRunsIssue } from "@twin/world";

import { jsonBytes, scanPayloads } from "./payloadCache";

export type Vec3 = readonly [number, number, number];

export const INSTANCES_FORMAT = "hexapod.instances";
export const INSTANCES_VERSION = 1;

/** `root.extras.instances`. */
export interface InstancesRef {
  uri: string;
  count: number;
}

export type Behaviour = "static" | "in-place" | "movable";

export interface InstanceTag {
  label: string;
  score: number;
}

export interface Instance {
  /** 1-based; 0 means none. */
  id: number;
  parent: number | null;
  level: number;
  splats: number;
  bounds: { min: Vec3; max: Vec3 };
  centroid: Vec3;
  /** Top-k, descending by score. */
  tags: InstanceTag[];
  /** Attribute scores, 0..1, named by the file. */
  properties: Record<string, number>;
  behaviour: Behaviour;
  views: number;
  /** Its broad scene category (`lib/categories.ts`), when the file says; newer runs do. */
  category?: string;
  /**
   * What it is called, when the file names it: a vision-language model's answer, or a ground
   * cover class ("Grass", "Hay"). The panel, the selection card and search use it before
   * any tag.
   */
  name?: string;
  /**
   * What the file says it is, when it says (docs/SCENE_OBJECTS.md §3b): "thing", or "ground"
   * (a ground cover class or a region of one). The brush reads it (lib/sceneSelect.ts
   * `topLevels`).
   */
  kind?: string;
}

export interface EmbeddingRef {
  file: string;
  model: string;
  dim: number;
  dtype: "float16";
}

export interface InstancesDoc {
  instances: Instance[];
  /** By id. */
  byId: ReadonlyMap<number, Instance>;
  /** The largest id; the per-id GPU state is sized by it. */
  maxId: number;
  /** Per tile checksum, run-length `[id, count, ...]`. Malformed tiles are dropped. */
  tiles: ReadonlyMap<string, Int32Array>;
  embedding: EmbeddingRef | null;
  /** Every property name any instance carries, sorted. */
  propertyNames: string[];
  /** Problems found while reading; the document is still usable. */
  issues: string[];
}

function finite(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function vec3(value: unknown): Vec3 | null {
  if (!Array.isArray(value) || value.length !== 3) return null;
  const [x, y, z] = value as unknown[];
  if (![x, y, z].every((v) => typeof v === "number" && Number.isFinite(v))) return null;
  return [x as number, y as number, z as number];
}

/** `extras.instances` off a tileset's root tile; null when absent or malformed. */
export function instancesRefOf(extras: unknown): InstancesRef | null {
  const ref = (extras as { instances?: Record<string, unknown> } | null | undefined)?.instances;
  if (typeof ref?.uri !== "string" || ref.uri.length === 0) return null;
  return { uri: ref.uri, count: Math.max(0, Math.round(finite(ref.count))) };
}

const BEHAVIOURS = new Set<Behaviour>(["static", "in-place", "movable"]);

function instanceOf(raw: unknown): Instance | null {
  const r = raw as Record<string, unknown> | null | undefined;
  if (typeof r !== "object" || r === null) return null;
  const id = r.id;
  if (!Number.isInteger(id) || (id as number) < 1) return null;
  const b = r.bounds as { min?: unknown; max?: unknown } | undefined;
  const min = vec3(b?.min);
  const max = vec3(b?.max);
  if (!min || !max) return null;
  const centroid = vec3(r.centroid) ?? [
    (min[0] + max[0]) / 2,
    (min[1] + max[1]) / 2,
    (min[2] + max[2]) / 2,
  ];
  const tags: InstanceTag[] = [];
  if (Array.isArray(r.tags)) {
    for (const t of r.tags as { label?: unknown; score?: unknown }[]) {
      if (typeof t?.label !== "string" || t.label.trim().length === 0) continue;
      tags.push({ label: t.label.trim(), score: finite(t.score) });
    }
  }
  tags.sort((a, b2) => b2.score - a.score);
  const properties: Record<string, number> = {};
  if (typeof r.properties === "object" && r.properties !== null) {
    for (const [name, value] of Object.entries(r.properties as Record<string, unknown>)) {
      if (typeof value === "number" && Number.isFinite(value)) properties[name] = value;
    }
  }
  const parent =
    Number.isInteger(r.parent) && (r.parent as number) >= 1 ? (r.parent as number) : null;
  return {
    id: id as number,
    parent: parent === id ? null : parent,
    level: Math.max(0, Math.round(finite(r.level))),
    splats: Math.max(0, Math.round(finite(r.splats))),
    bounds: { min, max },
    centroid,
    tags,
    properties,
    behaviour: BEHAVIOURS.has(r.behaviour as Behaviour) ? (r.behaviour as Behaviour) : "static",
    views: Math.max(0, Math.round(finite(r.views))),
    ...(typeof r.category === "string" && r.category !== "" ? { category: r.category } : {}),
    ...(typeof r.name === "string" && r.name.trim() !== "" ? { name: r.name.trim() } : {}),
    ...(typeof r.kind === "string" && r.kind !== "" ? { kind: r.kind } : {}),
  };
}

/**
 * Reads an `instances.json` document. Null when it is not one (wrong format or version, no
 * instance list); otherwise every well-formed instance and tile, with what was skipped listed
 * in `issues`. Defensive, like `evidenceOf`: a malformed entry costs that entry, not the file.
 */
export function parseInstances(raw: unknown): InstancesDoc | null {
  const doc = raw as Record<string, unknown> | null | undefined;
  if (doc?.format !== INSTANCES_FORMAT || doc.version !== INSTANCES_VERSION) return null;
  if (!Array.isArray(doc.instances)) return null;
  const issues: string[] = [];
  const instances: Instance[] = [];
  const byId = new Map<number, Instance>();
  for (const entry of doc.instances as unknown[]) {
    const instance = instanceOf(entry);
    if (!instance) {
      issues.push("an instance without an integer id >= 1 and bounds was skipped");
      continue;
    }
    if (byId.has(instance.id)) {
      issues.push(`instance ${String(instance.id)} is listed twice; the first is kept`);
      continue;
    }
    byId.set(instance.id, instance);
    instances.push(instance);
  }
  for (const instance of instances) {
    if (instance.parent !== null && !byId.has(instance.parent)) {
      issues.push(
        `instance ${String(instance.id)}'s parent ${String(instance.parent)} is not listed`,
      );
      instance.parent = null;
    }
  }
  let maxId = 0;
  for (const instance of instances) maxId = Math.max(maxId, instance.id);
  const tiles = new Map<string, Int32Array>();
  if (typeof doc.tiles === "object" && doc.tiles !== null && !Array.isArray(doc.tiles)) {
    for (const [key, value] of Object.entries(doc.tiles as Record<string, unknown>)) {
      const issue = tileRunsIssue(key, value, maxId, "a listed instance");
      if (issue !== undefined) {
        issues.push(issue);
        continue;
      }
      tiles.set(key, Int32Array.from(value as number[]));
    }
  } else {
    issues.push("tiles must be an object keyed by tile checksum");
  }
  const e = doc.embedding as Record<string, unknown> | null | undefined;
  const embedding: EmbeddingRef | null =
    typeof e?.file === "string" && Number.isInteger(e.dim) && (e.dim as number) > 0
      ? {
          file: e.file,
          model: typeof e.model === "string" ? e.model : "unknown",
          dim: e.dim as number,
          dtype: "float16",
        }
      : null;
  const names = new Set<string>();
  for (const instance of instances)
    for (const name of Object.keys(instance.properties)) names.add(name);
  return {
    instances,
    byId,
    maxId,
    tiles,
    embedding,
    propertyNames: [...names].sort(),
    issues,
  };
}

/** The instance id of every gaussian of the tile whose checksum is `checksum`, or undefined. */
export function tileInstanceIds(doc: InstancesDoc, checksum: string): Uint32Array | undefined {
  const runs = doc.tiles.get(checksum);
  if (runs === undefined) return undefined;
  return decodeRuns(runs, new Uint32Array(runsLength(runs)));
}

/** `uri` beside the tileset, keeping its query (a signed URL's). */
export function resolveBeside(tilesetUrl: string, uri: string): string {
  const parent = new URL(tilesetUrl, globalThis.location?.href ?? "http://localhost/");
  const resolved = new URL(uri, parent);
  if (!resolved.search && resolved.origin === parent.origin) resolved.search = parent.search;
  return resolved.href;
}

/**
 * Fetches and reads a scan's `instances.json`. Throws when it is missing or not one. Kept in
 * memory once read (lib/payloadCache.ts): picking an objects method again is instant.
 */
export function loadInstances(tilesetUrl: string, ref: InstancesRef): Promise<InstancesDoc> {
  const url = resolveBeside(tilesetUrl, ref.uri);
  return scanPayloads.get(url, async () => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`instances answered ${String(response.status)}`);
    const text = await response.text();
    const doc = parseInstances(JSON.parse(text));
    if (!doc) throw new Error("instances: not a hexapod.instances v1 document");
    return { value: doc, bytes: jsonBytes(text) };
  });
}

// ---- Hierarchy ---------------------------------------------------------------------------

/**
 * `ids` and every instance below them. Splats carry leaf ids, so hiding a coarse instance
 * means hiding the leaves it contains.
 */
export function withDescendants(
  doc: Pick<InstancesDoc, "instances">,
  ids: Iterable<number>,
): Set<number> {
  const out = new Set(ids);
  if (out.size === 0) return out;
  const children = new Map<number, number[]>();
  for (const instance of doc.instances) {
    if (instance.parent === null) continue;
    const list = children.get(instance.parent) ?? [];
    list.push(instance.id);
    children.set(instance.parent, list);
  }
  const stack = [...out];
  for (let id = stack.pop(); id !== undefined; id = stack.pop()) {
    for (const child of children.get(id) ?? []) {
      if (out.has(child)) continue;
      out.add(child);
      stack.push(child);
    }
  }
  return out;
}

// ---- Search ------------------------------------------------------------------------------

export type FilterOp = ">" | ">=" | "<" | "<=" | "=";

/** `name op value` on a property, or `behaviour = value`. */
export interface InstanceFilter {
  name: string;
  op: FilterOp;
  value: number | string;
}

export interface InstanceQuery {
  /** Lower-case words to match against tags. */
  terms: string[];
  filters: InstanceFilter[];
}

export interface SearchResult {
  id: number;
  /** 0..1: how well the query matched, times the matched tag's own score. */
  score: number;
  /** The tag (or property) that matched, or the top tag when only filters were given. */
  label: string;
  behaviour: Behaviour;
  /** Gaussians the instance holds: tells apart rows that share a label. */
  splats: number;
}

/** The default number of results a search lists; the whole match set is kept beside them. */
export const RESULT_LIMIT = 50;

const FILTER = /([a-z_][\w-]*)\s*(>=|<=|>|<|=|:)\s*([\w.-]+)/gi;

function tokens(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^\p{L}\p{N}]+/u)
    .filter((t) => t.length > 0);
}

/**
 * Splits a typed query into words and filters: `vegetation > 0.5`, `movable>=0.3`,
 * `behaviour:movable` (`:` is `=`). Whatever is not a filter is words.
 */
export function parseQuery(text: string): InstanceQuery {
  const filters: InstanceFilter[] = [];
  const rest = text.replace(FILTER, (whole, name: string, op: string, value: string) => {
    const number = Number(value);
    const key = name.toLowerCase();
    const isBehaviour = key === "behaviour" || key === "behavior";
    if (!isBehaviour && !Number.isFinite(number)) return whole;
    filters.push({
      name: isBehaviour ? "behaviour" : key,
      op: op === ":" ? "=" : (op as FilterOp),
      value: isBehaviour ? value.toLowerCase() : number,
    });
    return " ";
  });
  return { terms: tokens(rest), filters };
}

/** Whether an instance passes a filter. A property it does not carry fails. */
export function passes(instance: Instance, filter: InstanceFilter): boolean {
  if (filter.name === "behaviour") {
    return filter.op === "=" && instance.behaviour === filter.value;
  }
  const v = instance.properties[filter.name];
  const want = filter.value;
  if (v === undefined || typeof want !== "number") return false;
  switch (filter.op) {
    case ">":
      return v > want;
    case ">=":
      return v >= want;
    case "<":
      return v < want;
    case "<=":
      return v <= want;
    case "=":
      return Math.abs(v - want) < 1e-9;
  }
}

/**
 * How well query words match a label, 0..1: the whole phrase is 1; otherwise each word scores
 * 1 for an equal token, 0.75 when it begins one, 0.5 when it is inside the label, and the
 * words' mean is taken (so every word must count).
 */
export function matchLabel(terms: readonly string[], label: string): number {
  if (terms.length === 0) return 0;
  const lower = label.toLowerCase();
  const labelTokens = tokens(lower);
  if (labelTokens.join(" ") === terms.join(" ")) return 1;
  let sum = 0;
  for (const term of terms) {
    let best = 0;
    for (const t of labelTokens) {
      if (t === term) best = 1;
      else if (term.length >= 2 && t.startsWith(term)) best = Math.max(best, 0.75);
      else if (term.length >= 3 && t.includes(term)) best = Math.max(best, 0.5);
    }
    if (best === 0 && term.length >= 3 && lower.includes(term)) best = 0.5;
    sum += best;
  }
  // Short of the whole phrase, a perfect word match stays below an exact label.
  return (sum / terms.length) * 0.95;
}

/** The label a search result goes by: its name, else its top tag. */
export function instanceLabel(instance: Instance): string {
  return instance.name ?? instance.tags[0]?.label ?? "untagged";
}

/** A splat count, short: 940, 12.3k, 1.2M. */
export function formatSplats(splats: number): string {
  if (splats < 1000) return String(splats);
  if (splats < 1_000_000) return `${(splats / 1000).toFixed(splats < 10_000 ? 1 : 0)}k`;
  return `${(splats / 1_000_000).toFixed(1)}M`;
}

/**
 * How much of the scan an instance is, 0..1: its splats on a log scale against the largest
 * instance's. A query's match is scaled by `PROMINENCE_FLOOR + (1 - PROMINENCE_FLOOR) *
 * prominence`, so a fragment of a few hundred splats that happens to match well does not bury
 * the object itself (measured on the spool scan, where fragments took the top hits).
 */
export const PROMINENCE_FLOOR = 0.6;

export function prominence(splats: number, largest: number): number {
  if (largest <= 0 || splats <= 0) return 0;
  return Math.min(1, Math.log1p(splats) / Math.log1p(largest));
}

/**
 * Ranks instances for a query. Words match each tag's label (`matchLabel`) weighted by the
 * tag's own score, and property names weighted by the property's value — so "vegetation"
 * finds what scored as vegetation without a class list. Filters must all pass. With filters
 * and no words, every passing instance is returned, ordered by the first numeric filter's
 * property. Ties go to the larger instance.
 */
export function searchInstances(
  instances: readonly Instance[],
  query: InstanceQuery | string,
  limit = RESULT_LIMIT,
): SearchResult[] {
  const q = typeof query === "string" ? parseQuery(query) : query;
  if (q.terms.length === 0 && q.filters.length === 0) return [];
  const results: SearchResult[] = [];
  const sortBy = q.filters.find((f) => f.name !== "behaviour")?.name;
  const largest = instances.reduce((m, i) => Math.max(m, i.splats), 0);
  for (const instance of instances) {
    if (!q.filters.every((f) => passes(instance, f))) continue;
    let score = 0;
    let label = instanceLabel(instance);
    if (q.terms.length > 0) {
      if (instance.name !== undefined) score = matchLabel(q.terms, instance.name);
      for (const tag of instance.tags) {
        const s = matchLabel(q.terms, tag.label) * Math.max(0, Math.min(1, tag.score));
        if (s > score) {
          score = s;
          label = tag.label;
        }
      }
      for (const [name, value] of Object.entries(instance.properties)) {
        const s = matchLabel(q.terms, name) * Math.max(0, Math.min(1, value));
        if (s > score) {
          score = s;
          label = `${name} ${value.toFixed(2)}`;
        }
      }
      if (score <= 0) continue;
      score *= PROMINENCE_FLOOR + (1 - PROMINENCE_FLOOR) * prominence(instance.splats, largest);
    } else {
      score = sortBy === undefined ? 1 : Math.max(0, Math.min(1, instance.properties[sortBy] ?? 0));
    }
    results.push({
      id: instance.id,
      score,
      label,
      behaviour: instance.behaviour,
      splats: instance.splats,
    });
  }
  results.sort((a, b) => b.score - a.score || b.splats - a.splats || a.id - b.id);
  return Number.isFinite(limit) ? results.slice(0, limit) : results;
}

// ---- Search by meaning (the seam) --------------------------------------------------------

/** IEEE 754 half to number. */
export function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exponent = (h >> 10) & 0x1f;
  const fraction = h & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

/**
 * Ranks instances by cosine similarity between a query embedding and `instances.emb` (raw
 * float16 bits, `count × dim`, row `k` is id `k + 1`, rows L2-normalised). `queryVec` must
 * come from the text tower of the model `embedding.model` names, which v1 does not ship in
 * the browser; this is where it plugs in.
 */
export function rankByEmbedding(
  queryVec: Float32Array,
  embeddings: Uint16Array,
  dim: number,
  limit = 50,
): { id: number; score: number }[] {
  if (queryVec.length !== dim || dim <= 0) throw new Error("rankByEmbedding: dimension mismatch");
  let norm = 0;
  for (const v of queryVec) norm += v * v;
  const inv = norm > 0 ? 1 / Math.sqrt(norm) : 0;
  const count = Math.floor(embeddings.length / dim);
  const out: { id: number; score: number }[] = [];
  for (let k = 0; k < count; k += 1) {
    let dot = 0;
    for (let j = 0; j < dim; j += 1)
      dot += halfToFloat(embeddings[k * dim + j] ?? 0) * (queryVec[j] ?? 0);
    out.push({ id: k + 1, score: dot * inv });
  }
  out.sort((a, b) => b.score - a.score);
  return out.slice(0, limit);
}

// ---- More like this ----------------------------------------------------------------------

/** Every half's value, made once: a scan's file is tens of millions of halves. */
let HALVES: Float32Array | null = null;

function halfTable(): Float32Array {
  if (HALVES) return HALVES;
  const table = new Float32Array(65536);
  for (let h = 0; h < 65536; h += 1) table[h] = halfToFloat(h);
  HALVES = table;
  return table;
}

/** "43.6 MB": a download's size, decimal as a browser's network panel says it. */
export function formatBytes(bytes: number): string {
  if (bytes < 1e6) return `${Math.max(1, Math.round(bytes / 1e3)).toLocaleString()} kB`;
  return `${(bytes / 1e6).toFixed(1)} MB`;
}

/**
 * The body of `response`, read into one buffer of the `expected` size as it arrives (the camp's
 * file is 43.6 MB, so no second copy is made), with `onProgress` told the bytes received.
 * A body of another size is read to its end and returned at its own length, for the caller to
 * refuse.
 */
async function readBody(
  response: Response,
  expected: number,
  onProgress?: (received: number, total: number) => void,
): Promise<Uint8Array> {
  const reader = response.body?.getReader();
  if (!reader) return new Uint8Array(await response.arrayBuffer());
  const out = new Uint8Array(expected);
  const extra: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    const fits = Math.max(0, Math.min(value.length, expected - received));
    out.set(value.subarray(0, fits), received);
    if (fits < value.length) extra.push(value.subarray(fits));
    received += value.length;
    onProgress?.(received, expected);
  }
  if (received === expected) return out;
  if (received < expected) return out.subarray(0, received);
  // Longer than declared: whole, so the size it reports is its own.
  const whole = new Uint8Array(received);
  whole.set(out);
  let at = expected;
  for (const part of extra) {
    whole.set(part, at);
    at += part.length;
  }
  return whole;
}

/**
 * Fetches a scan's `instances.emb`: `count` rows of `dim` float16s, the raw bits. Throws when
 * it is missing or is not that size (a file of another scan or another model). Kept in memory
 * once read (lib/payloadCache.ts), so a second "More like this" on the scan is instant.
 */
export function loadEmbeddings(
  url: string,
  count: number,
  dim: number,
  onProgress?: (received: number, total: number) => void,
): Promise<Uint16Array> {
  const expected = count * dim * 2;
  return scanPayloads.get(url, async () => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`object signatures answered ${String(response.status)}`);
    const bytes = await readBody(response, expected, onProgress);
    if (bytes.byteLength !== expected)
      throw new Error(
        `object signatures are ${String(bytes.byteLength)} bytes, not ${String(expected)} ` +
          `(${String(count)} objects × ${String(dim)} values)`,
      );
    return {
      value: new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2),
      bytes: bytes.byteLength,
    };
  });
}

/**
 * A row this long or longer is a description; an object too small in every view for a useful
 * crop has a zero row (docs/SCENE_OBJECTS.md §2 step 4), and the file's rows are otherwise unit.
 */
export const DESCRIBED_MIN_NORM = 0.5;

/** The described rows of a scan's embeddings, centred: what "More like this" compares. */
export interface DescribedMatrix {
  dim: number;
  /** The instance id of each row of `m`. */
  ids: Int32Array;
  /** The row of `m` of each described instance id. */
  rowOf: ReadonlyMap<number, number>;
  /** `ids.length × dim`: each described row less their mean, then made unit length again. */
  m: Float32Array;
}

const MATRICES = new WeakMap<Uint16Array, DescribedMatrix>();

/**
 * The described rows of `raw` (row `k` is id `k + 1`), less their mean and made unit length
 * again. Raw SigLIP cosines between a scan's objects all sit between about 0.83 and 1.0 -- every
 * crop shares the scan's light, lens and ground -- so they cannot tell objects apart; taking
 * away what every object shares spreads them to about -0.5..1 (measured on the camp, the
 * pumpkin and the spool, 2026-10-16). Made once per file.
 */
export function describedMatrix(raw: Uint16Array, dim: number): DescribedMatrix {
  const kept = MATRICES.get(raw);
  if (kept?.dim === dim) return kept;
  if (!Number.isInteger(dim) || dim <= 0) throw new Error("describedMatrix: dimension must be > 0");
  const table = halfTable();
  const count = Math.floor(raw.length / dim);
  const rows: number[] = [];
  const minSquared = DESCRIBED_MIN_NORM * DESCRIBED_MIN_NORM;
  for (let k = 0; k < count; k += 1) {
    let sum = 0;
    for (let j = 0, at = k * dim; j < dim; j += 1, at += 1) {
      const v = table[raw[at] ?? 0] ?? 0;
      sum += v * v;
    }
    // NaN (a corrupt half) fails this too.
    if (sum >= minSquared) rows.push(k);
  }
  const m = new Float32Array(rows.length * dim);
  const mean = new Float64Array(dim);
  rows.forEach((k, r) => {
    for (let j = 0; j < dim; j += 1) {
      const v = table[raw[k * dim + j] ?? 0] ?? 0;
      m[r * dim + j] = v;
      mean[j] = (mean[j] ?? 0) + v;
    }
  });
  if (rows.length > 0) for (let j = 0; j < dim; j += 1) mean[j] = (mean[j] ?? 0) / rows.length;
  const ids = new Int32Array(rows.length);
  const rowOf = new Map<number, number>();
  rows.forEach((k, r) => {
    let sum = 0;
    for (let j = 0; j < dim; j += 1) {
      const v = (m[r * dim + j] ?? 0) - (mean[j] ?? 0);
      m[r * dim + j] = v;
      sum += v * v;
    }
    // One described object alone centres to nothing: it stays zero, alike to nothing.
    const scale = sum > 0 ? 1 / Math.sqrt(sum) : 0;
    for (let j = 0; j < dim; j += 1) m[r * dim + j] = (m[r * dim + j] ?? 0) * scale;
    ids[r] = k + 1;
    rowOf.set(k + 1, r);
  });
  const matrix: DescribedMatrix = { dim, ids, rowOf, m };
  MATRICES.set(raw, matrix);
  return matrix;
}

/**
 * The cosine between `query` and every row of `m` (`rows × dim`, rows unit length), by row.
 * Over a matrix already decoded, unlike `rankByEmbedding`: what a chosen object or, later, a
 * text query is compared with.
 */
export function rankByVector(query: Float32Array, m: Float32Array, dim: number): Float32Array {
  if (query.length !== dim || dim <= 0) throw new Error("rankByVector: dimension mismatch");
  let norm = 0;
  for (const v of query) norm += v * v;
  const inv = norm > 0 ? 1 / Math.sqrt(norm) : 0;
  const rows = Math.floor(m.length / dim);
  const out = new Float32Array(rows);
  for (let r = 0; r < rows; r += 1) {
    let dot = 0;
    for (let j = 0, at = r * dim; j < dim; j += 1, at += 1) dot += (m[at] ?? 0) * (query[j] ?? 0);
    out[r] = dot * inv;
  }
  return out;
}

export interface SimilarRanking {
  /**
   * What was compared: the chosen instance, or when its views were too small to describe, the
   * nearest instance above it that was described (the object it is part of).
   */
  seedUsed: number;
  /** Per described instance id, its cosine with `seedUsed` (centred). */
  scores: ReadonlyMap<number, number>;
}

/**
 * How alike every described object of the scan looks to `seedId` (`describedMatrix`). Not
 * scored: what `seedUsed` is part of and what is part of it -- they hold the seed, and lighting
 * a whole that holds it would light the seed itself -- and what was never described. Null when
 * neither `seedId` nor anything above it was described.
 */
export function rankSimilar(
  doc: Pick<InstancesDoc, "instances" | "byId">,
  matrix: DescribedMatrix,
  seedId: number,
): SimilarRanking | null {
  let seedUsed: number | null = null;
  const above = new Set<number>();
  for (let id: number | null = seedId; id !== null && !above.has(id);) {
    above.add(id);
    if (matrix.rowOf.has(id)) {
      seedUsed = id;
      break;
    }
    id = doc.byId.get(id)?.parent ?? null;
  }
  const row = seedUsed === null ? undefined : matrix.rowOf.get(seedUsed);
  if (seedUsed === null || row === undefined) return null;
  const skip = withDescendants(doc, [seedUsed]);
  const seen = new Set<number>();
  for (let id = doc.byId.get(seedUsed)?.parent ?? null; id !== null && !seen.has(id);) {
    seen.add(id);
    skip.add(id);
    id = doc.byId.get(id)?.parent ?? null;
  }
  const { dim, m, ids } = matrix;
  const cosines = rankByVector(m.subarray(row * dim, (row + 1) * dim), m, dim);
  const scores = new Map<number, number>();
  for (let r = 0; r < ids.length; r += 1) {
    const id = ids[r] ?? 0;
    if (!skip.has(id)) scores.set(id, cosines[r] ?? 0);
  }
  return { seedUsed, scores };
}

/** What an embedding model id is called in a sentence: "SigLIP 2" for `siglip:google/siglip2-…`. */
export function embeddingModelName(model: string): string {
  const lower = model.toLowerCase();
  if (lower.includes("siglip2")) return "SigLIP 2";
  if (lower.includes("siglip")) return "SigLIP";
  if (lower.includes("clip")) return "CLIP";
  return model;
}
