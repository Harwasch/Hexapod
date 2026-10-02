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
 * `instances.emb` (float16, `count × dim`, row `k` is id `k + 1`, L2-normalised) is what text
 * search by meaning will read once there is a text encoder at runtime: `rankByEmbedding` is
 * the seam. v1 search is by tags.
 */

import { decodeRuns, runsLength, tileRunsIssue } from "@twin/world";

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

/** Fetches and reads a scan's `instances.json`. Throws when it is missing or not one. */
export async function loadInstances(tilesetUrl: string, ref: InstancesRef): Promise<InstancesDoc> {
  const response = await fetch(resolveBeside(tilesetUrl, ref.uri));
  if (!response.ok) throw new Error(`instances answered ${String(response.status)}`);
  const doc = parseInstances(await response.json());
  if (!doc) throw new Error("instances: not a hexapod.instances v1 document");
  return doc;
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

/** The label a search result goes by: its top tag (the panel names objects by category). */
export function instanceLabel(instance: Instance): string {
  return instance.tags[0]?.label ?? "untagged";
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
