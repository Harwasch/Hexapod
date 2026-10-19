import { create } from "zustand";

import {
  categoryById,
  indexCategories,
  matchCategory,
  OTHER_CATEGORY,
  type CategoryIndex,
  type SceneCategory,
} from "@/lib/categories";
import { embedQueryText, type EncodeText, type QueryEmbedding } from "@/api/textEmbedding";
import {
  describedMatrix,
  loadEmbeddings,
  meaningScores,
  parseQuery,
  searchInstances,
  textCosines,
  withDescendants,
  type DescribedMatrix,
  type EmbeddingSource,
  type Instance,
  type MeaningScores,
} from "@/lib/instances";
import { createLogger } from "@/lib/log";
import { selectionLabel } from "@/lib/sceneSelect";

import { canExtend, currentTask, discard, record, touched, type UndoStep } from "./history";

const log = createLogger("instances");

/**
 * Search by meaning for one scan: `none` (no `instances.emb`), `idle` (not asked yet),
 * `loading` (its rows or the query's embedding are on the way; matches are by tags),
 * `ready` (the matches use it), `unavailable` (a load failed, or the API has no encoder, or
 * a different model; matches are by tags).
 */
export type MeaningStatus = "none" | "idle" | "loading" | "ready" | "unavailable";

/** What the highlight is of, so the panel can mark it and a second click clears it. */
export type Focus =
  | { kind: "category"; id: string }
  | { kind: "object"; id: number }
  /** The query's matches, or only those in one category. */
  | { kind: "matches"; category?: string }
  | { kind: "ids" }
  /** The objects that look like `seedId` ("More like this", `AssetInstances.similar`). */
  | { kind: "similar"; seedId: number };

/** Where "More like this" lights its matches unless the viewer moves the slider. */
export const DEFAULT_SIMILAR_THRESHOLD = 0.6;

/**
 * "More like this" on a scan: how alike every described object looks to the chosen one
 * (lib/instances.ts `rankSimilar`), and how alike an object must be to be lit.
 */
export interface SimilarState {
  /** The instance the viewer chose. */
  seedId: number;
  /** What was compared: `seedId`, or the described instance it is part of. */
  seedUsed: number;
  /** Per instance id, its cosine with `seedUsed`, centred (-1..1). */
  scores: ReadonlyMap<number, number>;
  /** Lit: a score at least this. */
  threshold: number;
}

/** One scan's objects (lib/instances.ts, lib/categories.ts) and what the viewer does with them. */
export interface AssetInstances {
  instances: Instance[];
  /** The instances by broad category and object (`indexCategories`). */
  index: CategoryIndex;
  /**
   * Instance ids not drawn. Exact: the renderers hide these ids and nothing else, so a
   * category's or an object's members (never another category's instances under them).
   */
  hidden: ReadonlySet<number>;
  /** Instance ids tinted (exact, as `hidden`); while any are, the rest are dimmed. */
  highlighted: ReadonlySet<number>;
  /** What `highlighted` is of, or null when nothing is. */
  focus: Focus | null;
  query: string;
  /** Object ids the query matches, best first: what "Hide all" and "Show only" act on. */
  matches: readonly number[];
  /** Of `matches`, those whose best evidence was meaning rather than a tag (`via`). */
  byMeaning: ReadonlySet<number>;
  /** Where the scan's `instances.emb` is, or null when it has none. */
  embedding: EmbeddingSource | null;
  meaning: MeaningStatus;
  /** "More like this", while its matches are what is lit (`focus` is `similar`). */
  similar: SimilarState | null;
}

/** A splat renderer that cannot hide or highlight this scan's objects, and why. */
export interface RendererGap {
  /** The renderer drawing the scan now. */
  renderer: string;
  reason: string;
}

interface InstancesState {
  /** Per asset id, present while the asset's tileset declares instances and they loaded. */
  assets: Record<string, AssetInstances>;
  /** Whether everything but a highlight is dimmed while one is active (always, in the app). */
  dimOthers: boolean;
  /**
   * Per asset id, set while the splat renderer drawing it cannot apply hide and highlight
   * (`cesium/scanView`): the panel says so and offers the CesiumJS renderer.
   */
  gaps: Record<string, RendererGap>;
  setGap: (assetId: string, gap: RendererGap | null) => void;
  /**
   * Per asset id, set while the splat renderer drawing it cannot move its objects (skins in
   * the wind, telemetry; `cesium/scanView/scanMotion.ts`): the objects panel and the wind
   * control say so and offer the CesiumJS renderer.
   */
  motionGaps: Record<string, RendererGap>;
  setMotionGap: (assetId: string, gap: RendererGap | null) => void;
  setTable: (
    assetId: string,
    table: { instances: Instance[]; embeddingSource?: EmbeddingSource | null } | null,
  ) => void;
  /**
   * Matches at once by tags, then -- when the scan has `instances.emb` and the query has
   * words -- again by meaning as soon as the rows and the query's embedding are here.
   */
  setQuery: (assetId: string, query: string) => void;
  // What changes `hidden` (below, up to `reset`) is one undoable step each (`state/history.ts`,
  // `changeHidden`); the query and the highlight are not recorded.
  /** Hides or shows instances with everything below them (`withDescendants`). */
  setHidden: (assetId: string, ids: readonly number[], hidden: boolean) => void;
  /** Hides or shows objects: their members (`SceneObject.members`). */
  setObjectsHidden: (assetId: string, objectIds: readonly number[], hidden: boolean) => void;
  /** Hides or shows every object of a category. */
  setCategoryHidden: (assetId: string, categoryId: string, hidden: boolean) => void;
  /** Hides every object the current query matches. */
  hideMatches: (assetId: string) => void;
  /** Hides everything but the objects the current query matches. */
  showOnlyMatches: (assetId: string) => void;
  showAll: (assetId: string) => void;
  /** Highlights instances with everything below them; an empty list clears the highlight. */
  highlight: (assetId: string, ids: readonly number[]) => void;
  /** Highlights a category, an object or the query's matches; the same again (or null) clears. */
  toggleFocus: (assetId: string, focus: Focus | null) => void;
  /**
   * Lights the objects that look like `seedId` at least `threshold` alike, with `seedUsed`
   * (what was compared) and everything below them; any other highlight gives way, and any
   * other highlight (`highlight`, `toggleFocus`, `reset`) ends it.
   */
  showSimilar: (
    assetId: string,
    seedId: number,
    seedUsed: number,
    scores: ReadonlyMap<number, number>,
    threshold?: number,
  ) => void;
  /** Lights the "More like this" matches at least `threshold` alike instead. */
  setSimilarThreshold: (assetId: string, threshold: number) => void;
  /** Nothing hidden, nothing highlighted (undo brings back what was hidden). */
  reset: (assetId: string) => void;
  setDimOthers: (dim: boolean) => void;
}

const EMPTY: ReadonlySet<number> = new Set();

function patch(
  state: InstancesState,
  assetId: string,
  change: (current: AssetInstances) => Partial<AssetInstances>,
): Pick<InstancesState, "assets"> | InstancesState {
  const current = state.assets[assetId];
  if (!current) return state;
  return { assets: { ...state.assets, [assetId]: { ...current, ...change(current) } } };
}

/** `gaps` with `assetId`'s set to `gap` (or cleared), or `gaps` itself when nothing changes. */
function withGap(
  gaps: Record<string, RendererGap>,
  assetId: string,
  gap: RendererGap | null,
): Record<string, RendererGap> {
  const current = gaps[assetId];
  if (gap === null) {
    if (!current) return gaps;
    return Object.fromEntries(Object.entries(gaps).filter(([id]) => id !== assetId));
  }
  if (current?.renderer === gap.renderer && current.reason === gap.reason) return gaps;
  return { ...gaps, [assetId]: gap };
}

// ---- Undo --------------------------------------------------------------------------------

/**
 * A second change of the same thing this soon after the first (an eye flicked off and on)
 * folds into one step: the net change, or none when it came back to where it was.
 */
export const TOGGLE_COALESCE_MS = 800;

/** One undoable change of a scan's hidden set: the exact sets before and after. */
interface HiddenStep extends UndoStep {
  assetId: string;
  /** The table it is of: a reloaded scan's ids are another scan's, and the step dies. */
  table: readonly Instance[];
  before: ReadonlySet<number>;
  after: ReadonlySet<number>;
  /** What was changed ("objects:3", "category:trees"), for folding a quick repeat in. */
  key: string;
  at: number;
  task: number;
}

let lastHidden: HiddenStep | null = null;

function sameMembers(a: ReadonlySet<number>, b: ReadonlySet<number>): boolean {
  if (a === b) return true;
  if (a.size !== b.size) return false;
  for (const id of a) if (!b.has(id)) return false;
  return true;
}

function capitalise(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** What a person calls instance `id`: its object's name as the panel lists it, else the card's. */
function instanceName(entry: AssetInstances, id: number): string {
  const object = entry.index.objects.get(id);
  if (object) return capitalise(object.name);
  const instance = entry.instances.find((i) => i.id === id);
  // Not in the file: an object painted in this browser (lib/customSets.ts).
  if (!instance) return "Painted object";
  return selectionLabel(instance, id, entry.index.categoryOf.get(id));
}

function namesOf(entry: AssetInstances, ids: readonly number[]): string {
  const [only] = ids;
  return ids.length === 1 && only !== undefined
    ? instanceName(entry, only)
    : `${ids.length.toLocaleString()} objects`;
}

/** Some objects in words: "Pumpkin 3", "Trees" (all of them), "3 objects in Trees". */
function objectsName(entry: AssetInstances, objectIds: readonly number[]): string {
  const objects = objectIds.flatMap((id) => entry.index.objects.get(id) ?? []);
  const [first] = objects;
  if (!first) return `${objectIds.length.toLocaleString()} objects`;
  if (objects.length === 1) return capitalise(first.name);
  const group = entry.index.groups.find((g) => g.category.id === first.category);
  if (!group || objects.some((o) => o.category !== first.category))
    return `${objects.length.toLocaleString()} objects`;
  return objects.length === group.objects.length
    ? group.category.name
    : `${objects.length.toLocaleString()} objects in ${group.category.name}`;
}

/** Of `ids`, those whose parent is not among them: the tops of what they cover. */
function topsOf(entry: AssetInstances, ids: ReadonlySet<number>): number[] {
  const parentOf = new Map(entry.instances.map((i) => [i.id, i.parent]));
  return [...ids].filter((id) => {
    const parent = parentOf.get(id);
    return parent === null || parent === undefined || !ids.has(parent);
  });
}

/** The tops of what is wholly drawn with `hidden` hidden: drawn, and all below it too. */
function wholeShownTops(entry: AssetInstances, hidden: ReadonlySet<number>): number[] {
  const children = new Map<number, number[]>();
  for (const i of entry.instances) {
    if (i.parent === null) continue;
    const list = children.get(i.parent) ?? [];
    list.push(i.id);
    children.set(i.parent, list);
  }
  const whole = new Map<number, boolean>();
  const isWhole = (id: number): boolean => {
    const known = whole.get(id);
    if (known !== undefined) return known;
    const value = !hidden.has(id) && (children.get(id) ?? []).every(isWhole);
    whole.set(id, value);
    return value;
  };
  return entry.instances
    .filter((i) => isWhole(i.id) && (i.parent === null || !isWhole(i.parent)))
    .map((i) => i.id);
}

/**
 * A change of the hidden set in words, from what it did: for a change made by ids (the
 * selection card's Hide and Show only, through `setHidden`) or by several actions in one
 * gesture ("Show only" is show all, then hide the rest).
 */
function describeHiddenChange(
  entry: AssetInstances,
  before: ReadonlySet<number>,
  after: ReadonlySet<number>,
): string {
  const added = new Set([...after].filter((id) => !before.has(id)));
  const removed = new Set([...before].filter((id) => !after.has(id)));
  if (added.size === 0) {
    const shownTops = topsOf(entry, removed);
    return after.size === 0 && shownTops.length > 1
      ? "Show all objects"
      : `Show ${namesOf(entry, shownTops)}`;
  }
  const hiddenTops = topsOf(entry, added);
  if (removed.size === 0 && hiddenTops.length === 1) return `Hide ${namesOf(entry, hiddenTops)}`;
  const shown = wholeShownTops(entry, after);
  if (shown.length > 0 && shown.length < hiddenTops.length)
    return `Show only ${namesOf(entry, shown)}`;
  return removed.size === 0 ? `Hide ${namesOf(entry, hiddenTops)}` : "Change what is shown";
}

function putHidden(assetId: string, hidden: ReadonlySet<number>): void {
  useInstances.setState((s) => patch(s, assetId, () => ({ hidden })));
}

/**
 * Records that `assetId`'s hidden set went from `before` to `after`: a step of its own, or
 * folded into the last one when it continues it -- the same gesture (one task), or the same
 * thing changed again within `TOGGLE_COALESCE_MS`. `label` null describes the change.
 */
function noteHidden(
  entry: AssetInstances,
  assetId: string,
  key: string,
  label: string | null,
  before: ReadonlySet<number>,
  after: ReadonlySet<number>,
): void {
  const now = Date.now();
  const task = currentTask();
  const last = lastHidden;
  if (
    last &&
    canExtend(last) &&
    last.assetId === assetId &&
    last.table === entry.instances &&
    (last.task === task || (last.key === key && now - last.at < TOGGLE_COALESCE_MS))
  ) {
    const gesture = last.task === task && last.key !== key;
    Object.assign(last, { after, at: now, task, key });
    last.label =
      gesture || label === null ? describeHiddenChange(entry, last.before, after) : label;
    if (sameMembers(last.before, after)) {
      discard(last);
      lastHidden = null;
    } else {
      touched(last);
    }
    return;
  }
  const step: HiddenStep = {
    label: label ?? describeHiddenChange(entry, before, after),
    scope: "site",
    assetId,
    table: entry.instances,
    before,
    after,
    key,
    at: now,
    task,
    alive: () => useInstances.getState().assets[assetId]?.instances === step.table,
    undo: () => putHidden(assetId, step.before),
    redo: () => putHidden(assetId, step.after),
  };
  record(step);
  lastHidden = step;
}

/**
 * Sets `assetId`'s hidden set to what `change` makes of it (null: leave it) as one undoable
 * step; a change that changes nothing is neither made nor recorded.
 */
function changeHidden(
  assetId: string,
  key: string,
  label: ((entry: AssetInstances) => string) | null,
  change: (current: AssetInstances) => ReadonlySet<number> | null,
): void {
  const current = useInstances.getState().assets[assetId];
  if (!current) return;
  const hidden = change(current);
  if (hidden === null || sameMembers(hidden, current.hidden)) return;
  putHidden(assetId, hidden);
  noteHidden(current, assetId, key, label ? label(current) : null, current.hidden, hidden);
}

/** `hidden` with `ids` added (`hide`) or taken out. */
function withIds(
  hidden: ReadonlySet<number>,
  ids: Iterable<number>,
  hide: boolean,
): ReadonlySet<number> {
  const next = new Set(hidden);
  for (const id of ids) {
    if (hide) next.add(id);
    else next.delete(id);
  }
  return next;
}

/** "Hide 12 matches for “pumpkin”". */
function matchesLabel(verb: string, entry: AssetInstances): string {
  const n = entry.matches.length;
  return `${verb} ${n.toLocaleString()} ${n === 1 ? "match" : "matches"} for “${entry.query.trim()}”`;
}

/** Whether two highlights are of the same thing (a second click on it clears it). */
export function sameFocus(a: Focus | null, b: Focus | null): boolean {
  if (a === null || b === null) return a === b;
  if (a.kind !== b.kind) return false;
  if (a.kind === "matches" && b.kind === "matches") return a.category === b.category;
  if (a.kind === "similar" && b.kind === "similar") return a.seedId === b.seedId;
  return !("id" in a) || a.id === (b as typeof a).id;
}

/** The instances "More like this" matches at its threshold, best first. */
export function similarMatches(similar: SimilarState): number[] {
  return [...similar.scores]
    .filter(([, score]) => score >= similar.threshold)
    .sort((a, b) => b[1] - a[1] || a[0] - b[0])
    .map(([id]) => id);
}

/** What "More like this" lights: the matches and what was compared, with all below them. */
function similarLit(
  entry: Pick<AssetInstances, "instances">,
  similar: SimilarState,
): ReadonlySet<number> {
  return withDescendants(entry, [similar.seedUsed, ...similarMatches(similar)]);
}

/** The matches of "More like this" as a person counts them: objects, and what they mostly are. */
export interface SimilarSummary {
  /** Objects with a match in them, not counting the one compared (`index.objectOf`). */
  objects: number;
  /** Those objects by category, the most first (ties to the category listed first). */
  categories: { category: SceneCategory; count: number }[];
}

/**
 * The objects "More like this" lights at its threshold, by category. Several matches in one
 * object (a conifer's branches) are one object; a match within the object compared is not
 * counted, as the object itself is not.
 */
export function similarSummary(
  entry: Pick<AssetInstances, "index" | "similar"> | undefined,
): SimilarSummary | null {
  const similar = entry?.similar;
  if (!entry || !similar) return null;
  const { objectOf, objects: byId, categoryOf } = entry.index;
  const own = objectOf.get(similar.seedUsed) ?? similar.seedUsed;
  const objects = new Set<number>();
  for (const id of similarMatches(similar)) {
    const object = objectOf.get(id) ?? id;
    if (object !== own) objects.add(object);
  }
  const counts = new Map<string, number>();
  for (const object of objects) {
    const category = byId.get(object)?.category ?? categoryOf.get(object) ?? OTHER_CATEGORY;
    counts.set(category, (counts.get(category) ?? 0) + 1);
  }
  const order = entry.index.groups.map((g) => g.category.id);
  const categories = [...counts]
    .map(([id, count]) => ({ category: categoryById(id), count }))
    .sort(
      (a, b) => b.count - a.count || order.indexOf(a.category.id) - order.indexOf(b.category.id),
    );
  return { objects: objects.size, categories };
}

/**
 * A summary in words, "37 objects · mostly Shrubs & bushes (24)": "mostly" when one category
 * is at least half of them, else the two largest.
 */
export function similarText(summary: SimilarSummary): string {
  const { objects, categories } = summary;
  if (objects === 0) return "No other object is this alike. Lower How alike to see more.";
  const [top, next] = categories;
  const counted = `${objects.toLocaleString()} ${objects === 1 ? "object" : "objects"}`;
  if (!top) return counted;
  if (objects === 1) return `${counted} · ${top.category.name}`;
  if (top.count * 2 >= objects)
    return `${counted} · mostly ${top.category.name} (${String(top.count)})`;
  // Neither is half, so there are more categories than these two.
  const named = [top, next].filter((c) => c !== undefined);
  return `${counted} · ${named.map((c) => `${c.category.name} (${String(c.count)})`).join(", ")}, …`;
}

/** The instance ids of some objects. */
function membersOf(index: CategoryIndex, objectIds: Iterable<number>): number[] {
  const out: number[] = [];
  for (const id of objectIds) out.push(...(index.objects.get(id)?.members ?? []));
  return out;
}

/** The instance ids a focus lights. */
function focusMembers(entry: AssetInstances, focus: Focus): readonly number[] {
  switch (focus.kind) {
    case "category":
      return entry.index.groups.find((g) => g.category.id === focus.id)?.members ?? [];
    case "object":
      return entry.index.objects.get(focus.id)?.members ?? [];
    case "matches":
      return membersOf(
        entry.index,
        focus.category === undefined
          ? entry.matches
          : entry.matches.filter((id) => entry.index.objects.get(id)?.category === focus.category),
      );
    case "ids":
      return [...entry.highlighted];
    case "similar":
      // Only while its scores are held: they are what it lights.
      return entry.similar?.seedId === focus.seedId ? [...similarLit(entry, entry.similar)] : [];
  }
}

/** The objects a query matches (`matchObjects`), and which of them meaning found. */
export interface ObjectMatches {
  matches: number[];
  byMeaning: ReadonlySet<number>;
}

/**
 * The objects a query matches, best first: objects whose tags (or property filters) match --
 * or, with `meaning`, whose look matches (lib/instances.ts `searchInstances`) -- each found
 * through any of its parts; and, for words that name a category ("trees", "water"), every
 * object in it, first.
 */
export function matchObjectsBy(
  entry: Pick<AssetInstances, "instances" | "index">,
  query: string,
  meaning?: MeaningScores,
): ObjectMatches {
  const parsed = parseQuery(query);
  const byMeaning = new Set<number>();
  if (parsed.terms.length === 0 && parsed.filters.length === 0) return { matches: [], byMeaning };
  const best = new Map<number, number>();
  const note = (objectId: number | undefined, score: number, viaMeaning = false): void => {
    if (objectId === undefined) return;
    if (score <= (best.get(objectId) ?? -1)) return;
    best.set(objectId, score);
    if (viaMeaning) byMeaning.add(objectId);
    else byMeaning.delete(objectId);
  };
  if (parsed.filters.length === 0) {
    for (const group of entry.index.groups) {
      if (matchCategory(parsed.terms, group.category) <= 0) continue;
      for (const object of group.objects) note(object.id, 2 + object.splats * 1e-12);
    }
  }
  for (const result of searchInstances(
    entry.instances,
    parsed,
    Number.POSITIVE_INFINITY,
    meaning,
  )) {
    note(entry.index.objectOf.get(result.id), result.score, result.via === "meaning");
  }
  const splats = (id: number): number => entry.index.objects.get(id)?.splats ?? 0;
  const matches = [...best]
    .sort((a, b) => b[1] - a[1] || splats(b[0]) - splats(a[0]))
    .map(([id]) => id);
  return { matches, byMeaning };
}

/** `matchObjectsBy`'s matches alone. */
export function matchObjects(
  entry: Pick<AssetInstances, "instances" | "index">,
  query: string,
  meaning?: MeaningScores,
): number[] {
  return matchObjectsBy(entry, query, meaning).matches;
}

// ---- Search by meaning: what it needs, loaded once and kept --------------------------------

export interface MeaningDeps {
  encode: EncodeText;
  /** A scan's `instances.emb`, the raw float16 bits (lib/instances.ts `loadEmbeddings`). */
  loadEmbeddings: (source: EmbeddingSource) => Promise<Uint16Array>;
  /** How long typing must pause before the query is sent to the encoder. */
  debounceMs: number;
  /** After a load fails (503, offline), how long search stays on tags before asking again. */
  retryMs: number;
}

const DEFAULT_DEPS: MeaningDeps = {
  encode: embedQueryText,
  // Shared with "More like this" (lib/payloadCache.ts): a scan's file is read once for both.
  loadEmbeddings: (source) => loadEmbeddings(source.url, source.rows, source.dim),
  debounceMs: 200,
  retryMs: 30_000,
};

let deps: MeaningDeps = DEFAULT_DEPS;
/** Per `instances.emb` URL: its described rows (`describedMatrix`), or their load. */
const rowsLoading = new Map<string, Promise<DescribedMatrix>>();
const rowsLoaded = new Map<string, DescribedMatrix>();
/** Per `instances.emb` URL whose load failed: when to try again. */
const rowsDownUntil = new Map<string, number>();
/** Per query text: its embedding, or its request. */
const queriesLoading = new Map<string, Promise<QueryEmbedding>>();
const queriesLoaded = new Map<string, QueryEmbedding>();
const QUERY_CACHE = 200;
let encoderDownUntil = 0;
const timers = new Map<string, ReturnType<typeof setTimeout>>();
/** The last scores computed, so a filter change does not redo the dot products. */
let lastScores: { rows: DescribedMatrix; text: string; scores: MeaningScores } | undefined;

/** Swaps what search by meaning calls (tests), and forgets everything it had loaded. */
export function configureMeaning(change: Partial<MeaningDeps> | null): void {
  deps = change === null ? DEFAULT_DEPS : { ...deps, ...change };
  rowsLoading.clear();
  rowsLoaded.clear();
  rowsDownUntil.clear();
  queriesLoading.clear();
  queriesLoaded.clear();
  encoderDownUntil = 0;
  for (const timer of timers.values()) clearTimeout(timer);
  timers.clear();
  lastScores = undefined;
}

function rowsFor(source: EmbeddingSource): Promise<DescribedMatrix> {
  let pending = rowsLoading.get(source.url);
  if (!pending) {
    if (Date.now() < (rowsDownUntil.get(source.url) ?? 0)) {
      return Promise.reject(new Error("instances.emb did not load a moment ago"));
    }
    pending = deps.loadEmbeddings(source).then(
      (raw) => {
        const rows = describedMatrix(raw, source.dim);
        rowsLoaded.set(source.url, rows);
        return rows;
      },
      (error: unknown) => {
        rowsLoading.delete(source.url);
        rowsDownUntil.set(source.url, Date.now() + deps.retryMs);
        throw error;
      },
    );
    rowsLoading.set(source.url, pending);
  }
  return pending;
}

function queryFor(text: string): Promise<QueryEmbedding> {
  let pending = queriesLoading.get(text);
  if (!pending) {
    if (Date.now() < encoderDownUntil) return Promise.reject(new Error("text encoder is down"));
    pending = deps.encode(text).then(
      (embedding) => {
        queriesLoaded.set(text, embedding);
        if (queriesLoaded.size > QUERY_CACHE) {
          const oldest = queriesLoaded.keys().next().value;
          if (oldest !== undefined) {
            queriesLoaded.delete(oldest);
            queriesLoading.delete(oldest);
          }
        }
        return embedding;
      },
      (error: unknown) => {
        queriesLoading.delete(text);
        encoderDownUntil = Date.now() + deps.retryMs;
        throw error;
      },
    );
    queriesLoading.set(text, pending);
  }
  return pending;
}

/** The words of a query, as the encoder is asked about them (filters are not meaning). */
function meaningText(query: string): string {
  return parseQuery(query).terms.join(" ");
}

/** `unavailable` is tried again once the back-offs have passed. */
function retryable(status: MeaningStatus): MeaningStatus {
  if (status !== "unavailable") return status;
  const now = Date.now();
  const rowsDown = [...rowsDownUntil.values()].some((until) => now < until);
  return now >= encoderDownUntil && !rowsDown ? "idle" : status;
}

/** Matches for `query`, by meaning too when everything it needs is already here. */
function searchNow(
  current: AssetInstances,
  query: string,
): ObjectMatches & { meaning: MeaningStatus } {
  const text = meaningText(query);
  const source = current.embedding;
  if (!source) return { ...matchObjectsBy(current, query), meaning: "none" };
  const rows = rowsLoaded.get(source.url);
  const embedding = queriesLoaded.get(text);
  if (text && rows && embedding?.model === source.model) {
    let scores = lastScores?.rows === rows && lastScores.text === text ? lastScores.scores : null;
    if (!scores) {
      scores = meaningScores(current.instances, textCosines(embedding.vector, rows));
      lastScores = { rows, text, scores };
    }
    return { ...matchObjectsBy(current, query, scores), meaning: "ready" };
  }
  const found = matchObjectsBy(current, query);
  const keep = current.meaning === "unavailable" ? "unavailable" : null;
  if (!text) return { ...found, meaning: keep ?? "idle" };
  if (embedding && embedding.model !== source.model) return { ...found, meaning: "unavailable" };
  return { ...found, meaning: keep ?? "loading" };
}

/** What setting `query` changes: the matches, and a highlight of the old matches follows. */
function withQuery(current: AssetInstances, query: string): Partial<AssetInstances> {
  const { matches, byMeaning, meaning } = searchNow(current, query);
  if (current.focus?.kind === "matches") {
    const lit = focusMembers({ ...current, matches }, current.focus);
    return {
      query,
      matches,
      byMeaning,
      meaning,
      highlighted: lit.length ? new Set(lit) : EMPTY,
      // Kept while meaning is on the way, so its matches light when they arrive.
      focus: lit.length || meaning === "loading" ? current.focus : null,
    };
  }
  return { query, matches, byMeaning, meaning };
}

/** Fetches what `query` needs for meaning, then matches again if it is still the query. */
function refine(assetId: string, query: string): void {
  const entry = useInstances.getState().assets[assetId];
  const source = entry?.embedding;
  const text = meaningText(query);
  const pendingTimer = timers.get(assetId);
  if (pendingTimer !== undefined) clearTimeout(pendingTimer);
  timers.delete(assetId);
  if (!entry || !source || !text || entry.meaning === "ready") return;
  // The rows start loading on the first search (they are the larger download); the query
  // waits for typing to pause.
  const rows = rowsFor(source);
  const table = entry.instances;
  const current = (): boolean => {
    const now = useInstances.getState().assets[assetId];
    return now?.query === query && now.instances === table;
  };
  const fail = (what: string) => (error: unknown) => {
    log.warn(`search by meaning: ${what} failed; searching by tags`, {
      asset: assetId,
      message: error instanceof Error ? error.message : String(error),
    });
    if (current())
      useInstances.setState((s) => patch(s, assetId, () => ({ meaning: "unavailable" })));
  };
  rows.catch(fail("instances.emb"));
  timers.set(
    assetId,
    setTimeout(() => {
      timers.delete(assetId);
      if (!current()) return;
      const embedding = queryFor(text);
      embedding.catch(fail("the query's embedding"));
      Promise.all([rows, embedding])
        .then(([, asked]) => {
          if (asked.model !== source.model) {
            fail("matching models")(
              new Error(`the encoder is ${asked.model}, the scan's rows ${source.model}`),
            );
            return;
          }
          if (current())
            useInstances.setState((s) => patch(s, assetId, (now) => withQuery(now, query)));
        })
        // Each failure was reported where it happened.
        .catch(() => undefined);
    }, deps.debounceMs),
  );
}

export const useInstances = create<InstancesState>()((set) => ({
  assets: {},
  dimOthers: true,
  gaps: {},
  setGap: (assetId, gap) =>
    set((s) => {
      const gaps = withGap(s.gaps, assetId, gap);
      return gaps === s.gaps ? s : { gaps };
    }),
  motionGaps: {},
  setMotionGap: (assetId, gap) =>
    set((s) => {
      const motionGaps = withGap(s.motionGaps, assetId, gap);
      return motionGaps === s.motionGaps ? s : { motionGaps };
    }),
  setTable: (assetId, table) =>
    set((s) => {
      const next: Record<string, AssetInstances> = {};
      for (const [id, entry] of Object.entries(s.assets)) if (id !== assetId) next[id] = entry;
      if (table && table.instances.length > 0) {
        const embedding = table.embeddingSource ?? null;
        next[assetId] = {
          instances: table.instances,
          index: indexCategories(table.instances),
          hidden: EMPTY,
          highlighted: EMPTY,
          focus: null,
          query: "",
          matches: [],
          byMeaning: EMPTY,
          embedding,
          meaning: embedding ? "idle" : "none",
          similar: null,
        };
      }
      return { assets: next };
    }),
  setQuery: (assetId, query) => {
    set((s) =>
      patch(s, assetId, (current) =>
        withQuery({ ...current, meaning: retryable(current.meaning) }, query),
      ),
    );
    refine(assetId, query);
  },
  setHidden: (assetId, ids, hidden) =>
    changeHidden(assetId, `ids:${String(ids[0])}:${String(ids.length)}`, null, (current) =>
      withIds(current.hidden, withDescendants(current, ids), hidden),
    ),
  setObjectsHidden: (assetId, objectIds, hidden) =>
    changeHidden(
      assetId,
      `objects:${objectIds.join(",")}`,
      (entry) => `${hidden ? "Hide" : "Show"} ${objectsName(entry, objectIds)}`,
      (current) => withIds(current.hidden, membersOf(current.index, objectIds), hidden),
    ),
  setCategoryHidden: (assetId, categoryId, hidden) =>
    changeHidden(
      assetId,
      `category:${categoryId}`,
      (entry) => {
        const group = entry.index.groups.find((g) => g.category.id === categoryId);
        return `${hidden ? "Hide" : "Show"} ${group?.category.name ?? "a category"}`;
      },
      (current) => {
        const group = current.index.groups.find((g) => g.category.id === categoryId);
        return group ? withIds(current.hidden, group.members, hidden) : null;
      },
    ),
  hideMatches: (assetId) =>
    changeHidden(
      assetId,
      "matches",
      (entry) => matchesLabel("Hide", entry),
      (current) =>
        current.matches.length === 0
          ? null
          : withIds(current.hidden, membersOf(current.index, current.matches), true),
    ),
  showOnlyMatches: (assetId) =>
    changeHidden(
      assetId,
      "matches",
      (entry) => matchesLabel("Show only", entry),
      (current) => {
        if (current.matches.length === 0) return null;
        const keep = new Set(membersOf(current.index, current.matches));
        return new Set(current.instances.filter((i) => !keep.has(i.id)).map((i) => i.id));
      },
    ),
  showAll: (assetId) =>
    changeHidden(
      assetId,
      "all",
      () => "Show all objects",
      () => EMPTY,
    ),
  highlight: (assetId, ids) =>
    set((s) =>
      patch(s, assetId, (current) =>
        ids.length
          ? { highlighted: withDescendants(current, ids), focus: { kind: "ids" }, similar: null }
          : { highlighted: EMPTY, focus: null, similar: null },
      ),
    ),
  toggleFocus: (assetId, focus) =>
    set((s) =>
      patch(s, assetId, (current) => {
        if (focus === null || sameFocus(current.focus, focus)) {
          return { highlighted: EMPTY, focus: null, similar: null };
        }
        const lit = focusMembers(current, focus);
        // Lighting its own matches again keeps "More like this"; anything else ends it.
        const similar = focus.kind === "similar" ? current.similar : null;
        return lit.length
          ? { highlighted: new Set(lit), focus, similar }
          : { highlighted: EMPTY, focus: null, similar: null };
      }),
    ),
  showSimilar: (assetId, seedId, seedUsed, scores, threshold = DEFAULT_SIMILAR_THRESHOLD) =>
    set((s) =>
      patch(s, assetId, (current) => {
        const similar: SimilarState = { seedId, seedUsed, scores, threshold };
        return {
          similar,
          highlighted: similarLit(current, similar),
          focus: { kind: "similar", seedId },
        };
      }),
    ),
  setSimilarThreshold: (assetId, threshold) =>
    set((s) => {
      const current = s.assets[assetId]?.similar;
      if (!current || current.threshold === threshold) return s;
      return patch(s, assetId, (entry) => {
        const similar: SimilarState = { ...current, threshold };
        return { similar, highlighted: similarLit(entry, similar) };
      });
    }),
  reset: (assetId) => {
    changeHidden(
      assetId,
      "reset",
      () => "Reset objects",
      () => EMPTY,
    );
    set((s) => patch(s, assetId, () => ({ highlighted: EMPTY, focus: null, similar: null })));
  },
  setDimOthers: (dimOthers) => set({ dimOthers }),
}));
