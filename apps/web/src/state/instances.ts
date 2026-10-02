import { create } from "zustand";

import { embedQueryText, type EncodeText, type QueryEmbedding } from "@/api/textEmbedding";
import {
  cosineById,
  loadEmbeddings,
  meaningScores,
  parseQuery,
  quickFilters,
  searchInstances,
  type EmbeddingRef,
  type Instance,
  type InstanceEmbeddings,
  type MeaningScores,
  type QuickFilter,
  type SearchResult,
} from "@/lib/instances";
import { createLogger } from "@/lib/log";

const log = createLogger("instances");

/** Where a scan's `instances.emb` is, for search by meaning. */
export interface EmbeddingSource {
  /** The `instances.json` URL; the file is resolved beside it. */
  instancesUrl: string;
  ref: EmbeddingRef;
}

/**
 * Search by meaning for one scan: `none` (no `instances.emb`), `idle` (not asked yet),
 * `loading` (its rows or the query's embedding are on the way; results are by tags),
 * `ready` (the results use it), `unavailable` (a load failed, or the API has no encoder, or
 * a different model; results are by tags).
 */
export type MeaningStatus = "none" | "idle" | "loading" | "ready" | "unavailable";

/** One scan's objects (lib/instances.ts) and what the viewer does with them. */
export interface AssetInstances {
  instances: Instance[];
  propertyNames: string[];
  filters: QuickFilter[];
  /** Instances not drawn (with everything below them; see `withDescendants`). */
  hidden: ReadonlySet<number>;
  /** Instances tinted; while any are, the rest may be dimmed (`dimOthers`). */
  highlighted: ReadonlySet<number>;
  query: string;
  results: SearchResult[];
  embeddingSource: EmbeddingSource | null;
  meaning: MeaningStatus;
}

interface InstancesState {
  /** Per asset id, present while the asset's tileset declares instances and they loaded. */
  assets: Record<string, AssetInstances>;
  /** Whether everything but a highlight is dimmed while one is active. */
  dimOthers: boolean;
  setTable: (
    assetId: string,
    table: {
      instances: Instance[];
      propertyNames: string[];
      embeddingSource?: EmbeddingSource | null;
    } | null,
  ) => void;
  /**
   * Searches at once by tags, then -- when the scan has `instances.emb` and the query has
   * words -- again by meaning as soon as the rows and the query's embedding are here.
   */
  setQuery: (assetId: string, query: string) => void;
  setHidden: (assetId: string, ids: readonly number[], hidden: boolean) => void;
  toggleHidden: (assetId: string, id: number) => void;
  showAll: (assetId: string) => void;
  /** Replaces the highlight; an empty list clears it. */
  highlight: (assetId: string, ids: readonly number[]) => void;
  setDimOthers: (dim: boolean) => void;
}

const EMPTY: ReadonlySet<number> = new Set();

// ---- Search by meaning: what it needs, loaded once and kept --------------------------------

export interface MeaningDeps {
  encode: EncodeText;
  loadEmbeddings: (instancesUrl: string, ref: EmbeddingRef) => Promise<InstanceEmbeddings>;
  /** How long typing must pause before the query is sent to the encoder. */
  debounceMs: number;
  /** After the encoder fails (503, offline), how long search stays on tags before asking again. */
  retryMs: number;
}

const DEFAULT_DEPS: MeaningDeps = {
  encode: embedQueryText,
  loadEmbeddings,
  debounceMs: 200,
  retryMs: 30_000,
};

let deps: MeaningDeps = DEFAULT_DEPS;
/** Per `instances.emb` URL: the rows, or their load. A failed load is forgotten. */
const rowsLoading = new Map<string, Promise<InstanceEmbeddings>>();
const rowsLoaded = new Map<string, InstanceEmbeddings>();
/** Per `instances.emb` URL whose load failed: when to try again. */
const rowsDownUntil = new Map<string, number>();
/** Per query text: its embedding, or its request. */
const queriesLoading = new Map<string, Promise<QueryEmbedding>>();
const queriesLoaded = new Map<string, QueryEmbedding>();
const QUERY_CACHE = 200;
let encoderDownUntil = 0;
const timers = new Map<string, ReturnType<typeof setTimeout>>();
/** The last scores computed, so a filter change does not redo the dot products. */
let lastScores: { rows: InstanceEmbeddings; text: string; scores: MeaningScores } | undefined;

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

function rowsFor(source: EmbeddingSource): Promise<InstanceEmbeddings> {
  const key = `${source.instancesUrl} ${source.ref.file}`;
  let pending = rowsLoading.get(key);
  if (!pending) {
    if (Date.now() < (rowsDownUntil.get(key) ?? 0)) {
      return Promise.reject(new Error("instances.emb did not load a moment ago"));
    }
    pending = deps.loadEmbeddings(source.instancesUrl, source.ref).then(
      (rows) => {
        rowsLoaded.set(key, rows);
        return rows;
      },
      (error: unknown) => {
        rowsLoading.delete(key);
        rowsDownUntil.set(key, Date.now() + deps.retryMs);
        throw error;
      },
    );
    rowsLoading.set(key, pending);
  }
  return pending;
}

function loadedRows(source: EmbeddingSource): InstanceEmbeddings | undefined {
  return rowsLoaded.get(`${source.instancesUrl} ${source.ref.file}`);
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

/** Results for `query`, by meaning too when everything it needs is already here. */
function searchNow(
  current: Pick<AssetInstances, "instances" | "embeddingSource" | "meaning">,
  query: string,
): Pick<AssetInstances, "results" | "meaning"> {
  const text = meaningText(query);
  const source = current.embeddingSource;
  if (!source) return { results: searchInstances(current.instances, query), meaning: "none" };
  const rows = loadedRows(source);
  const embedding = queriesLoaded.get(text);
  if (text && rows && embedding?.model === rows.model) {
    let scores = lastScores?.rows === rows && lastScores.text === text ? lastScores.scores : null;
    if (!scores) {
      scores = meaningScores(current.instances, cosineById(embedding.vector, rows));
      lastScores = { rows, text, scores };
    }
    return { results: searchInstances(current.instances, query, 50, scores), meaning: "ready" };
  }
  const results = searchInstances(current.instances, query);
  if (!text)
    return { results, meaning: current.meaning === "unavailable" ? "unavailable" : "idle" };
  if (embedding && rows && embedding.model !== rows.model)
    return { results, meaning: "unavailable" };
  return { results, meaning: current.meaning === "unavailable" ? "unavailable" : "loading" };
}

function patch(
  state: InstancesState,
  assetId: string,
  change: (current: AssetInstances) => Partial<AssetInstances>,
): Pick<InstancesState, "assets"> | InstancesState {
  const current = state.assets[assetId];
  if (!current) return state;
  return { assets: { ...state.assets, [assetId]: { ...current, ...change(current) } } };
}

export const useInstances = create<InstancesState>()((set, get) => {
  /** Fetches what `query` needs for meaning, then searches again if it is still the query. */
  const refine = (assetId: string, query: string): void => {
    const entry = get().assets[assetId];
    const source = entry?.embeddingSource;
    const text = meaningText(query);
    const pendingTimer = timers.get(assetId);
    if (pendingTimer !== undefined) clearTimeout(pendingTimer);
    timers.delete(assetId);
    if (!entry || !source || !text || entry.meaning === "ready") return;
    // The rows start loading on the first search (they are the larger download); the
    // query waits for typing to pause.
    const rows = rowsFor(source);
    const current = (): boolean => get().assets[assetId]?.query === query;
    const fail = (what: string) => (error: unknown) => {
      log.warn(`search by meaning: ${what} failed; searching by tags`, {
        asset: assetId,
        message: error instanceof Error ? error.message : String(error),
      });
      if (current()) set((s) => patch(s, assetId, () => ({ meaning: "unavailable" })));
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
          .then(([loaded, asked]) => {
            if (asked.model !== loaded.model) {
              fail("matching models")(
                new Error(`the encoder is ${asked.model}, the scan's rows ${loaded.model}`),
              );
              return;
            }
            if (current()) set((s) => patch(s, assetId, (now) => searchNow(now, query)));
          })
          // Each failure was reported where it happened.
          .catch(() => undefined);
      }, deps.debounceMs),
    );
  };

  return {
    assets: {},
    dimOthers: true,
    setTable: (assetId, table) =>
      set((s) => {
        const next: Record<string, AssetInstances> = {};
        for (const [id, entry] of Object.entries(s.assets)) if (id !== assetId) next[id] = entry;
        if (table && table.instances.length > 0) {
          const source = table.embeddingSource ?? null;
          next[assetId] = {
            instances: table.instances,
            propertyNames: table.propertyNames,
            filters: quickFilters(table),
            hidden: EMPTY,
            highlighted: EMPTY,
            query: "",
            results: [],
            embeddingSource: source,
            meaning: source ? "idle" : "none",
          };
        }
        return { assets: next };
      }),
    setQuery: (assetId, query) => {
      set((s) =>
        patch(s, assetId, (current) => ({
          query,
          ...searchNow({ ...current, meaning: retryable(current.meaning) }, query),
        })),
      );
      refine(assetId, query);
    },
    setHidden: (assetId, ids, hidden) =>
      set((s) =>
        patch(s, assetId, (current) => {
          const next = new Set(current.hidden);
          for (const id of ids) {
            if (hidden) next.add(id);
            else next.delete(id);
          }
          return { hidden: next };
        }),
      ),
    toggleHidden: (assetId, id) =>
      set((s) =>
        patch(s, assetId, (current) => {
          const next = new Set(current.hidden);
          if (next.has(id)) next.delete(id);
          else next.add(id);
          return { hidden: next };
        }),
      ),
    showAll: (assetId) => set((s) => patch(s, assetId, () => ({ hidden: EMPTY }))),
    highlight: (assetId, ids) =>
      set((s) => patch(s, assetId, () => ({ highlighted: ids.length ? new Set(ids) : EMPTY }))),
    setDimOthers: (dimOthers) => set({ dimOthers }),
  };
});

/** `unavailable` is tried again once the back-offs have passed. */
function retryable(status: MeaningStatus): MeaningStatus {
  if (status !== "unavailable") return status;
  const now = Date.now();
  const rowsDown = [...rowsDownUntil.values()].some((until) => now < until);
  return now >= encoderDownUntil && !rowsDown ? "idle" : status;
}
