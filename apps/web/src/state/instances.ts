import { create } from "zustand";

import { indexCategories, matchCategory, type CategoryIndex } from "@/lib/categories";
import { parseQuery, searchInstances, withDescendants, type Instance } from "@/lib/instances";

/** What the highlight is of, so the panel can mark it and a second click clears it. */
export type Focus =
  | { kind: "category"; id: string }
  | { kind: "object"; id: number }
  /** The query's matches, or only those in one category. */
  | { kind: "matches"; category?: string }
  | { kind: "ids" };

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
  setTable: (assetId: string, table: { instances: Instance[] } | null) => void;
  setQuery: (assetId: string, query: string) => void;
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
  /** Nothing hidden, nothing highlighted. */
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

/** Whether two highlights are of the same thing (a second click on it clears it). */
export function sameFocus(a: Focus | null, b: Focus | null): boolean {
  if (a === null || b === null) return a === b;
  if (a.kind !== b.kind) return false;
  if (a.kind === "matches" && b.kind === "matches") return a.category === b.category;
  return !("id" in a) || a.id === (b as typeof a).id;
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
  }
}

/**
 * The objects a query matches, best first: objects whose tags (or property filters) match,
 * each found through any of its parts; and, for words that name a category ("trees",
 * "water"), every object in it, first.
 */
export function matchObjects(
  entry: Pick<AssetInstances, "instances" | "index">,
  query: string,
): number[] {
  const parsed = parseQuery(query);
  if (parsed.terms.length === 0 && parsed.filters.length === 0) return [];
  const best = new Map<number, number>();
  const note = (objectId: number | undefined, score: number): void => {
    if (objectId === undefined) return;
    if (score > (best.get(objectId) ?? -1)) best.set(objectId, score);
  };
  if (parsed.filters.length === 0) {
    for (const group of entry.index.groups) {
      if (matchCategory(parsed.terms, group.category) <= 0) continue;
      for (const object of group.objects) note(object.id, 2 + object.splats * 1e-12);
    }
  }
  for (const result of searchInstances(entry.instances, parsed, Number.POSITIVE_INFINITY)) {
    note(entry.index.objectOf.get(result.id), result.score);
  }
  const splats = (id: number): number => entry.index.objects.get(id)?.splats ?? 0;
  return [...best].sort((a, b) => b[1] - a[1] || splats(b[0]) - splats(a[0])).map(([id]) => id);
}

export const useInstances = create<InstancesState>()((set) => ({
  assets: {},
  dimOthers: true,
  gaps: {},
  setGap: (assetId, gap) =>
    set((s) => {
      const current = s.gaps[assetId];
      if (gap === null) {
        if (!current) return s;
        return {
          gaps: Object.fromEntries(Object.entries(s.gaps).filter(([id]) => id !== assetId)),
        };
      }
      if (current?.renderer === gap.renderer && current.reason === gap.reason) return s;
      return { gaps: { ...s.gaps, [assetId]: gap } };
    }),
  setTable: (assetId, table) =>
    set((s) => {
      const next: Record<string, AssetInstances> = {};
      for (const [id, entry] of Object.entries(s.assets)) if (id !== assetId) next[id] = entry;
      if (table && table.instances.length > 0) {
        next[assetId] = {
          instances: table.instances,
          index: indexCategories(table.instances),
          hidden: EMPTY,
          highlighted: EMPTY,
          focus: null,
          query: "",
          matches: [],
        };
      }
      return { assets: next };
    }),
  setQuery: (assetId, query) =>
    set((s) =>
      patch(s, assetId, (current) => {
        const matches = matchObjects(current, query);
        // A highlight of the old matches follows the query.
        if (current.focus?.kind === "matches") {
          const lit = focusMembers({ ...current, matches }, current.focus);
          return {
            query,
            matches,
            highlighted: lit.length ? new Set(lit) : EMPTY,
            focus: lit.length ? current.focus : null,
          };
        }
        return { query, matches };
      }),
    ),
  setHidden: (assetId, ids, hidden) =>
    set((s) =>
      patch(s, assetId, (current) => {
        const next = new Set(current.hidden);
        for (const id of withDescendants(current, ids)) {
          if (hidden) next.add(id);
          else next.delete(id);
        }
        return { hidden: next };
      }),
    ),
  setObjectsHidden: (assetId, objectIds, hidden) =>
    set((s) =>
      patch(s, assetId, (current) => {
        const next = new Set(current.hidden);
        for (const id of membersOf(current.index, objectIds)) {
          if (hidden) next.add(id);
          else next.delete(id);
        }
        return { hidden: next };
      }),
    ),
  setCategoryHidden: (assetId, categoryId, hidden) =>
    set((s) =>
      patch(s, assetId, (current) => {
        const group = current.index.groups.find((g) => g.category.id === categoryId);
        if (!group) return {};
        const next = new Set(current.hidden);
        for (const id of group.members) {
          if (hidden) next.add(id);
          else next.delete(id);
        }
        return { hidden: next };
      }),
    ),
  hideMatches: (assetId) =>
    set((s) =>
      patch(s, assetId, (current) => {
        if (current.matches.length === 0) return {};
        const next = new Set(current.hidden);
        for (const id of membersOf(current.index, current.matches)) next.add(id);
        return { hidden: next };
      }),
    ),
  showOnlyMatches: (assetId) =>
    set((s) =>
      patch(s, assetId, (current) => {
        if (current.matches.length === 0) return {};
        const keep = new Set(membersOf(current.index, current.matches));
        return {
          hidden: new Set(current.instances.filter((i) => !keep.has(i.id)).map((i) => i.id)),
        };
      }),
    ),
  showAll: (assetId) => set((s) => patch(s, assetId, () => ({ hidden: EMPTY }))),
  highlight: (assetId, ids) =>
    set((s) =>
      patch(s, assetId, (current) =>
        ids.length
          ? { highlighted: withDescendants(current, ids), focus: { kind: "ids" } }
          : { highlighted: EMPTY, focus: null },
      ),
    ),
  toggleFocus: (assetId, focus) =>
    set((s) =>
      patch(s, assetId, (current) => {
        if (focus === null || sameFocus(current.focus, focus)) {
          return { highlighted: EMPTY, focus: null };
        }
        const lit = focusMembers(current, focus);
        return lit.length
          ? { highlighted: new Set(lit), focus }
          : { highlighted: EMPTY, focus: null };
      }),
    ),
  reset: (assetId) =>
    set((s) => patch(s, assetId, () => ({ hidden: EMPTY, highlighted: EMPTY, focus: null }))),
  setDimOthers: (dimOthers) => set({ dimOthers }),
}));
