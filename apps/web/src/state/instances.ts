import { create } from "zustand";

import {
  hiddenForOnly,
  quickFilters,
  RESULT_LIMIT,
  searchInstances,
  type Instance,
  type QuickFilter,
  type SearchResult,
} from "@/lib/instances";

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
  /** The first `RESULT_LIMIT` matches, listed. */
  results: SearchResult[];
  /** Every match's id, best first: what "hide all matches" and "show only matches" act on. */
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
  /** Whether everything but a highlight is dimmed while one is active. */
  dimOthers: boolean;
  /**
   * Per asset id, set while the splat renderer drawing it cannot apply hide and highlight
   * (`cesium/scanView`): the panel says so and offers the CesiumJS renderer.
   */
  gaps: Record<string, RendererGap>;
  setGap: (assetId: string, gap: RendererGap | null) => void;
  setTable: (
    assetId: string,
    table: { instances: Instance[]; propertyNames: string[] } | null,
  ) => void;
  setQuery: (assetId: string, query: string) => void;
  setHidden: (assetId: string, ids: readonly number[], hidden: boolean) => void;
  toggleHidden: (assetId: string, id: number) => void;
  /** Hides every match of the current query, not only the ones listed. */
  hideMatches: (assetId: string) => void;
  /** Hides everything but the current query's matches (and what they contain). */
  showOnlyMatches: (assetId: string) => void;
  showAll: (assetId: string) => void;
  /** Replaces the highlight; an empty list clears it. */
  highlight: (assetId: string, ids: readonly number[]) => void;
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
          propertyNames: table.propertyNames,
          filters: quickFilters(table),
          hidden: EMPTY,
          highlighted: EMPTY,
          query: "",
          results: [],
          matches: [],
        };
      }
      return { assets: next };
    }),
  setQuery: (assetId, query) =>
    set((s) =>
      patch(s, assetId, (current) => {
        const all = searchInstances(current.instances, query, Number.POSITIVE_INFINITY);
        return {
          query,
          results: all.slice(0, RESULT_LIMIT),
          matches: all.map((r) => r.id),
        };
      }),
    ),
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
  hideMatches: (assetId) =>
    set((s) =>
      patch(s, assetId, (current) => {
        if (current.matches.length === 0) return {};
        const next = new Set(current.hidden);
        for (const id of current.matches) next.add(id);
        return { hidden: next };
      }),
    ),
  showOnlyMatches: (assetId) =>
    set((s) =>
      patch(s, assetId, (current) =>
        current.matches.length === 0
          ? {}
          : { hidden: new Set(hiddenForOnly(current, current.matches)) },
      ),
    ),
  showAll: (assetId) => set((s) => patch(s, assetId, () => ({ hidden: EMPTY }))),
  highlight: (assetId, ids) =>
    set((s) => patch(s, assetId, () => ({ highlighted: ids.length ? new Set(ids) : EMPTY }))),
  setDimOthers: (dimOthers) => set({ dimOthers }),
}));
