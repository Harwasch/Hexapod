import { create } from "zustand";

import {
  quickFilters,
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
  results: SearchResult[];
}

interface InstancesState {
  /** Per asset id, present while the asset's tileset declares instances and they loaded. */
  assets: Record<string, AssetInstances>;
  /** Whether everything but a highlight is dimmed while one is active. */
  dimOthers: boolean;
  setTable: (
    assetId: string,
    table: { instances: Instance[]; propertyNames: string[] } | null,
  ) => void;
  setQuery: (assetId: string, query: string) => void;
  setHidden: (assetId: string, ids: readonly number[], hidden: boolean) => void;
  toggleHidden: (assetId: string, id: number) => void;
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
        };
      }
      return { assets: next };
    }),
  setQuery: (assetId, query) =>
    set((s) =>
      patch(s, assetId, (current) => ({
        query,
        results: searchInstances(current.instances, query),
      })),
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
  showAll: (assetId) => set((s) => patch(s, assetId, () => ({ hidden: EMPTY }))),
  highlight: (assetId, ids) =>
    set((s) => patch(s, assetId, () => ({ highlighted: ids.length ? new Set(ids) : EMPTY }))),
  setDimOthers: (dimOthers) => set({ dimOthers }),
}));
