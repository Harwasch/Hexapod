import { create } from "zustand";

import {
  findVariant,
  hasVariants,
  NO_VARIANTS,
  VARIANT_SYSTEMS,
  type ScanVariants,
  type VariantOf,
  type VariantSystem,
} from "@/lib/variants";

/** Where a system's pick stands: its files loading, drawn, or not loaded (and why). */
export type VariantStatus =
  { state: "loading" } | { state: "ready" } | { state: "error"; message: string };

/** Per system, the name of the variant picked; a system absent is Today. */
export type VariantPicks = Partial<Record<VariantSystem, string>>;

/** What a loaded scan offers: its variants, and per system whether it has its own (Today). */
export interface OfferedVariants extends ScanVariants {
  today: Record<VariantSystem, boolean>;
}

interface VariantsState {
  /** Per asset id, the variants its loaded tileset declares (none: absent). */
  offered: Record<string, OfferedVariants>;
  /**
   * Per asset id, what is picked per system. Kept for the session (this tab, across scans
   * and reloads), never saved beyond it: a bake-off is looked at, not a setting.
   */
  picks: Record<string, VariantPicks>;
  /** Per asset id, per system, how the pick's files are doing (the drawers report it). */
  status: Record<string, Partial<Record<VariantSystem, VariantStatus>>>;
  setOffered: (assetId: string, variants: OfferedVariants | null) => void;
  /** Picks variant `name` of `system` for `assetId`, or Today (`null`). */
  pick: (assetId: string, system: VariantSystem, name: string | null) => void;
  setStatus: (assetId: string, system: VariantSystem, status: VariantStatus | null) => void;
}

/** Where the picks are kept for the session. */
export const PICKS_KEY = "hexapod.variants.picks";

function readPicks(): Record<string, VariantPicks> {
  try {
    const raw: unknown = JSON.parse(globalThis.sessionStorage?.getItem(PICKS_KEY) ?? "{}");
    if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return {};
    const out: Record<string, VariantPicks> = {};
    for (const [assetId, picks] of Object.entries(raw as Record<string, unknown>)) {
      if (typeof picks !== "object" || picks === null) continue;
      const kept: VariantPicks = {};
      for (const system of VARIANT_SYSTEMS) {
        const name = (picks as Record<string, unknown>)[system];
        if (typeof name === "string" && name !== "") kept[system] = name;
      }
      if (Object.keys(kept).length > 0) out[assetId] = kept;
    }
    return out;
  } catch {
    return {};
  }
}

function writePicks(picks: Record<string, VariantPicks>): void {
  try {
    globalThis.sessionStorage?.setItem(PICKS_KEY, JSON.stringify(picks));
  } catch {
    // Storage may be blocked: the picks then last as long as the page.
  }
}

/** `record` without `key`, or `record` itself when it has none. */
function without<T>(record: Record<string, T>, key: string): Record<string, T> {
  if (!(key in record)) return record;
  return Object.fromEntries(Object.entries(record).filter(([k]) => k !== key));
}

/** `record` with `key` set to `value`, or without it for `undefined`. */
function withEntry<K extends string, T>(
  record: Partial<Record<K, T>>,
  key: K,
  value: T | undefined,
): Partial<Record<K, T>> {
  const rest = Object.fromEntries(Object.entries(record).filter(([k]) => k !== key)) as Partial<
    Record<K, T>
  >;
  return value === undefined ? rest : { ...rest, [key]: value };
}

export const useVariants = create<VariantsState>()((set) => ({
  offered: {},
  picks: readPicks(),
  status: {},
  setOffered: (assetId, variants) =>
    set((s) => {
      if (variants === null || !hasVariants(variants)) {
        const offered = without(s.offered, assetId);
        const status = without(s.status, assetId);
        return offered === s.offered && status === s.status ? s : { offered, status };
      }
      return { offered: { ...s.offered, [assetId]: variants } };
    }),
  pick: (assetId, system, name) =>
    set((s) => {
      const current = s.picks[assetId] ?? {};
      if ((current[system] ?? null) === name) return s;
      const next: VariantPicks = withEntry(current, system, name ?? undefined);
      const picks =
        Object.keys(next).length > 0 ? { ...s.picks, [assetId]: next } : without(s.picks, assetId);
      writePicks(picks);
      return { picks };
    }),
  setStatus: (assetId, system, status) =>
    set((s) => {
      const current = s.status[assetId] ?? {};
      const was = current[system];
      if (status === null ? was === undefined : JSON.stringify(was) === JSON.stringify(status)) {
        return s;
      }
      const next = withEntry(current, system, status ?? undefined);
      return {
        status:
          Object.keys(next).length > 0
            ? { ...s.status, [assetId]: next }
            : without(s.status, assetId),
      };
    }),
}));

/**
 * The variant of `system` drawn for `assetId` now: the one picked, when `variants` (the scan's
 * own declaration, by default what the store was told) still offers it; null is Today.
 */
export function pickedVariant<S extends VariantSystem>(
  assetId: string,
  system: S,
  variants: ScanVariants = useVariants.getState().offered[assetId] ?? NO_VARIANTS,
): VariantOf<S> | null {
  return findVariant(variants, system, useVariants.getState().picks[assetId]?.[system]);
}

/** Calls `listener` whenever `assetId`'s pick for `system` changes. Returns the unsubscriber. */
export function onPickChange(
  assetId: string,
  system: VariantSystem,
  listener: () => void,
): () => void {
  return useVariants.subscribe((state, previous) => {
    if (state.picks[assetId]?.[system] !== previous.picks[assetId]?.[system]) listener();
  });
}
