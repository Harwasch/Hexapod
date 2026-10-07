/**
 * Which measured splats each scan's chosen fill method supersedes (lib/supersedes.ts), now.
 *
 * A scan's entry is set only while a fill variant that names a `supersedes` file is picked
 * (state/variants.ts) and Inferred is Show or Highlight (state/settings.ts); Hide, Today, a
 * method without the file, or one whose file did not load, leave none -- the untouched scan.
 * Switching methods clears the entry before the next one loads, so another method's splats are
 * never hidden under this one. `effectiveDoc` (state/sceneSelect.ts) draws from it.
 *
 * A scan without object ids (no `instances.json`, nor an objects variant) has no id path for
 * its splats: the swap does nothing there, and the fill is drawn over the whole scan as before.
 */

import { create } from "zustand";

import { createLogger } from "@/lib/log";
import { loadSupersedes, type SupersedesDoc } from "@/lib/supersedes";
import type { ScanVariants } from "@/lib/variants";
import { useSettings } from "@/state/settings";
import { onPickChange, pickedVariant } from "@/state/variants";

const log = createLogger("supersedes");

interface SupersedesState {
  /** Per asset id: the superseded splats drawn hidden now. */
  active: Readonly<Record<string, SupersedesDoc>>;
  setActive(assetId: string, doc: SupersedesDoc | null): void;
}

export const useSupersedes = create<SupersedesState>()((set) => ({
  active: {},
  setActive: (assetId, doc) =>
    set((state) => {
      if ((state.active[assetId] ?? null) === doc) return state;
      const rest = Object.fromEntries(Object.entries(state.active).filter(([k]) => k !== assetId));
      return { active: doc ? { ...rest, [assetId]: doc } : rest };
    }),
}));

/** The splats `assetId`'s chosen fill supersedes now, or null. */
export function activeSupersedes(assetId: string): SupersedesDoc | null {
  return useSupersedes.getState().active[assetId] ?? null;
}

/** Calls `listener` whenever `assetId`'s superseded splats change. Returns the unsubscriber. */
export function onSupersedesChange(assetId: string, listener: () => void): () => void {
  return useSupersedes.subscribe((state, previous) => {
    if (state.active[assetId] !== previous.active[assetId]) listener();
  });
}

export type LoadSupersedes = (tilesetUrl: string, uri: string) => Promise<SupersedesDoc>;

/**
 * Keeps `assetId`'s entry as its fill pick and the Inferred style say (see the module comment),
 * loading each method's file once. Returns the disposer, which clears the entry.
 */
export function followSupersedes(
  assetId: string,
  variants: ScanVariants,
  tilesetUrl: string,
  load: LoadSupersedes = loadSupersedes,
): () => void {
  const store = useSupersedes;
  const loaded = new Map<string, Promise<SupersedesDoc>>();
  let wanted: string | null = null;
  let disposed = false;
  const follow = (): void => {
    const uri = pickedVariant(assetId, "fill", variants)?.supersedes ?? null;
    const next = useSettings.getState().inferredStyle === "hide" ? null : uri;
    if (next === wanted) return;
    wanted = next;
    store.getState().setActive(assetId, null);
    if (next === null) return;
    let pending = loaded.get(next);
    if (!pending) {
      pending = load(tilesetUrl, next);
      loaded.set(next, pending);
    }
    pending.then(
      (doc) => {
        if (disposed || wanted !== next) return;
        store.getState().setActive(assetId, doc);
        log.info("fill supersedes measured splats", {
          asset: assetId,
          splats: doc.superseded,
          tiles: doc.tiles.size,
          issues: doc.issues.length,
        });
      },
      (error: unknown) => {
        loaded.delete(next);
        if (disposed || wanted !== next) return;
        // Drawn over the whole scan, as before the swap: nothing hidden that should not be.
        log.warn("supersedes did not load; the fill is drawn over the scan", {
          asset: assetId,
          error: error instanceof Error ? error.message : String(error),
        });
      },
    );
  };
  const offPick = onPickChange(assetId, "fill", follow);
  const offStyle = useSettings.subscribe((state, previous) => {
    if (state.inferredStyle !== previous.inferredStyle) follow();
  });
  follow();
  return () => {
    disposed = true;
    offPick();
    offStyle();
    store.getState().setActive(assetId, null);
  };
}
