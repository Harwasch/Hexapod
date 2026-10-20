/**
 * One companion list per splat primitive, several owners.
 *
 * The engine patch draws other tilesets' tiles in a primitive's own texture and sort from one
 * array, `primitive.companions`. More than one thing puts tilesets there: each scan's inferred
 * layers (`inferredLayers.ts`), and the per-cell view rules, which draw a site's other dates
 * on the frame anchor's primitive (`compositorSession.ts`). Each owner says only its own list
 * (`setCompanions`); the primitive's array is the lists merged in the order the owners first
 * joined, assigned only when it changes, so one owner never drops another's tilesets.
 *
 * `from` says what an owner's tilesets travel with (an inferred layer's scan): the view rules
 * read it (`companionGroups`) to know whose date a layer is.
 */

import type { Cesium3DTileset } from "cesium";

/** The patched primitive's companion array, as far as this module goes. */
export interface CompanionHost {
  companions?: Cesium3DTileset[];
  isDestroyed?(): boolean;
}

/** One owner's tilesets on a host, and what they travel with. */
export interface CompanionGroup {
  owner: symbol;
  tilesets: readonly Cesium3DTileset[];
  from: unknown;
}

const GROUPS = new WeakMap<CompanionHost, Map<symbol, CompanionGroup>>();

/**
 * Sets `owner`'s companions on `host` (none: `[]`), keeping every other owner's. The host's
 * array is reassigned only when the merged list differs.
 */
export function setCompanions(
  host: CompanionHost,
  owner: symbol,
  tilesets: readonly Cesium3DTileset[],
  from?: unknown,
): void {
  let groups = GROUPS.get(host);
  if (!groups) GROUPS.set(host, (groups = new Map<symbol, CompanionGroup>()));
  if (tilesets.length > 0) groups.set(owner, { owner, tilesets: [...tilesets], from });
  else groups.delete(owner);
  if (host.isDestroyed?.() === true) return;
  const merged = [...groups.values()].flatMap((g) => g.tilesets);
  const now = host.companions ?? [];
  if (now.length !== merged.length || merged.some((t, i) => now[i] !== t)) host.companions = merged;
}

/** The owners' lists on `host`, in merge order. */
export function companionGroups(host: CompanionHost | undefined): CompanionGroup[] {
  return host ? [...(GROUPS.get(host)?.values() ?? [])] : [];
}
