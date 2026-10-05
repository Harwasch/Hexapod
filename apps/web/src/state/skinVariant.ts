import { create } from "zustand";

/**
 * Which skin each scan's objects move by: the scan's own `extras.skin` ("Today", the default)
 * or one of the motion-skins bake-off's candidates declared in `extras.variants.skins`
 * (docs/SCENE_OBJECTS.md §9). `cesium/splatSkin.ts` `attachSkin` follows it: a change detaches
 * the skin drawn now and loads the chosen one, so the wind and the poke move the new one.
 *
 * The "Compare methods" switcher sets it per scan (`select`). Until that lands, the temporary
 * URL parameter `?skinVariant=<name>` picks that candidate on every scan that declares it --
 * for testing the bake-off; it changes nothing on a scan without that variant.
 */

/** The temporary URL parameter (see the module comment). */
export const SKIN_VARIANT_PARAM = "skinVariant";

/** One candidate skin, as `extras.variants.skins` declares it. */
export interface SkinVariant {
  name: string;
  label: string;
  about: string;
  /** `skin.json`, relative to the tileset. */
  skin: string;
}

/** `extras.variants.skins` off a tileset's root extras; empty when absent or malformed. */
export function skinVariantsOf(extras: unknown): SkinVariant[] {
  const list = (extras as { variants?: { skins?: unknown } } | null | undefined)?.variants?.skins;
  if (!Array.isArray(list)) return [];
  const out: SkinVariant[] = [];
  for (const entry of list as unknown[]) {
    const e = entry as Record<string, unknown> | null;
    if (typeof e?.name !== "string" || typeof e.skin !== "string" || e.skin.length === 0) continue;
    out.push({
      name: e.name,
      label: typeof e.label === "string" ? e.label : e.name,
      about: typeof e.about === "string" ? e.about : "",
      skin: e.skin,
    });
  }
  return out;
}

/** The candidate the URL asks for (`?skinVariant=`), or null. */
export function requestedSkinVariant(search = globalThis.location?.search ?? ""): string | null {
  const asked = new URLSearchParams(search).get(SKIN_VARIANT_PARAM)?.trim();
  return asked === undefined || asked === "" ? null : asked;
}

interface SkinVariantState {
  /** Per scan (asset id): the candidate chosen, or null for the scan's own skin. */
  chosen: Readonly<Record<string, string | null>>;
  /** Chooses a candidate for a scan (null: its own skin). */
  select: (assetId: string, name: string | null) => void;
}

export const useSkinVariant = create<SkinVariantState>((set) => ({
  chosen: {},
  select: (assetId, name) =>
    set((state) =>
      state.chosen[assetId] === name ? state : { chosen: { ...state.chosen, [assetId]: name } },
    ),
}));

/**
 * The candidate a scan should draw now: the one chosen for it, else the URL's, else none --
 * and only one the scan declares.
 */
export function skinVariantFor(
  assetId: string,
  extras: unknown,
  search?: string,
): SkinVariant | null {
  const chosen = useSkinVariant.getState().chosen;
  const name = assetId in chosen ? (chosen[assetId] ?? null) : requestedSkinVariant(search);
  if (name === null) return null;
  return skinVariantsOf(extras).find((v) => v.name === name) ?? null;
}
