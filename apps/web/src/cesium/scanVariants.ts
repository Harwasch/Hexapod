/**
 * Tells the viewer which method variants a loaded scan offers (lib/variants.ts), for the
 * "Compare methods" panel (features/sites/CompareMethods.tsx): read off the measured tileset's
 * root once it loads, cleared when it unloads. What each system draws for the pick is its own
 * drawer's business (`attachInstances`, `attachInferredLayers`, `attachSkin`).
 */
import type { Cesium3DTileset } from "cesium";

import { hasVariants, todayOf, variantsOf } from "@/lib/variants";
import { useVariants } from "@/state/variants";

/** Puts `tileset`'s variants in the store under `assetId`; returns the remover. */
export function attachVariants(tileset: Cesium3DTileset, assetId: string): () => void {
  const extras = (tileset.root as { extras?: unknown } | undefined)?.extras;
  const declared = variantsOf(extras);
  if (!hasVariants(declared)) return () => undefined;
  const variants = { ...declared, today: todayOf(extras) };
  useVariants.getState().setOffered(assetId, variants);
  return () => {
    if (useVariants.getState().offered[assetId] === variants) {
      useVariants.getState().setOffered(assetId, null);
    }
  };
}
