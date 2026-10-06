import { create } from "zustand";

import type { InferredEvidence } from "@/lib/inferred";

/**
 * The inferred layers loaded beside each measured asset (cesium/inferredLayers.ts), for the
 * switcher to say what they are. How they are drawn -- Show, Highlight or Hide -- is a
 * viewer's setting (`useSettings` `inferredStyle`).
 */
interface InferredState {
  /** The inferred layers loaded beside each measured asset. */
  layers: Record<string, InferredEvidence[]>;
  setLayers: (assetId: string, layers: InferredEvidence[]) => void;
}

export const useInferred = create<InferredState>()((set) => ({
  layers: {},
  setLayers: (assetId, layers) =>
    set((s) => {
      if (layers.length === 0 && !(assetId in s.layers)) return s;
      const next: Record<string, InferredEvidence[]> = {};
      for (const [id, list] of Object.entries(s.layers)) if (id !== assetId) next[id] = list;
      if (layers.length > 0) next[assetId] = layers;
      return { layers: next };
    }),
}));
