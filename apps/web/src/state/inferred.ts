import { create } from "zustand";

import type { InferredEvidence } from "@/lib/inferred";

interface InferredState {
  /** Whether inferred layers are drawn at all. On: the scan reads whole, and is labelled. */
  show: boolean;
  /** The inferred layers loaded beside each measured asset. */
  layers: Record<string, InferredEvidence[]>;
  setShow: (show: boolean) => void;
  setLayers: (assetId: string, layers: InferredEvidence[]) => void;
}

export const useInferred = create<InferredState>()((set) => ({
  show: true,
  layers: {},
  setShow: (show) => set({ show }),
  setLayers: (assetId, layers) =>
    set((s) => {
      const next: Record<string, InferredEvidence[]> = {};
      for (const [id, list] of Object.entries(s.layers)) if (id !== assetId) next[id] = list;
      if (layers.length > 0) next[assetId] = layers;
      return { layers: next };
    }),
}));
