import { create } from "zustand";

/**
 * Which per-cell drape has the ground (`cesium/GroundGridLayer.ts`): one at a time, so the
 * fog and a similarity grid never sit on each other (or compete for a tap). Claiming takes
 * it; the layer that held it hides itself.
 */
interface GridDrapeState {
  owner: string | null;
  claim: (owner: string) => void;
  release: (owner: string) => void;
}

export const useGridDrape = create<GridDrapeState>()((set, get) => ({
  owner: null,
  claim: (owner) => {
    if (get().owner !== owner) set({ owner });
  },
  release: (owner) => {
    if (get().owner === owner) set({ owner: null });
  },
}));
