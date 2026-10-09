import { create } from "zustand";
import type { LandCandidate, LandMapGeometry } from "@twin/contracts";

export interface LandContextFeature {
  id: string;
  label: string;
  geometry: LandMapGeometry;
  value?: number | null;
}
export interface LandContextLayer {
  id: string;
  title: string;
  features: LandContextFeature[];
  selectedIds?: string[];
}
interface LandContextState {
  selectedInventoryId: string | null;
  selectInventory: (id: string | null) => void;
  layers: Record<string, LandContextLayer>;
  candidates: LandCandidate[];
  selectedIds: string[];
  setCandidates: (candidates: LandCandidate[]) => void;
  toggleCandidate: (id: string) => void;
  setLayer: (layer: LandContextLayer) => void;
  removeLayer: (id: string) => void;
  clear: () => void;
}

export const useLandContext = create<LandContextState>((set, get) => ({
  selectedInventoryId: null,
  selectInventory: (selectedInventoryId) => set({ selectedInventoryId }),
  layers: {},
  candidates: [],
  selectedIds: [],
  setCandidates: (candidates) =>
    set((state) => ({
      candidates,
      selectedIds: [],
      layers: {
        ...state.layers,
        candidates: {
          id: "candidates",
          title: "Selection candidates",
          features: candidates.map((candidate) => ({
            id: candidate.id,
            label: candidate.label,
            geometry: candidate.geometry,
          })),
          selectedIds: [],
        },
      },
    })),
  toggleCandidate: (id) => {
    if (!get().candidates.some((candidate) => candidate.id === id)) return;
    set((state) => {
      const selectedIds = state.selectedIds.includes(id)
        ? state.selectedIds.filter((value) => value !== id)
        : [...state.selectedIds, id];
      const layer = state.layers.candidates;
      return {
        selectedIds,
        layers: layer ? { ...state.layers, candidates: { ...layer, selectedIds } } : state.layers,
      };
    });
  },
  setLayer: (layer) => set((state) => ({ layers: { ...state.layers, [layer.id]: layer } })),
  removeLayer: (id) =>
    set((state) => ({
      layers: Object.fromEntries(Object.entries(state.layers).filter(([key]) => key !== id)),
      ...(id === "candidates" ? { candidates: [], selectedIds: [] } : {}),
    })),
  clear: () => set({ layers: {}, candidates: [], selectedIds: [], selectedInventoryId: null }),
}));
