import { create } from "zustand";
import type { LandCandidate, LandMapGeometry } from "@twin/contracts";

export interface LandContextFeature {
  id: string;
  label: string;
  geometry: LandMapGeometry;
  value?: number | null;
  /** A colour of its own (a soil unit, a land-cover class); the layer's otherwise. */
  color?: string;
}
/** How a lens draws its layer; absent, a layer keeps the neutral analysis blue. */
export interface LandContextStyle {
  color: string;
  /** Fill opacity for areas, 0–1. */
  fill?: number;
  /** Outline and line width in pixels. */
  width?: number;
}
export interface LandContextLayer {
  drawingGuideSources?: LandCandidate["source"][];
  /** A research map: made by the Land agent, so As captured draws none of it (`agentMade`). */
  researchArtifactId?: string;
  /** Made by the Land agent (its findings, where it is looking): see `agentMade`. */
  agent?: boolean;
  unit?: string | null;
  legend?: string;
  id: string;
  title: string;
  features: LandContextFeature[];
  selectedIds?: string[];
  style?: LandContextStyle;
}
/**
 * Whether the Land agent made a layer: its findings and where it is looking, or a research map
 * it produced. As captured draws none of it (ADR 0019); the layer is kept for after.
 */
export function agentMade(layer: LandContextLayer): boolean {
  return layer.agent === true || layer.researchArtifactId !== undefined;
}

export interface LandRasterLayer {
  kind?: "archive-alignment";
  categorical?: boolean;
  id: string;
  band: number;
  bounds: number[];
  attribution: string;
  opacity: number;
}
export type LandWorkspaceSection =
  "discover" | "records" | "inventory" | "ecology" | "scenarios" | "actions";
export interface LandContextState {
  researchFocus: { artifactId: string; featureIndex: number; label: string } | null;
  setResearchFocus: (focus: LandContextState["researchFocus"]) => void;
  selectedMapFeature: { layerId: string; featureId: string; origin: "map" | "list" } | null;
  selectMapFeature: (layerId: string, featureId: string, origin?: "map" | "list") => void;
  clearMapSelection: () => void;
  inventoryVisible: boolean;
  setInventoryVisible: (value: boolean) => void;
  pointPicker: string | null;
  setPointPicker: (id: string | null) => void;
  selectedInvestigationId: string | null;
  selectInvestigation: (id: string | null) => void;
  researchQuestion: string;
  setResearchQuestion: (question: string) => void;
  section: LandWorkspaceSection;
  setSection: (section: LandWorkspaceSection) => void;
  selectedSolarId: string | null;
  selectSolar: (id: string | null) => void;
  selectedSurveyId: string | null;
  selectSurvey: (id: string | null) => void;
  selectedInventoryId: string | null;
  selectInventory: (id: string | null) => void;
  layers: Record<string, LandContextLayer>;
  rasters: Record<string, LandRasterLayer>;
  rasterErrors: Record<string, string>;
  setRaster: (layer: LandRasterLayer) => void;
  removeRaster: (id: string) => void;
  setRasterError: (id: string, message: string) => void;
  candidates: LandCandidate[];
  selectedIds: string[];
  setCandidates: (candidates: LandCandidate[]) => void;
  /** Adds records found around a new point, keeping the ones already chosen. */
  addCandidates: (candidates: LandCandidate[]) => void;
  toggleCandidate: (id: string) => void;
  setLayer: (layer: LandContextLayer) => void;
  removeLayer: (id: string) => void;
  clear: () => void;
}

export const useLandContext = create<LandContextState>((set, get) => ({
  researchFocus: null,
  setResearchFocus: (researchFocus) => set({ researchFocus }),
  selectedMapFeature: null,
  selectMapFeature: (layerId, featureId, origin = "list") => {
    const state = get(),
      layer = state.layers[layerId];
    if (!layer?.researchArtifactId || !layer.features.some((feature) => feature.id === featureId))
      return;
    const previous = state.selectedMapFeature && state.layers[state.selectedMapFeature.layerId];
    set({
      selectedMapFeature: { layerId, featureId, origin },
      layers: {
        ...state.layers,
        ...(previous ? { [previous.id]: { ...previous, selectedIds: [] } } : {}),
        [layerId]: { ...layer, selectedIds: [featureId] },
      },
    });
  },
  clearMapSelection: () => {
    const state = get(),
      layer = state.selectedMapFeature && state.layers[state.selectedMapFeature.layerId];
    set({
      selectedMapFeature: null,
      ...(layer ? { layers: { ...state.layers, [layer.id]: { ...layer, selectedIds: [] } } } : {}),
    });
  },
  inventoryVisible: true,
  setInventoryVisible: (inventoryVisible) => set({ inventoryVisible }),
  pointPicker: null,
  setPointPicker: (pointPicker) => set({ pointPicker }),
  selectedInvestigationId: null,
  selectInvestigation: (selectedInvestigationId) => set({ selectedInvestigationId }),
  researchQuestion: "",
  setResearchQuestion: (researchQuestion) => set({ researchQuestion }),
  section: "discover",
  setSection: (section) => set({ section }),
  selectedSolarId: null,
  selectSolar: (selectedSolarId) => set({ selectedSolarId }),
  selectedSurveyId: null,
  selectSurvey: (selectedSurveyId) => set({ selectedSurveyId }),
  selectedInventoryId: null,
  selectInventory: (selectedInventoryId) =>
    set({ selectedInventoryId, ...(selectedInventoryId ? { section: "inventory" as const } : {}) }),
  layers: {},
  rasters: {},
  rasterErrors: {},
  setRaster: (layer) =>
    set((state) => ({
      rasters: Object.fromEntries([
        ...Object.entries(state.rasters)
          .filter(([id]) => id !== layer.id)
          .slice(-1),
        [layer.id, layer],
      ]),
      rasterErrors: Object.fromEntries(
        Object.entries(state.rasterErrors).filter(([id]) => id !== layer.id),
      ),
    })),
  removeRaster: (id) =>
    set((state) => ({
      rasters: Object.fromEntries(Object.entries(state.rasters).filter(([key]) => key !== id)),
      rasterErrors: Object.fromEntries(
        Object.entries(state.rasterErrors).filter(([key]) => key !== id),
      ),
    })),
  setRasterError: (id, message) =>
    set((state) => ({ rasterErrors: { ...state.rasterErrors, [id]: message } })),
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
  addCandidates: (found) =>
    set((state) => {
      const kept = state.candidates.filter((candidate) => state.selectedIds.includes(candidate.id));
      const ids = new Set(kept.map((candidate) => candidate.id));
      const candidates = [...kept, ...found.filter((candidate) => !ids.has(candidate.id))];
      return {
        candidates,
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
            selectedIds: state.selectedIds,
          },
        },
      };
    }),
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
  setLayer: (layer) =>
    set((state) => {
      const selection = state.selectedMapFeature;
      if (selection?.layerId !== layer.id)
        return { layers: { ...state.layers, [layer.id]: layer } };
      const retained = Boolean(
        layer.researchArtifactId &&
        layer.features.some((feature) => feature.id === selection.featureId),
      );
      return {
        selectedMapFeature: retained ? selection : null,
        layers: {
          ...state.layers,
          [layer.id]: { ...layer, selectedIds: retained ? [selection.featureId] : [] },
        },
      };
    }),
  removeLayer: (id) =>
    set((state) => ({
      layers: Object.fromEntries(Object.entries(state.layers).filter(([key]) => key !== id)),
      ...(id === "candidates" ? { candidates: [], selectedIds: [] } : {}),
      ...(state.selectedMapFeature?.layerId === id ? { selectedMapFeature: null } : {}),
    })),
  clear: () =>
    set({
      researchFocus: null,
      selectedMapFeature: null,
      inventoryVisible: true,
      pointPicker: null,
      layers: {},
      rasters: {},
      rasterErrors: {},
      candidates: [],
      selectedIds: [],
      selectedInventoryId: null,
      section: "discover",
      selectedSolarId: null,
      selectedSurveyId: null,
      selectedInvestigationId: null,
      researchQuestion: "",
    }),
}));
