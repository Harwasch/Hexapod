import { create } from "zustand";

import type { Attribution } from "@twin/contracts";

export type SelectionKind = "ground" | "site" | "layer-feature" | "tile-feature";

export interface SelectionProperty {
  key: string;
  value: string;
}

export interface Selection {
  kind: SelectionKind;
  title: string;
  longitude: number;
  latitude: number;
  height: number | null;
  /** Terrain elevation sampled at the point, when available. */
  terrainHeight: number | null;
  siteId?: string;
  layerId?: string;
  sourceLabel?: string;
  observedAt?: string | null;
  attribution?: Attribution[];
  properties?: SelectionProperty[];
  at: number;
}

interface SelectionState {
  selection: Selection | null;
  hoverInfo: string | null;
  setSelection: (selection: Selection | null) => void;
  setHoverInfo: (hoverInfo: string | null) => void;
}

export const useSelection = create<SelectionState>()((set) => ({
  selection: null,
  hoverInfo: null,
  setSelection: (selection) => set({ selection }),
  setHoverInfo: (hoverInfo) => set({ hoverInfo }),
}));
