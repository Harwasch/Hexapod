import { create } from "zustand";

export type ToolPanel = "layers" | "sites" | "measure" | "compare" | "bookmarks";
export type MeasureMode = "point" | "distance" | "area" | "height" | "elevation";

interface UiState {
  activePanel: ToolPanel | null;
  inspectorOpen: boolean;
  paletteOpen: boolean;
  settingsOpen: boolean;
  addDataOpen: boolean;
  aboutLayerId: string | null;
  measureMode: MeasureMode | null;
  exploreMode: boolean;
  compareActive: boolean;
  togglePanel: (panel: ToolPanel) => void;
  setPanel: (panel: ToolPanel | null) => void;
  setInspectorOpen: (open: boolean) => void;
  setPaletteOpen: (open: boolean) => void;
  setSettingsOpen: (open: boolean) => void;
  setAddDataOpen: (open: boolean) => void;
  setAboutLayerId: (id: string | null) => void;
  setMeasureMode: (mode: MeasureMode | null) => void;
  setExploreMode: (on: boolean) => void;
  setCompareActive: (on: boolean) => void;
}

export const useUi = create<UiState>()((set) => ({
  activePanel: null,
  inspectorOpen: false,
  paletteOpen: false,
  settingsOpen: false,
  addDataOpen: false,
  aboutLayerId: null,
  measureMode: null,
  exploreMode: false,
  compareActive: false,
  togglePanel: (panel) => set((s) => ({ activePanel: s.activePanel === panel ? null : panel })),
  setPanel: (panel) => set({ activePanel: panel }),
  setInspectorOpen: (inspectorOpen) => set({ inspectorOpen }),
  setPaletteOpen: (paletteOpen) => set({ paletteOpen }),
  setSettingsOpen: (settingsOpen) => set({ settingsOpen }),
  setAddDataOpen: (addDataOpen) => set({ addDataOpen }),
  setAboutLayerId: (aboutLayerId) => set({ aboutLayerId }),
  setMeasureMode: (measureMode) =>
    set((s) => ({ measureMode, activePanel: measureMode ? "measure" : s.activePanel })),
  setExploreMode: (exploreMode) => set({ exploreMode }),
  setCompareActive: (compareActive) => set({ compareActive }),
}));
