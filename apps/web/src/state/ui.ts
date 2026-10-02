import { create } from "zustand";

export type ToolPanel = "layers" | "sites" | "captures" | "measure" | "compare" | "bookmarks";
export type MeasureMode = "point" | "distance" | "area" | "height" | "elevation";

interface UiState {
  activePanel: ToolPanel | null;
  inspectorOpen: boolean;
  /** The keyboard shortcut sheet (`?`). */
  shortcutsOpen: boolean;
  /**
   * The agent's activity log, above the status line. Only the operator opens it (its chevron,
   * `a`, the command box): the status line already says what the agent is doing and its
   * latest reply, so nothing that happens on its own puts thirteen lines over the map.
   */
  activityOpen: boolean;
  settingsOpen: boolean;
  addDataOpen: boolean;
  aboutLayerId: string | null;
  measureMode: MeasureMode | null;
  exploreMode: boolean;
  compareActive: boolean;
  /**
   * A write came back 401, so the API has a token configured and this browser does not
   * have it. Never set up front: a deployment without `API_WRITE_TOKEN` leaves writes
   * open, and prompting there would invent a step that does not exist.
   */
  writeTokenPrompt: boolean;
  togglePanel: (panel: ToolPanel) => void;
  setPanel: (panel: ToolPanel | null) => void;
  setInspectorOpen: (open: boolean) => void;
  setShortcutsOpen: (open: boolean) => void;
  setActivityOpen: (open: boolean) => void;
  setSettingsOpen: (open: boolean) => void;
  setAddDataOpen: (open: boolean) => void;
  setAboutLayerId: (id: string | null) => void;
  setMeasureMode: (mode: MeasureMode | null) => void;
  setExploreMode: (on: boolean) => void;
  setCompareActive: (on: boolean) => void;
  setWriteTokenPrompt: (open: boolean) => void;
}

export const useUi = create<UiState>()((set) => ({
  activePanel: null,
  inspectorOpen: false,
  shortcutsOpen: false,
  activityOpen: false,
  settingsOpen: false,
  addDataOpen: false,
  aboutLayerId: null,
  measureMode: null,
  exploreMode: false,
  compareActive: false,
  writeTokenPrompt: false,
  togglePanel: (panel) => set((s) => ({ activePanel: s.activePanel === panel ? null : panel })),
  setPanel: (panel) => set({ activePanel: panel }),
  setInspectorOpen: (inspectorOpen) => set({ inspectorOpen }),
  setShortcutsOpen: (shortcutsOpen) => set({ shortcutsOpen }),
  setActivityOpen: (activityOpen) => set({ activityOpen }),
  setSettingsOpen: (settingsOpen) => set({ settingsOpen }),
  setAddDataOpen: (addDataOpen) => set({ addDataOpen }),
  setAboutLayerId: (aboutLayerId) => set({ aboutLayerId }),
  setMeasureMode: (measureMode) =>
    set((s) => ({ measureMode, activePanel: measureMode ? "measure" : s.activePanel })),
  setExploreMode: (exploreMode) => set({ exploreMode }),
  setCompareActive: (compareActive) => set({ compareActive }),
  setWriteTokenPrompt: (writeTokenPrompt) => set({ writeTokenPrompt }),
}));
