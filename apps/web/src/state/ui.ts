import { create } from "zustand";

/**
 * The left rail's tool panels. Four tools sit on the rail — Layers, Measure, Add and Settings —
 * and Settings is a sheet, so three of them open a panel. What used to be its own tool lives
 * inside one of them: Compare is a mode of Layers, Captures (with the phone handoff) and
 * "Add data" are the two tabs of Add, and Sites and Saved views are in the site switcher.
 */
export type ToolPanel = "layers" | "measure" | "add" | "land";
export type MeasureMode = "point" | "distance" | "area" | "height" | "elevation";
/** The Layers panel's two modes: the catalog, or a swipe between two layers. */
export type LayersMode = "browse" | "compare";
/** The Add panel's two tabs: upload a capture, or link a source that is already hosted. */
export type AddTab = "upload" | "link";
/** Where the site switcher puts the keyboard when it opens. */
export type SwitcherFocus = "sites" | "views";
/**
 * Where the map's menu is open (`features/map/MapContextMenu`): the point in CSS px from the
 * canvas's top left, and the ground there.
 */
export interface MapMenuAt {
  x: number;
  y: number;
  longitude: number;
  latitude: number;
  height: number;
}

interface UiState {
  activePanel: ToolPanel | null;
  layersMode: LayersMode;
  addTab: AddTab;
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
  /** The phone's "More" sheet above the tab bar: Layers, Measure, Add, Settings. */
  moreOpen: boolean;
  /** Which part of the site switcher takes focus when it opens (`s` sites, `v` saved views). */
  switcherFocus: SwitcherFocus;
  aboutLayerId: string | null;
  /** The map's menu (a right-click or long press on the map), or null while it is closed. */
  mapMenu: MapMenuAt | null;
  measureMode: MeasureMode | null;
  exploreMode: boolean;
  /** The swipe comparison, kept here so it outlives the Layers panel that sets it up. */
  compareActive: boolean;
  compareLeft: string;
  compareRight: string;
  /**
   * A write came back 401, so the API has a token configured and this browser does not
   * have it. Never set up front: a deployment without `API_WRITE_TOKEN` leaves writes
   * open, and prompting there would invent a step that does not exist.
   */
  writeTokenPrompt: boolean;
  togglePanel: (panel: ToolPanel) => void;
  setPanel: (panel: ToolPanel | null) => void;
  /** Opens Layers in a mode; pressing the same mode's key again closes it. */
  openLayers: (mode: LayersMode) => void;
  setLayersMode: (mode: LayersMode) => void;
  /** Opens Add on a tab. */
  openAdd: (tab: AddTab) => void;
  setAddTab: (tab: AddTab) => void;
  setInspectorOpen: (open: boolean) => void;
  setShortcutsOpen: (open: boolean) => void;
  setActivityOpen: (open: boolean) => void;
  setSettingsOpen: (open: boolean) => void;
  setMoreOpen: (open: boolean) => void;
  setSwitcherFocus: (focus: SwitcherFocus) => void;
  setAboutLayerId: (id: string | null) => void;
  setMapMenu: (at: MapMenuAt | null) => void;
  setMeasureMode: (mode: MeasureMode | null) => void;
  setExploreMode: (on: boolean) => void;
  setCompareActive: (on: boolean) => void;
  setCompareLayers: (layers: { left?: string; right?: string }) => void;
  setWriteTokenPrompt: (open: boolean) => void;
}

export const useUi = create<UiState>()((set) => ({
  activePanel: null,
  layersMode: "browse",
  addTab: "upload",
  inspectorOpen: false,
  shortcutsOpen: false,
  activityOpen: false,
  settingsOpen: false,
  moreOpen: false,
  switcherFocus: "sites",
  aboutLayerId: null,
  mapMenu: null,
  measureMode: null,
  exploreMode: false,
  compareActive: false,
  compareLeft: "",
  compareRight: "",
  writeTokenPrompt: false,
  togglePanel: (panel) => set((s) => ({ activePanel: s.activePanel === panel ? null : panel })),
  setPanel: (panel) => set({ activePanel: panel }),
  openLayers: (mode) =>
    set((s) => ({
      activePanel: s.activePanel === "layers" && s.layersMode === mode ? null : "layers",
      layersMode: mode,
    })),
  setLayersMode: (layersMode) => set({ layersMode }),
  openAdd: (addTab) => set({ activePanel: "add", addTab }),
  setAddTab: (addTab) => set({ addTab }),
  setInspectorOpen: (inspectorOpen) => set({ inspectorOpen }),
  setShortcutsOpen: (shortcutsOpen) => set({ shortcutsOpen }),
  setActivityOpen: (activityOpen) => set({ activityOpen }),
  setSettingsOpen: (settingsOpen) => set({ settingsOpen }),
  setMoreOpen: (moreOpen) => set({ moreOpen }),
  setSwitcherFocus: (switcherFocus) => set({ switcherFocus }),
  setAboutLayerId: (aboutLayerId) => set({ aboutLayerId }),
  setMapMenu: (mapMenu) => set({ mapMenu }),
  setMeasureMode: (measureMode) =>
    set((s) => ({ measureMode, activePanel: measureMode ? "measure" : s.activePanel })),
  setExploreMode: (exploreMode) => set({ exploreMode }),
  setCompareActive: (compareActive) => set({ compareActive }),
  setCompareLayers: ({ left, right }) =>
    set((s) => ({ compareLeft: left ?? s.compareLeft, compareRight: right ?? s.compareRight })),
  setWriteTokenPrompt: (writeTokenPrompt) => set({ writeTokenPrompt }),
}));
