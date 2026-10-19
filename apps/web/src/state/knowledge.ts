import { create } from "zustand";

/**
 * The Fog of Knowing on the map: whether it shows, and the cell whose card is open. Shown or
 * not is remembered for the browser tab (sessionStorage, when it is available); the cell
 * never is.
 */
export interface KnowledgeCellSelection {
  landId: string;
  longitude: number;
  latitude: number;
  col: number;
  row: number;
}

interface KnowledgeState {
  visible: boolean;
  selected: KnowledgeCellSelection | null;
  /**
   * A point asked about before its cell is known (the map menu's "What do we know here"):
   * the fog's bridge turns it into `selected` once the grid is in.
   */
  pending: { landId: string; longitude: number; latitude: number } | null;
  setVisible: (visible: boolean) => void;
  select: (selected: KnowledgeCellSelection) => void;
  /** Shows the fog and opens the card for the cell at a point. */
  selectAt: (landId: string, longitude: number, latitude: number) => void;
  clear: () => void;
}

const STORAGE_KEY = "twin.knowledge.visible";

function storedVisible(): boolean {
  try {
    return globalThis.sessionStorage?.getItem(STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

function storeVisible(visible: boolean): void {
  try {
    globalThis.sessionStorage?.setItem(STORAGE_KEY, visible ? "1" : "0");
  } catch {
    // Private windows and blocked storage: the fog still shows, it is just not remembered.
  }
}

export const useKnowledge = create<KnowledgeState>()((set) => ({
  visible: storedVisible(),
  selected: null,
  pending: null,
  setVisible: (visible) => {
    storeVisible(visible);
    set((s) => ({
      visible,
      selected: visible ? s.selected : null,
      pending: visible ? s.pending : null,
    }));
  },
  select: (selected) => set({ selected, pending: null }),
  selectAt: (landId, longitude, latitude) => {
    storeVisible(true);
    set({ visible: true, pending: { landId, longitude, latitude } });
  },
  clear: () => set({ selected: null, pending: null }),
}));
