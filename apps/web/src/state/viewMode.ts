import { create } from "zustand";

/**
 * How splat scans are coloured: as photographed, by how well the capture measured each part
 * (Confidence, the `weak-v1` map), or by how old the data shown is (Data age). The analytic
 * views are drawn by the CesiumJS splat primitive only (lib/viewModes.ts `needsCesium`).
 *
 * Not persisted: an analytic tint is a question asked now, and a reload that kept it would
 * show a red scan to someone who never asked.
 */
export type ViewMode = "photo" | "confidence" | "data-age";

/** What the scene says about one model's analytic view (cesium/cellViewSlot.ts). */
export type CellViewReport =
  | { kind: "loading" }
  /** Its root declares no confidence grid: nothing measured how well it was captured. */
  | { kind: "not-measured" }
  | { kind: "failed"; message: string }
  | {
      kind: "ready";
      mode: Exclude<ViewMode, "photo">;
      /** Data age: the shown date's age bucket (lib/dataAge.ts), null when it has no date. */
      ageBucket?: number | null;
      /** Data age: the date is the upload's (`observedAtSource: "created"`), not the capture's. */
      ageFromUpload?: boolean;
      /** Data age: later dates of the site, and how many of them could be compared. */
      newer?: { dates: number; compared: number; cells: number; skipped: string[] };
    };

interface ViewModeState {
  mode: ViewMode;
  set: (mode: ViewMode) => void;
  /** Per asset id: whether its analytic view is drawn, and what it found. */
  cells: Record<string, CellViewReport>;
  report: (assetId: string, report: CellViewReport | null) => void;
}

export const useViewMode = create<ViewModeState>()((set) => ({
  mode: "photo",
  set: (mode) => set({ mode }),
  cells: {},
  report: (assetId, report) =>
    set((s) => ({
      cells: report
        ? { ...s.cells, [assetId]: report }
        : Object.fromEntries(Object.entries(s.cells).filter(([id]) => id !== assetId)),
    })),
}));
