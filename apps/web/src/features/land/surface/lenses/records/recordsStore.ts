import { create } from "zustand";

import { useLandContext } from "@/state/landContext";

import { useLens } from "../../lenses";

/** What the Records card shows: the summary, or one thing opened from it or from the map. */
export type RecordsView =
  | { kind: "summary" }
  | { kind: "archive"; id: string }
  | { kind: "align"; id: string }
  | {
      kind: "document";
      id: string;
      page: number;
      ocrId?: string;
      /** Bumped when the page is changed from outside the viewer, so the viewer follows. */
      jump: number;
    }
  | { kind: "add" };

/** An aligned historical map laid over the land ("then and now"). */
export interface RecordsOverlay {
  id: string;
  label: string;
  bounds: number[];
  attribution: string;
}

interface RecordsState {
  landId: string | null;
  view: RecordsView;
  overlay: RecordsOverlay | null;
  /** The person hid the overlay; it stays hidden until they show one again. */
  overlayHidden: boolean;
  opacity: number;
  /** Starts fresh when another land is opened. */
  forLand: (landId: string) => void;
  open: (view: RecordsView) => void;
  back: () => void;
  showOverlay: (overlay: RecordsOverlay) => void;
  hideOverlay: () => void;
  setOpacity: (opacity: number) => void;
  /** Takes the overlay off the map without forgetting it (the lens closed). */
  stowOverlay: () => void;
  /** Puts a remembered overlay back on the map (the lens opened again). */
  restoreOverlay: () => void;
}

const INITIAL = {
  landId: null,
  view: { kind: "summary" } as RecordsView,
  overlay: null,
  overlayHidden: false,
  opacity: 0.6,
};

function draw(overlay: RecordsOverlay, opacity: number) {
  useLandContext.getState().setRaster({
    id: overlay.id,
    kind: "archive-alignment",
    band: 1,
    bounds: overlay.bounds,
    attribution: overlay.attribution,
    opacity,
  });
}

/** Takes every aligned map off the map; only the Records lens lays them. */
export function removeAlignedMaps() {
  const context = useLandContext.getState();
  for (const raster of Object.values(context.rasters))
    if (raster.kind === "archive-alignment") context.removeRaster(raster.id);
}

export const useRecords = create<RecordsState>((set, get) => ({
  ...INITIAL,
  forLand: (landId) => {
    const previous = get().landId;
    if (previous === landId) return;
    // Something opened before any land was shown (`openRecords`) is kept for this land.
    if (previous === null) {
      set({ landId });
      return;
    }
    removeAlignedMaps();
    set({ ...INITIAL, landId });
  },
  open: (view) => set({ view }),
  back: () => {
    const view = get().view;
    set({ view: view.kind === "align" ? { kind: "archive", id: view.id } : { kind: "summary" } });
  },
  showOverlay: (overlay) => {
    const previous = get().overlay;
    if (previous && previous.id !== overlay.id) useLandContext.getState().removeRaster(previous.id);
    draw(overlay, get().opacity);
    set({ overlay, overlayHidden: false });
  },
  hideOverlay: () => {
    const overlay = get().overlay;
    if (overlay) useLandContext.getState().removeRaster(overlay.id);
    set({ overlayHidden: true });
  },
  setOpacity: (opacity) => {
    const { overlay, overlayHidden } = get();
    set({ opacity });
    if (overlay && !overlayHidden) draw(overlay, opacity);
  },
  stowOverlay: () => removeAlignedMaps(),
  restoreOverlay: () => {
    const { overlay, overlayHidden, opacity } = get();
    if (overlay && !overlayHidden) draw(overlay, opacity);
  },
}));

/** Opens something in the Records lens from anywhere (a pin, a cited document, a gallery). */
export function openRecords(view: RecordsView) {
  useRecords.getState().open(view);
  useLens.getState().setLens("records");
}
