import { create } from "zustand";

import type { Footprint, LandArea, LandCreate } from "@twin/contracts";

import { recordAction } from "./history";

export type LandMode = "browse" | "draw" | "corridor" | "pick" | "candidates" | "edit";
export type LandPoint = [number, number];
export interface LandSketch {
  mode: "draw" | "corridor";
  points: LandPoint[];
  width: number;
  unit: "ft" | "m";
}

interface LandState {
  session: number;
  active: LandArea | null;
  draft: LandCreate | null;
  mode: LandMode;
  points: LandPoint[];
  corridorWidth: number;
  corridorUnit: "ft" | "m";
  setCorridorWidth: (width: number) => void;
  setCorridorUnit: (unit: "ft" | "m") => void;
  restoreSketch: (sketch: LandSketch, draft: LandCreate | null) => void;
  error: string | null;
  select: (land: LandArea) => void;
  begin: (mode: LandMode) => void;
  addPoint: (point: LandPoint) => void;
  removePoint: () => void;
  propose: (draft: LandCreate) => void;
  updateBoundary: (boundary: Footprint) => void;
  updateDraft: (patch: Partial<LandCreate>) => void;
  cancel: () => void;
  clear: () => void;
  setError: (error: string | null) => void;
}

/** The land under investigation is independent of the map's inspected feature. */
export const useLand = create<LandState>()((set, get) => ({
  session: 0,
  active: null,
  draft: null,
  mode: "browse",
  points: [],
  corridorWidth: 100,
  corridorUnit: "ft",
  setCorridorWidth: (corridorWidth) => set({ corridorWidth }),
  setCorridorUnit: (corridorUnit) =>
    set((state) => ({
      corridorUnit,
      corridorWidth:
        corridorUnit === state.corridorUnit
          ? state.corridorWidth
          : Number((state.corridorWidth * (corridorUnit === "m" ? 0.3048 : 1 / 0.3048)).toFixed(4)),
    })),
  restoreSketch: (sketch, draft) =>
    set((state) => ({
      draft,
      mode: sketch.mode,
      points: sketch.points,
      corridorWidth: sketch.width,
      corridorUnit: sketch.unit,
      error: null,
      session: state.session + 1,
    })),
  error: null,
  select: (active) =>
    set((s) => ({
      active,
      draft: null,
      mode: "browse",
      points: [],
      error: null,
      session: s.session + 1,
    })),
  begin: (mode) =>
    set((s) => ({
      mode,
      points: [],
      error: null,
      session:
        mode === "draw" || mode === "corridor" || mode === "pick" || mode === "candidates"
          ? s.session + 1
          : s.session,
    })),
  addPoint: (point) => {
    const previous = get().points;
    if ((get().mode === "pick" || get().mode === "candidates") && previous.length) return;
    const last = previous.at(-1);
    if (last?.[0] === point[0] && last[1] === point[1]) return;
    if (previous.length >= 2000) {
      set({ error: "This outline has reached 2,000 points. Finish it or import a boundary." });
      return;
    }
    const mode = get().mode;
    const session = get().session;
    recordAction(
      "Add boundary point",
      () => set({ points: [...previous, point] }),
      () => set({ points: previous }),
      { alive: () => get().mode === mode && get().session === session },
    );
  },
  removePoint: () => {
    const previous = get().points;
    const session = get().session;
    const mode = get().mode;
    if (!previous.length) return;
    recordAction(
      "Remove boundary point",
      () => set({ points: previous.slice(0, -1) }),
      () => set({ points: previous }),
      { alive: () => get().session === session && get().mode === mode },
    );
  },
  propose: (draft) =>
    set((s) => ({ draft, mode: "browse", points: [], error: null, session: s.session + 1 })),
  updateBoundary: (boundary) => {
    const previous = get().draft;
    if (!previous) return;
    const session = get().session;
    const next = {
      ...previous,
      boundary,
      source: {
        ...previous.source,
        method: "drawn" as const,
        meaning: "study-area" as const,
        label: "Adjusted on the map",
      },
    };
    recordAction(
      "Adjust land boundary",
      () => set({ draft: next }),
      () => set({ draft: previous }),
      { alive: () => get().draft !== null && get().session === session },
    );
  },
  updateDraft: (patch) => set((s) => ({ draft: s.draft ? { ...s.draft, ...patch } : null })),
  cancel: () =>
    set((s) => ({ draft: null, mode: "browse", points: [], error: null, session: s.session + 1 })),
  clear: () =>
    set((s) => ({
      active: null,
      draft: null,
      mode: "browse",
      points: [],
      corridorWidth: 100,
      corridorUnit: "ft",
      error: null,
      session: s.session + 1,
    })),
  setError: (error) => set({ error }),
}));
