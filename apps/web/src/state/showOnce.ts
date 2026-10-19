import { create } from "zustand";

import type { SignatureSimilarity } from "@twin/contracts";

import { ALIKE_DEFAULT, type ScoreGrid } from "@/features/showOnce/grid";
import type { JobSummary } from "@/features/showOnce/makeJob";

/**
 * Show once, everywhere: one session at a time, on the selected Land's freshest signature
 * raster. The card in the Land panel starts and ends it; the map's bridge
 * (`cesium/ShowOnceBridge.tsx`) answers taps and drapes the answer. Never persisted: a Land's
 * cells are workspace-private.
 */
export type ShowOnceMode = "compare" | "protect";

export interface ShowOnceSession {
  landId: string;
  revision: number;
  rasterId: string;
}

export interface ShowOnceResult {
  read: SignatureSimilarity;
  grid: ScoreGrid;
}

interface ShowOnceState {
  session: ShowOnceSession | null;
  /** What a tap on the land does: compare that place, or protect the lit patch there. */
  mode: ShowOnceMode;
  /** Whether the card is on screen; the map only answers taps (and drapes) while it is. */
  shown: boolean;
  result: ShowOnceResult | null;
  /** A tap being compared. */
  pending: { longitude: number; latitude: number } | null;
  /** What went wrong with the last tap, in words for the card. */
  message: string | null;
  threshold: number;
  /** Protected cells of the display grid (1 = left out of the job). */
  guarded: Uint8Array | null;
  /**
   * What "Make this the job" last made, and on which Land. Kept here, not in the card: the plan
   * card opens as the job is made, which closes Land and the card with it, and the card says
   * what became of the lit ground when it is next on screen. A new session clears it.
   */
  job: { landId: string; summary: JobSummary } | null;
  start: (session: ShowOnceSession) => void;
  stop: () => void;
  setShown: (shown: boolean) => void;
  setMode: (mode: ShowOnceMode) => void;
  setThreshold: (threshold: number) => void;
  setPending: (pending: ShowOnceState["pending"]) => void;
  setResult: (result: ShowOnceResult) => void;
  setMessage: (message: string | null) => void;
  setGuarded: (guarded: Uint8Array | null) => void;
  setJob: (job: ShowOnceState["job"]) => void;
}

const IDLE = {
  session: null,
  mode: "compare" as const,
  result: null,
  pending: null,
  message: null,
  threshold: ALIKE_DEFAULT,
  guarded: null,
};

export const useShowOnce = create<ShowOnceState>()((set, get) => ({
  ...IDLE,
  shown: false,
  job: null,
  start: (session) => {
    const current = get().session;
    // The same raster keeps its answer; another one (a new boundary, new signatures) starts over.
    if (current?.rasterId === session.rasterId && current.landId === session.landId) return;
    set({ ...IDLE, session });
  },
  stop: () => set({ ...IDLE }),
  setShown: (shown) => {
    if (get().shown !== shown) set({ shown });
  },
  setMode: (mode) => set({ mode, message: null }),
  setThreshold: (threshold) => set({ threshold }),
  setPending: (pending) => set({ pending, ...(pending ? { message: null } : {}) }),
  setResult: (result) =>
    set((s) => {
      const previous = s.result?.grid;
      const same =
        previous?.columns === result.grid.columns &&
        previous.rows === result.grid.rows &&
        previous.west === result.grid.west &&
        previous.north === result.grid.north;
      // Protected ground stays protected for the next tap on the same grid.
      return { result, pending: null, message: null, guarded: same ? s.guarded : null };
    }),
  setMessage: (message) => set({ message, pending: null }),
  setGuarded: (guarded) => set({ guarded, message: null }),
  setJob: (job) => set({ job }),
}));
