import { create } from "zustand";

import type { OsmCandidate } from "@/missions/osm";

/** Where the Descent is going: a place the geocoder found, or coordinates someone typed. */
export interface DescentTarget {
  label: string;
  longitude: number;
  latitude: number;
  /** The geocoder's bounding box, when it gave one (degrees). */
  rectangle?: { west: number; south: number; east: number; north: number };
  /** Who answered the search: the geocoder's credit, or "Coordinates you entered". */
  attribution: string;
}

/**
 * The mapped area around the target: being looked for, found, none found, the lookup failed, or
 * not looked for (a town or a region is too large to be one piece of land).
 */
export type MappedStatus = "pending" | "found" | "none" | "error" | "skipped";

export type DescentPhase = "idle" | "flying" | "arrived" | "cancelled";

interface DescentState {
  phase: DescentPhase;
  target: DescentTarget | null;
  mapped: OsmCandidate | null;
  mappedStatus: MappedStatus;
  /** Why the mapped-area lookup failed, in words; null otherwise. */
  error: string | null;
  start: (target: DescentTarget) => void;
  setMapped: (mapped: OsmCandidate | null, status: MappedStatus, error?: string | null) => void;
  arrive: () => void;
  cancel: () => void;
  clear: () => void;
}

/**
 * The Descent: the onboarding card's "Where is your land?" flight from orbit down to a place,
 * and the card it lands on (features/descent). The camera and the lookups are the orchestrator's
 * (features/descent/descent.ts); this is what the arrival card reads.
 */
export const useDescent = create<DescentState>()((set) => ({
  phase: "idle",
  target: null,
  mapped: null,
  mappedStatus: "pending",
  error: null,
  start: (target) =>
    set({ phase: "flying", target, mapped: null, mappedStatus: "pending", error: null }),
  setMapped: (mapped, mappedStatus, error = null) => set({ mapped, mappedStatus, error }),
  arrive: () => set((s) => (s.target ? { phase: "arrived" } : s)),
  cancel: () => set((s) => (s.target && s.phase === "flying" ? { phase: "cancelled" } : s)),
  clear: () =>
    set({ phase: "idle", target: null, mapped: null, mappedStatus: "pending", error: null }),
}));
