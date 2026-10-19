import { create } from "zustand";

import type { Footprint, LandMapGeometry } from "@twin/contracts";

import type { LandPoint } from "./land";

/** What a mark on the map is: the land's own facts, the agent's work, or something to watch. */
export type MapTone = "accent" | "agent" | "warn" | "neutral";

/** A labelled place on the map, drawn as a chip that stays on its spot as the camera moves. */
export interface MapPin {
  id: string;
  position: LandPoint;
  label: string;
  tone?: MapTone;
  /** Opens what the pin stands for (a finding, a document, an asset). */
  onSelect?: () => void;
}

/** The geometry a card is pointing at right now, outlined on the map. */
export interface MapFocus {
  key: string;
  geometry: LandMapGeometry | Footprint;
  /** Where a leader line from the card ends: a spot on or inside the geometry. */
  anchor: LandPoint;
  tone: MapTone;
}

interface LandMapState {
  /** Pins by owner (a lens, the agent), so each owner replaces only its own. */
  pins: Record<string, MapPin[]>;
  setPins: (owner: string, pins: MapPin[]) => void;
  clearPins: (owner: string) => void;
  focus: MapFocus | null;
  setFocus: (focus: MapFocus | null) => void;
  /** Clears focus only if it is still the given one; a later hover may already own it. */
  releaseFocus: (key: string) => void;
}

export const useLandMap = create<LandMapState>((set, get) => ({
  pins: {},
  setPins: (owner, pins) => set((state) => ({ pins: { ...state.pins, [owner]: pins } })),
  clearPins: (owner) => {
    if (!(owner in get().pins)) return;
    set((state) => ({
      pins: Object.fromEntries(Object.entries(state.pins).filter(([key]) => key !== owner)),
    }));
  },
  focus: null,
  setFocus: (focus) => set({ focus }),
  releaseFocus: (key) => {
    if (get().focus?.key === key) set({ focus: null });
  },
}));

/**
 * A line from a card to the place on the map it describes. Leaders are drawn every frame from
 * live DOM rectangles, so they are kept outside React state: registering one re-renders nothing.
 */
export interface Leader {
  id: string;
  element: HTMLElement;
  anchor: LandPoint;
  tone: MapTone;
}

const leaders = new Map<string, Leader>();
const listeners = new Set<() => void>();

export function registerLeader(leader: Leader): () => void {
  leaders.set(leader.id, leader);
  listeners.forEach((listener) => listener());
  return () => {
    if (leaders.get(leader.id) === leader) leaders.delete(leader.id);
    listeners.forEach((listener) => listener());
  };
}

export function currentLeaders(): Leader[] {
  return [...leaders.values()];
}

/** Called whenever a leader is added or removed, so the overlay can draw without waiting. */
export function onLeadersChange(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
