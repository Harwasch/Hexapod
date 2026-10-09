import { create } from "zustand";

/**
 * The poke tool (`cesium/skinPoke.ts`): while it is on, a press on an object that moves by a
 * skin grabs it -- drag to pull, let go to see it ring -- and a press anywhere else still moves
 * the camera. `K` turns it on and off (app/hotkeys.ts), as does Settings' switch under the
 * wind; Escape turns it off.
 */
export interface PokeGrabbed {
  assetId: string;
  instance: number;
  /** What it is called (its top tag, else its category), when known. */
  label: string;
  /** Its slowest mode's frequency under its material, Hz. */
  hz: number | null;
  /** Whether it moves whole (movable) or is rooted. */
  movable: boolean;
}

interface SkinPokeState {
  active: boolean;
  /** What the last press grabbed, while held or ringing; null otherwise. */
  grabbed: PokeGrabbed | null;
  setActive: (on: boolean) => void;
  toggle: () => void;
  setGrabbed: (grabbed: PokeGrabbed | null) => void;
}

export const useSkinPoke = create<SkinPokeState>((set) => ({
  active: false,
  grabbed: null,
  setActive: (on) =>
    set((s) => (s.active === on ? s : { active: on, grabbed: on ? s.grabbed : null })),
  toggle: () => set((s) => ({ active: !s.active, grabbed: s.active ? null : s.grabbed })),
  setGrabbed: (grabbed) => set({ grabbed }),
}));
