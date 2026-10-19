import { create } from "zustand";

/**
 * off: no lens. peek: shown while Alt is held over the map (the scene sets it, LanternBridge
 * mirrors it here). sticky: shown until closed (Y, the strip's Look back, Esc to close).
 */
export type LanternMode = "off" | "peek" | "sticky";
/** The rail's kind chips: aerial photographs, old maps, satellite scenes. */
export type LanternKind = "aerial" | "maps" | "satellite";

/** Why the lens cannot show here and now; the strip says so in words (TimeLanternControl). */
export type LanternUnavailable =
  "open-world" | "webgl1" | "orbit" | "clipping-off" | "explore" | "paint" | "altitude";

export type LanternAvailability = { ok: true } | { ok: false; reason: LanternUnavailable };

/** Above this altitude (m) the lens is not offered: a NAIP pixel is less than a screen pixel. */
export const LANTERN_MAX_ALTITUDE_M = 20_000;

/** West, south, east, north (degrees). */
export type LanternBbox = [number, number, number, number];

interface LanternState {
  mode: LanternMode;
  /** The source shown, in the rail's order (newest first, after the kind filter). */
  index: number;
  kinds: Record<LanternKind, boolean>;
  /** The area to list dated imagery for, when something chose one (the Descent's arrival). */
  bboxOverride: LanternBbox | null;
  /**
   * Around the view's centre when the lantern opened with no area chosen and no land selected
   * (LanternBridge sets it): the lens then looks back where the person was looking.
   */
  autoBbox: LanternBbox | null;
  /** Whether the scene can show the lens now (cesium/TimeLantern.ts, mirrored by the bridge). */
  availability: LanternAvailability;
  /** Set once the lens has been asked for, or a descent has arrived: the catalog may load. */
  requested: boolean;
  setMode: (mode: LanternMode) => void;
  /** Sticky on, or off from any mode. */
  toggle: () => void;
  select: (index: number) => void;
  /** Older (+1) or newer (-1) by `delta` steps, within `count` sources. */
  step: (delta: number, count: number) => void;
  toggleKind: (kind: LanternKind) => void;
  setBbox: (bbox: LanternBbox | null) => void;
  setAutoBbox: (bbox: LanternBbox | null) => void;
  setAvailability: (availability: LanternAvailability) => void;
  request: () => void;
}

/**
 * The Time Lantern's state for the HUD: the scene draws the lens (cesium/TimeLantern.ts) and
 * LanternBridge keeps the two in step. Satellite scenes are off by default: at 10 m a pixel
 * they show a field as a few dozen squares, coarse beside 0.6 m aerial photographs.
 */
export const useLantern = create<LanternState>()((set) => ({
  mode: "off",
  index: 0,
  kinds: { aerial: true, maps: true, satellite: false },
  bboxOverride: null,
  autoBbox: null,
  availability: { ok: false, reason: "open-world" },
  requested: false,
  setMode: (mode) =>
    set((s) => (s.mode === mode ? s : { mode, requested: s.requested || mode !== "off" })),
  toggle: () => set((s) => ({ mode: s.mode === "off" ? "sticky" : "off", requested: true })),
  select: (index) => set((s) => (s.index === index ? s : { index: Math.max(0, index) })),
  step: (delta, count) =>
    set((s) => {
      if (count <= 0) return s;
      const index = Math.min(count - 1, Math.max(0, s.index + delta));
      return index === s.index ? s : { index };
    }),
  toggleKind: (kind) => set((s) => ({ kinds: { ...s.kinds, [kind]: !s.kinds[kind] }, index: 0 })),
  setBbox: (bboxOverride) => set({ bboxOverride, index: 0 }),
  // Re-centred around the view, the lens keeps the source it shows (TimeLantern.setSources
  // finds it, or the nearest of its kind in time, in the new list).
  setAutoBbox: (autoBbox) => set({ autoBbox }),
  setAvailability: (availability) =>
    set((s) =>
      s.availability.ok === availability.ok &&
      (availability.ok || (!s.availability.ok && s.availability.reason === availability.reason))
        ? s
        : { availability },
    ),
  request: () => set((s) => (s.requested ? s : { requested: true })),
}));
