import { create } from "zustand";

import { useAsCaptured } from "@/state/asCaptured";

/**
 * The living lab (prototype, branch `claude/research-living-water-motion-*`): water and wind
 * drawn per frame on a scan by PlayCanvas's final-draw hook (cesium/scanView/livingHook.ts),
 * behind a developer toggle (`?living=1`, or the developer build). Everything it draws is
 * invented, so it is labelled Simulated while on and draws nothing under As captured.
 *
 * Arms (`motion`):
 * - `gust`: a travelling gust field over every plant splat -- a wave train downwind, gust
 *   patches that drift with the wind, leaf shimmer on small splats;
 * - `clumps`: resonant clumps -- each ~1.5 m tuft or crown its own oscillator (frequency by
 *   height), forced by the same gust field, with branch flutter under it.
 *
 * Water (`water`) is an ellipse and a level in the scan's frame, a flooded hollow: splats inside
 * the ellipse from just above the level down to `waterThickness` below it are water, drawn as a
 * level surface with ripples, a sky reflection, sun glints and streaks carried along
 * `flowBearingDeg`. The shore is where the ground meets the level.
 */
export type LivingMotionArm = "off" | "gust" | "clumps" | "spectra" | "articulated";

export interface LivingLabState {
  /** The lab's panel shows (the developer toggle). */
  open: boolean;
  water: boolean;
  motion: LivingMotionArm;
  /** Explicit lab-only sidecar URL, in this scan's metre frame. */
  spectraUrl: string;
  spectraStatus: string;
  articulationUrl: string;
  articulationStatus: string;
  /** Overall motion amplitude, 1 a moderate breeze. */
  strength: number;
  /** Where the wind blows to, degrees clockwise from north. */
  windBearingDeg: number;
  /** How fast gusts travel, m/s. */
  gustSpeed: number;
  /** Greenness a splat needs to move (excess green over brightness); -1: everything above ground. */
  greenness: number;
  /** Draws the classes (water blue, moving plants magenta) instead of the effect. */
  mask: boolean;
  /** Water: the level (scan frame z, metres), depth below it, the ellipse's centre and radii. */
  waterLevel: number;
  waterThickness: number;
  waterBox: [number, number, number, number];
  flowBearingDeg: number;
  flowSpeed: number;
  /** The sun for glints when the relight is off: azimuth from north and elevation, degrees. */
  sunAzimuthDeg: number;
  sunElevationDeg: number;
  /** Flat ground (scan frame z) where no ground grid can be built (a streamed scan). */
  flatGround: number;
  /** A fixed time (s) for recordings; null: the wall clock. */
  clock: number | null;
  /**
   * Measuring: a frame every frame even with every arm off, each frame's draw timed to its end
   * (`gl.finish`), the times kept on `window.__livingBench`.
   */
  bench: boolean;
  /**
   * `?norig=1`: a scan with a skeleton rig (the Living Survey's wind, drawn by CesiumJS) is drawn
   * by the scan renderer like any other, so the lab's arms can run on the same tree.
   */
  noRig: boolean;
  set: (patch: Partial<Omit<LivingLabState, "set">>) => void;
}

function flagFromLocation(name: string): boolean {
  try {
    return new URLSearchParams(window.location.search).get(name) === "1";
  } catch {
    return false;
  }
}

export const useLivingLab = create<LivingLabState>()((set) => ({
  open: flagFromLocation("living"),
  water: false,
  motion: "off",
  spectraUrl: "",
  spectraStatus: "No spectra loaded",
  articulationUrl: "",
  articulationStatus: "No articulation loaded",
  strength: 1,
  windBearingDeg: 60,
  gustSpeed: 4,
  greenness: 0.04,
  mask: false,
  waterLevel: 0,
  waterThickness: 0.15,
  waterBox: [0, 0, 5, 5],
  flowBearingDeg: 90,
  flowSpeed: 0.4,
  sunAzimuthDeg: 210,
  sunElevationDeg: 30,
  flatGround: 0,
  clock: null,
  bench: false,
  noRig: flagFromLocation("norig"),
  set: (patch) => set(patch),
}));

/** What the lab draws now: nothing under As captured or with every arm off. */
export function effectiveLivingLab(): LivingLabState | null {
  const lab = useLivingLab.getState();
  if (useAsCaptured.getState().on)
    return lab.bench ? { ...lab, water: false, motion: "off" } : null;
  if (!lab.water && lab.motion === "off" && !lab.bench) return null;
  return lab;
}

/** Calls `listener` whenever what the lab draws may have changed; the remover. */
export function watchLivingLab(listener: () => void): () => void {
  const a = useLivingLab.subscribe(listener);
  const b = useAsCaptured.subscribe(listener);
  return () => {
    a();
    b();
  };
}

// The lab's recordings and bench drive the store from the page (scripts/living-lab/).
if (typeof window !== "undefined" && import.meta.env.DEV)
  (window as unknown as { __livingLab?: typeof useLivingLab }).__livingLab = useLivingLab;
