import { create } from "zustand";

/**
 * Sunlight on splat scans (ADR 0019 "Sunlight", cesium/splatSunlight.ts): off, the sun of now,
 * or the sun at the moment the shown date was captured ("capture" also holds the scene clock
 * there while it is on, features/views/RelitBadge.tsx). Drawn by the CesiumJS splat primitive
 * only, and never while As captured is on (state/asCaptured.ts): a relight is a model of light
 * the capture did not see.
 *
 * Not persisted: a reload that kept it would show a relit scan to someone who never asked.
 */
export type SunMode = "off" | "now" | "capture";

interface LightingState {
  sun: SunMode;
  set: (sun: SunMode) => void;
}

export const useLighting = create<LightingState>()((set) => ({
  sun: "off",
  set: (sun) => set({ sun }),
}));
