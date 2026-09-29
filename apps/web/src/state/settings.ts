import { create } from "zustand";
import { persist } from "zustand/middleware";

import type { UnitSystem } from "@twin/geo";

export type ThemeMode = "auto" | "light" | "dark";
export type QualityPreset = "performance" | "balanced" | "ultra";
export type WorldMode = "open" | "photorealistic";

export interface SettingsState {
  theme: ThemeMode;
  reducedMotion: boolean;
  highContrast: boolean;
  reducedTransparency: boolean;
  units: UnitSystem;
  quality: QualityPreset;
  /** When set, overrides the preset's screen-space error for site tilesets. */
  manualScreenSpaceError: number | null;
  adaptiveQuality: boolean;
  world: WorldMode;
  onboardingDismissed: boolean;
  /** The evaluation-ion-token hint is a setup note for whoever deploys this, not an
      operator's concern; once they have read it, it stays gone. */
  ionTokenNoticeDismissed: boolean;
  devToolsOpen: boolean;
  exploreSpeed: number;
  /**
   * Living Survey motion in the splat vertex shader ("Motion on GPU"). On by default; off puts
   * every rigged site on the CPU path, for comparing the two or for hardware where the shader
   * path misbehaves. A build can still force the CPU path (`VITE_SPLAT_GPU_MOTION=0`), and a
   * site falls back to it on its own when the engine or its tiles cannot take the shader path.
   *
   * Persisted, unlike the wind (`state/living.ts`): this is how the device renders, not what
   * the scene shows — both paths draw the same motion, so an old choice cannot make a survey
   * silently animate.
   */
  livingGpuMotion: boolean;
  /**
   * The API's shared write token (`API_WRITE_TOKEN`), entered once in the UI.
   *
   * It lives here, beside the other preferences, and deliberately not in a `VITE_`
   * variable: those are inlined into the bundle and served to everyone who loads the
   * page (README.md), which for a shared write credential means publishing it. An
   * empty token is the normal case — a deployment with no token configured leaves
   * writes open, so development never has to enter one.
   *
   * Trade-off, stated plainly: localStorage is readable by any XSS on this origin, so
   * this is acceptable for a single-user prototype and would not be for real accounts.
   * Real accounts want a session cookie the page cannot read.
   */
  writeToken: string;
  set: (patch: Partial<Omit<SettingsState, "set" | "reset">>) => void;
  reset: () => void;
}

const defaults = {
  theme: "auto" as ThemeMode,
  reducedMotion: false,
  highContrast: false,
  reducedTransparency: false,
  units: "metric" as UnitSystem,
  quality: "balanced" as QualityPreset,
  manualScreenSpaceError: null,
  adaptiveQuality: true,
  world: "photorealistic" as WorldMode,
  onboardingDismissed: false,
  ionTokenNoticeDismissed: false,
  devToolsOpen: false,
  exploreSpeed: 4,
  livingGpuMotion: true,
  writeToken: "",
};

export const useSettings = create<SettingsState>()(
  persist(
    (set) => ({
      ...defaults,
      set: (patch) => set(patch),
      reset: () => set({ ...defaults }),
    }),
    {
      name: "twin.settings.v1",
      version: 2,
      // v2 switched the default world to Google Photorealistic; stored v1 settings still
      // carried the old default, so they are moved along once.
      migrate: (persisted, version) => {
        const state = (persisted ?? {}) as Partial<SettingsState>;
        return version < 2 ? { ...state, world: "photorealistic" as WorldMode } : state;
      },
    },
  ),
);

/** Base screen-space error for each quality preset (lower = sharper). */
export const QUALITY_SSE: Record<QualityPreset, { base: number; min: number; max: number }> = {
  // The minimum is where idle refinement ends up; memory pressure is the governor. A 2 to
  // 5 cm mesh only shows its detail below about 2 px of error (measured: 8 px draws 176k
  // triangles and looks like the Google world, 2 px draws 416k and looks like the survey).
  // The base is what a moving camera gets, for sites and the Google world alike (parity, so
  // collected data never looks worse than its surroundings). Google's tiles are coarse at
  // Cesium's default of 16, hence 8 on balanced.
  performance: { base: 16, min: 4, max: 48 },
  balanced: { base: 8, min: 2, max: 32 },
  ultra: { base: 4, min: 1, max: 16 },
};
