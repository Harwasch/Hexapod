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
  devToolsOpen: boolean;
  exploreSpeed: number;
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
  devToolsOpen: false,
  exploreSpeed: 4,
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
  performance: { base: 24, min: 4, max: 48 },
  balanced: { base: 16, min: 2, max: 32 },
  ultra: { base: 8, min: 1, max: 16 },
};
