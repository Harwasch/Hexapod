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
  world: "open" as WorldMode,
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
    { name: "twin.settings.v1", version: 1 },
  ),
);

/** Base screen-space error for each quality preset (lower = sharper). */
export const QUALITY_SSE: Record<QualityPreset, { base: number; min: number; max: number }> = {
  performance: { base: 24, min: 12, max: 48 },
  balanced: { base: 16, min: 6, max: 32 },
  ultra: { base: 8, min: 2, max: 16 },
};
