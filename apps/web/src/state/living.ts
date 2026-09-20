/**
 * Wind, and what the Living Survey is doing with it.
 *
 * Deliberately **not** persisted. `state/settings.ts` is the only persisted store, and wind does
 * not belong there: the first thing a person sees must be the measurement, so every session
 * starts calm and motion is opted into again each time. Persisting it would mean a survey that
 * silently animates on load because of a slider someone moved a week ago.
 */

import { create } from "zustand";

import { WIND_CALM, type WindSettings } from "@twin/world";

import type { DeformerPhase, DeformerReason } from "@/cesium/SplatDeformer";

/**
 * Default strength the wind control lands on when someone turns wind on — **not** the value the
 * app starts at, which is always 0.
 *
 * Chosen from the sort-staleness bound, not from a screenshot. The splat sorter reads canonical
 * positions we never touch, so a displaced splat carries a draw-order key that is stale by
 * `maxDisplacement / medianGaussianScale` splat radii (`sortStaleness` in `@twin/world`).
 * Against the ~2 cm median gaussian of a real drone capture and the 6 m synthetic tree's rig:
 *
 * | strength | worst-case displacement | staleness |
 * | --- | --- | --- |
 * | 0.02 | 3.3 cm | 1.6 radii |
 * | 0.12 | 19.5 cm | 9.7 radii |
 * | 0.5 | 78.9 cm | 39.4 radii |
 * | 1 | 1.45 m | 72.6 radii |
 *
 * `SORT_STALENESS_NOTICEABLE` is 1, and it is a *hypothesis* — headless GL here is SwiftShader,
 * so where the threshold really sits needs a human eye on real hardware. 0.12 is the largest
 * strength whose staleness bound stays inside one decade of that hypothesis, which is the
 * honest width of our ignorance. It is also legible as motion: 19.5 cm is 3.2 % of the tree's
 * height in one glance, where 0.02 (3 cm) would be indistinguishable from the feature being off.
 *
 * Two things keep that bound pessimistic, and neither is an argument for going higher: it is a
 * proven maximum over the outermost leaf's whole ancestor chain with every noise term
 * simultaneously extreme, and the fixture's own gaussians are coarser (10.7 cm median measured
 * over `data/tiles/synthetic-tree`), where the same strength is only 1.8 radii.
 */
export const DEFAULT_WIND_STRENGTH = 0.12;

/** The median gaussian extent the staleness figures above are quoted against, metres. */
export const REFERENCE_GAUSSIAN_SCALE_M = 0.02;

/** What one deformed site is doing. Mirrors `DeformerStatus`, minus what only the GPU cares about. */
export interface LivingSiteStatus {
  readonly siteId: string;
  readonly siteSlug: string;
  readonly assetId: string;
  readonly phase: DeformerPhase;
  readonly reason?: DeformerReason;
  readonly numSplats: number;
  /**
   * True while the GPU holds displaced positions rather than the measured ones. Changes only on
   * transitions — the first frame of a gust and the frame that restores the measurement — so it
   * is safe to mirror into React and is what an S5 badge hangs off.
   */
  readonly displaced: boolean;
  /** What the rig says it was built from. Carried into the provenance UI. */
  readonly rigSourceNote: string;
  /** Worst-case displacement at the scene's current wind, metres. A bound, not a sample. */
  readonly maxDisplacementM: number;
  /** That displacement in splat radii, against {@link REFERENCE_GAUSSIAN_SCALE_M}. */
  readonly sortStaleness: number;
}

/** Everything the scene knows about the Living Survey, published whole on every change. */
export interface LivingSurveyStatus {
  /** The wind the scene is actually running — 0 whenever reduced motion is on. */
  readonly wind: WindSettings;
  /** True while at least one site holds displaced positions rather than its measured ones. */
  readonly animating: boolean;
  readonly sites: readonly LivingSiteStatus[];
}

export const LIVING_SURVEY_IDLE: LivingSurveyStatus = {
  wind: WIND_CALM,
  animating: false,
  sites: [],
};

interface LivingState {
  /** What the person asked for. The scene forces 0 under reduced motion; this keeps their choice. */
  wind: WindSettings;
  /** The scene's own report, mirrored from the `living` event. */
  status: LivingSurveyStatus;
  setWind: (patch: Partial<WindSettings>) => void;
  setStatus: (status: LivingSurveyStatus) => void;
  reset: () => void;
}

/** Clamps to the model's own domain so a stray value can never reach `deform`. */
function normalizeWind(wind: WindSettings): WindSettings {
  const strength = Number.isFinite(wind.strength) ? Math.min(1, Math.max(0, wind.strength)) : 0;
  const raw = Number.isFinite(wind.bearingDeg) ? wind.bearingDeg : 0;
  return { strength, bearingDeg: ((raw % 360) + 360) % 360 };
}

export const useLiving = create<LivingState>()((set) => ({
  wind: WIND_CALM,
  status: LIVING_SURVEY_IDLE,
  setWind: (patch) => set((s) => ({ wind: normalizeWind({ ...s.wind, ...patch }) })),
  setStatus: (status) => set({ status }),
  reset: () => set({ wind: WIND_CALM, status: LIVING_SURVEY_IDLE }),
}));
