/**
 * Wind, and what the Living Survey is doing with it.
 *
 * Deliberately **not** persisted. `state/settings.ts` is the only persisted store, and wind does
 * not belong there: the first thing a person sees must be the measurement, so every session
 * starts calm and motion is opted into again each time. Persisting it would mean a survey that
 * silently animates on load because of a slider someone moved a week ago.
 */

import { create } from "zustand";

import { DEFAULT_WIND_STRENGTH, WIND_CALM, type WindSettings } from "@twin/world";

import type { DeformerPhase, DeformerReason } from "@/cesium/SplatDeformer";

/**
 * Re-exported from `@twin/world`, where it is derived.
 *
 * The strength the wind control lands on when someone turns wind on is a property of the motion
 * model, not of this store: it is read off the model's own sort-staleness bound. The derivation,
 * and the table it comes from, live beside that bound in `wind.ts`. This store only decides that
 * the app starts at 0 and never persists it.
 */
export { DEFAULT_WIND_STRENGTH };

/**
 * The median gaussian extent the staleness figures are quoted against, metres — the denominator
 * of the table beside `DEFAULT_WIND_STRENGTH` in `@twin/world`.
 *
 * A **reference** yardstick — a real drone capture's median — and not the median of whichever
 * tileset is on screen. Deriving that would mean decoding scales out of the packed splat buffer,
 * which nothing does today; the fixture's own median, measured over `data/tiles/synthetic-tree`,
 * is 10.7 cm, so `sortStaleness` overstates the fixture's staleness about fivefold.
 *
 * That makes the ratio engineering intuition, not a measurement of what is displayed. Anything
 * in the UI that prints it must print this yardstick beside it — the developer panel does, and
 * it is the only place that shows it. A figure in "splat radii" with no stated denominator
 * reads as measured and is not.
 */
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
   * is safe to mirror into React and is what site-scoped UI hangs off. The ambient badge uses
   * {@link LivingSurveyStatus.animating} instead, so it cannot blink between two uploads.
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
  /**
   * True while the wind is blowing and at least one rig is attached and ready.
   *
   * Derived from wind × attachment, **not** from the last frame's `displaced` flags, so it does
   * not flicker between uploads and does not drop out during a `holdMeasuredPose()` snapshot.
   * This is what the ambient "Simulated motion" badge hangs off, and the reason the badge can
   * be trusted to be on screen for as long as the motion is.
   */
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
