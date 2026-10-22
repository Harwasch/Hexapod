import { create } from "zustand";

/**
 * What the relight drew (cesium/sunlight.ts), per model: the scene time its sun was for and
 * how far up that sun was (`sunUpFactor`, 0 below the horizon). The badge says what was
 * drawn, so it reads this rather than working the sun out again: a model with no report is
 * not relit yet, and the badge says nothing about it. Loaded only with the relight.
 */
export interface SunReport {
  /** The scene time of the frame drawn, ISO 8601, to the minute. */
  at: string;
  /** Sine of the sun's elevation at the scan's centre (NaN where it is not placed). */
  sine: number;
}

interface SunReportState {
  reports: Record<string, SunReport>;
  report: (assetId: string, report: SunReport | null) => void;
}

export const useSunReport = create<SunReportState>()((set) => ({
  reports: {},
  report: (assetId, report) =>
    set((s) => {
      const rest = Object.fromEntries(Object.entries(s.reports).filter(([id]) => id !== assetId));
      return { reports: report ? { ...rest, [assetId]: report } : rest };
    }),
}));
