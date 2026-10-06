import { create } from "zustand";

import type { MeasureMode } from "./ui";

export interface MeasurementPoint {
  longitude: number;
  latitude: number;
  height: number;
}

export interface Measurement {
  id: string;
  mode: MeasureMode;
  points: MeasurementPoint[];
  /** Geodesic (surface) distance in metres, for distance mode. */
  distance2dM?: number;
  /** Straight-line 3D distance in metres. */
  distance3dM?: number;
  areaM2?: number;
  heightDeltaM?: number;
  elevationM?: number;
  complete: boolean;
  createdAt: number;
}

interface MeasurementsState {
  items: Measurement[];
  upsert: (measurement: Measurement) => void;
  remove: (id: string) => void;
  clear: () => void;
  /** Puts removed measurements back, in the order they were made (an undo; `MeasurePanel`). */
  restore: (items: readonly Measurement[]) => void;
}

export const useMeasurements = create<MeasurementsState>()((set) => ({
  items: [],
  upsert: (measurement) =>
    set((s) => {
      const index = s.items.findIndex((m) => m.id === measurement.id);
      if (index === -1) return { items: [...s.items, measurement] };
      const items = s.items.slice();
      items[index] = measurement;
      return { items };
    }),
  remove: (id) => set((s) => ({ items: s.items.filter((m) => m.id !== id) })),
  clear: () => set({ items: [] }),
  restore: (back) =>
    set((s) => {
      const known = new Set(s.items.map((m) => m.id));
      const items = [...s.items, ...back.filter((m) => !known.has(m.id))];
      return { items: items.sort((a, b) => a.createdAt - b.createdAt) };
    }),
}));
