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
}));
