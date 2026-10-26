import { create } from "zustand";

import { useAsCaptured } from "./asCaptured";
import { useLivingLab } from "./livingLab";

export type LightArm = "off" | "a" | "b";
interface LightState {
  arm: LightArm;
  /** Fraction of the baked equinox solar day, 07:00–17:00. */
  day: number;
  seconds: number | null;
  cloud: number;
  diffuse: boolean;
  manifest: string;
  status: string;
  set: (patch: Partial<Omit<LightState, "set">>) => void;
}
const initialManifest = (): string => {
  if (typeof window === "undefined") return "";
  return new URLSearchParams(window.location.search).get("lightManifest") ?? "";
};
export const useLivingLight = create<LightState>()((set) => ({
  arm: "off",
  day: 0.5,
  seconds: null,
  cloud: 0.55,
  diffuse: false,
  manifest: initialManifest(),
  status: "No baked lighting loaded",
  set,
}));
export function effectiveLightArm(): LightArm {
  return useLivingLab.getState().open && !useAsCaptured.getState().on
    ? useLivingLight.getState().arm
    : "off";
}
/** Matches light_bake.sun_path: equinox, latitude 45.21°, solar hours 07–17. */
export function lightSun(day: number): [number, number, number] {
  const hour = ((-75 + 150 * Math.min(1, Math.max(0, day))) * Math.PI) / 180;
  const lat = (45.21 * Math.PI) / 180;
  return [-Math.sin(hour), -Math.sin(lat) * Math.cos(hour), Math.cos(lat) * Math.cos(hour)];
}
