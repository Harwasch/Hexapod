import {
  Boxes,
  ClipboardList,
  Droplets,
  Mountain,
  ScrollText,
  Shovel,
  Sprout,
  Sun,
  type LucideIcon,
} from "lucide-react";
import { create } from "zustand";

/**
 * The ways to look at a piece of land. Each lens draws on the map and keeps one card in the
 * left dock: a plain sentence, a few facts that point at the map, and the full tools a tap away.
 */
export type LensId =
  "overview" | "water" | "soil" | "ecology" | "solar" | "records" | "assets" | "plans";

export interface LensMeta {
  id: LensId;
  label: string;
  /** The dot in the dock and the lens's map colour. */
  color: string;
  icon: LucideIcon;
}

const OVERVIEW: LensMeta = { id: "overview", label: "Overview", color: "#b9edc6", icon: Mountain };

export const LENSES: readonly LensMeta[] = [
  OVERVIEW,
  { id: "water", label: "Water", color: "#8fbcf2", icon: Droplets },
  { id: "soil", label: "Soil", color: "#e2a95a", icon: Shovel },
  { id: "ecology", label: "Ecology", color: "#93d37f", icon: Sprout },
  { id: "solar", label: "Solar", color: "#ffcf5a", icon: Sun },
  { id: "records", label: "Records", color: "#d9c3a0", icon: ScrollText },
  { id: "assets", label: "Assets", color: "#88c8e5", icon: Boxes },
  { id: "plans", label: "Plans", color: "#c7a6f0", icon: ClipboardList },
];

export function lensMeta(id: LensId): LensMeta {
  return LENSES.find((lens) => lens.id === id) ?? OVERVIEW;
}

interface LensState {
  lens: LensId;
  setLens: (lens: LensId) => void;
}

export const useLens = create<LensState>((set) => ({
  lens: "overview",
  setLens: (lens) => set({ lens }),
}));
