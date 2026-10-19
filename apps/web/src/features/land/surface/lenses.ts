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

import { useLandContext } from "@/state/landContext";

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

type Section = "discover" | "records" | "inventory" | "ecology" | "scenarios" | "actions";

/** The workspace section a lens stood in before lenses, for views read by older clients. */
export function sectionOfLens(lens: LensId): Section {
  const sections: Record<LensId, Section> = {
    overview: "discover",
    water: "discover",
    soil: "discover",
    ecology: "ecology",
    solar: "scenarios",
    records: "records",
    assets: "inventory",
    plans: "actions",
  };
  return sections[lens];
}

/** The lens for a view saved before lenses, which names only a workspace section. */
export function lensOfSection(section: Section | undefined): LensId {
  const lenses: Record<Section, LensId> = {
    discover: "overview",
    records: "records",
    inventory: "assets",
    ecology: "ecology",
    scenarios: "plans",
    actions: "plans",
  };
  return section ? lenses[section] : "overview";
}

interface LensState {
  lens: LensId;
  setLens: (lens: LensId) => void;
}

export const useLens = create<LensState>((set) => ({
  lens: "overview",
  setLens: (lens) => set({ lens }),
}));

// Picking something on the map opens the lens it belongs to: an asset clicked on the map
// (`LandContextBridge` → `selectInventory`) shows Assets; a cited survey shows Ecology.
useLandContext.subscribe((next, previous) => {
  if (next.selectedInventoryId && next.selectedInventoryId !== previous.selectedInventoryId)
    useLens.getState().setLens("assets");
  else if (next.selectedSurveyId && next.selectedSurveyId !== previous.selectedSurveyId)
    useLens.getState().setLens("ecology");
});
