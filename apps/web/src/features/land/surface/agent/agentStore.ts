import { create } from "zustand";

import type { Investigation } from "@twin/contracts";

import { RASTER_RUNS } from "../../rasterRuns";

/**
 * What the agent card remembers across its two homes (the right dock on wide screens, the
 * left sheet on narrow ones), so resizing the window never loses the person's place.
 */
interface AgentUi {
  /** A finding picked on the map: the card scrolls to it and marks it. */
  highlight: string | null;
  setHighlight: (id: string | null) => void;
  /** Bumped to move keyboard focus into the question box. */
  focusTick: number;
  focusInput: () => void;
  /** An investigation the person closed while it ran; it is not reopened on its own. */
  closed: string | null;
  setClosed: (id: string | null) => void;
  /** The land whose still-running question was already brought back once. */
  autoShown: string | null;
  setAutoShown: (landId: string) => void;
}

export const useAgentUi = create<AgentUi>((set) => ({
  highlight: null,
  setHighlight: (highlight) => set({ highlight }),
  focusTick: 0,
  focusInput: () => set((state) => ({ focusTick: state.focusTick + 1 })),
  closed: null,
  setClosed: (closed) => set({ closed }),
  autoShown: null,
  setAutoShown: (autoShown) => set({ autoShown }),
}));

/** Titles of the analyses the lenses start; the agent's own questions are everything else. */
const ANALYSIS_TITLES = new Set([
  "Land overview",
  ...Object.values(RASTER_RUNS).map((run) => run.title),
  "Photographs and historical maps",
  "Ecological context and name review",
]);

/** A question put to the agent, rather than an analysis a lens ran. */
export function isAgentInvestigation(item: Investigation): boolean {
  if (item.kinds) return item.kinds.includes("investigation");
  return !ANALYSIS_TITLES.has(item.title) && !item.title.startsWith("Hourly solar assessment");
}
