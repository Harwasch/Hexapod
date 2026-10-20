import { create } from "zustand";

import type { Coverage, ViewRule, ViewRuleEnd } from "@/lib/compositor";

/** What the scene says a per-cell view rule draws now (cesium/compositorSession.ts). */
export interface ViewRuleLegend {
  /** The owner plane's cell, metres. */
  cellM: number;
  /** Each date taking part, oldest first: how its coverage was read, and the cells it owns. */
  dates: { epochId: string; coverage: Coverage | "loading"; cells: number }[];
  /** Cells inferred layers own (only where no date saw anything). */
  inferredCells: number;
  /** Tilesets past the sixteen one primitive tells apart: not drawn under the rule. */
  left: number;
}

/**
 * The per-cell view rule (SiteManager `setViewRule`, ADR 0017), as the scene says it -- the
 * scene writes it, as it writes the inferred layers' store: which site draws which rule, why
 * the last one ended when nobody asked (`ended`), and the legend. Not persisted: off by
 * default, every visit.
 */
interface ViewRuleState {
  active: { siteId: string; rule: ViewRule } | null;
  ended: ViewRuleEnd | null;
  legend: ViewRuleLegend | null;
}

export const useViewRule = create<ViewRuleState>()(() => ({
  active: null,
  ended: null,
  legend: null,
}));
