/**
 * What the per-cell view rules borrow from the globe's first load, lent to them at runtime
 * rather than imported (ADR 0017 § "Per-cell view rules under PlayCanvas", Cost).
 *
 * The rules' own code loads on demand: the rule's session (cesium/compositorSession.ts), what
 * reaches PlayCanvas (scanRule.ts) and the other dates' streaming (ruleDates.ts). They need
 * classes and helpers the globe already loads up front -- the tile streamer, a scan's inferred
 * layers and split objects, the parsers. Imported, those modules would be shared by the entry
 * and an on-demand chunk, and the bundler would cut them out of the entry into chunks of their
 * own: no new code on the first load, but more of it (1,602,079 → 1,603,880 B gzip, measured).
 * Lent, nothing the first load holds moves: SiteManager hands this tuple to the rule's session
 * when a rule starts, which lends it on to the renderer (scanRule.ts `lendRuleKit`). A tuple,
 * not an object, so the first load carries no property names for it; scanRule.ts `namedKit`
 * names its parts on the on-demand side. The on-demand code imports nothing from here but the
 * type: a value imported from this module would bring its imports along.
 */

import { splitObjectsOf } from "@/lib/sceneObjects";
import { TileStreamer } from "@/view/stream";
import { parseTileset } from "@/view/tiles";

import { pickedLayers } from "../inferredLayers";
import { LAYER_BUDGET_SHARE, layerView, ScanLayers } from "./scanLayers";
import { ScanObjects } from "./scanObjects";

export type RuleKit = readonly [
  tileStreamer: typeof TileStreamer,
  parseTileset: typeof parseTileset,
  scanLayers: typeof ScanLayers,
  scanObjects: typeof ScanObjects,
  layerView: typeof layerView,
  layerBudgetShare: number,
  pickedLayers: typeof pickedLayers,
  splitObjectsOf: typeof splitObjectsOf,
];

export const RULE_KIT: RuleKit = [
  TileStreamer,
  parseTileset,
  ScanLayers,
  ScanObjects,
  layerView,
  LAYER_BUDGET_SHARE,
  pickedLayers,
  splitObjectsOf,
];
