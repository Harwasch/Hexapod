/**
 * Command bar intents. A small deterministic parser that maps natural phrases
 * onto the app's imperative API. It is the seam where an LLM agent plugs in:
 * the same actions become tools, and unmatched text becomes a model call.
 */

export interface IntentContext {
  flyToPlace: (query: string) => Promise<string>;
  flyToSite: (query: string) => string | null;
  toggleLayer: (query: string, visible: boolean | null) => string | null;
  selectMachine: (id: string) => string | null;
  selectZone: (id: string) => string | null;
  openPlan: (query: string | null) => string;
  setView: (view: "map" | "plan" | "fleet") => string;
  startMeasure: (mode: "distance" | "area" | "height" | "elevation" | "point") => string;
  camera: (action: "north" | "top-down" | "home" | "explore") => string;
  openSettings: () => string;
}

export async function runIntent(input: string, ctx: IntentContext): Promise<string> {
  const text = input.trim();
  const lower = text.toLowerCase();
  if (!text) return "Say where to go or what to do.";

  const machine = /\b(tr-\d{2})\b/i.exec(text)?.[1]?.toUpperCase();
  if (machine && /\b(select|show|find|locate|where|go to|fly to)\b/.test(lower))
    return ctx.selectMachine(machine) ?? `I don't know a machine called ${machine}.`;
  const zone = /\b(z-\d{2})\b/i.exec(text)?.[1]?.toUpperCase();
  if (zone) return ctx.selectZone(zone) ?? `I don't know a zone called ${zone}.`;

  if (/\b(measure|distance between|how far)\b/.test(lower)) {
    const mode = /\barea\b/.test(lower)
      ? "area"
      : /\bheight\b/.test(lower)
        ? "height"
        : /\belevation\b/.test(lower)
          ? "elevation"
          : "distance";
    return ctx.startMeasure(mode);
  }
  if (/\b(reset north|face north|north up)\b/.test(lower)) return ctx.camera("north");
  if (/\b(top[- ]down|look down|overhead)\b/.test(lower)) return ctx.camera("top-down");
  if (/\b(explore|walk|ground level|first person)\b/.test(lower)) return ctx.camera("explore");
  if (/\b(earth|planet|zoom out all the way|home)\b/.test(lower) && !/\bfly to\b/.test(lower))
    return ctx.camera("home");
  if (/\bsettings?\b/.test(lower)) return ctx.openSettings();
  if (/\b(fleet|machines|robots)\b/.test(lower) && /\b(show|open|list|view)\b/.test(lower))
    return ctx.setView("fleet");
  if (/\bplans?\b/.test(lower)) {
    const named = /\bplan (?:for|about|on)?\s*(.+)$/.exec(lower)?.[1] ?? null;
    return ctx.openPlan(named && named !== "s" ? named : null);
  }
  if (/\bmap\b/.test(lower) && /\b(show|back to|open)\b/.test(lower)) return ctx.setView("map");

  const layerToggle =
    /\b(show|turn on|enable|hide|turn off|disable|toggle)\b\s+(?:the\s+)?(.+?)(?:\s+layer)?$/.exec(
      lower,
    );
  if (layerToggle) {
    const verb = layerToggle[1] ?? "";
    const visible = /^(hide|turn off|disable)$/.test(verb)
      ? false
      : verb === "toggle"
        ? null
        : true;
    const result = ctx.toggleLayer(layerToggle[2] ?? "", visible);
    if (result) return result;
  }

  const fly =
    /^(?:fly|go|take me|jump|navigate)\s+(?:to|towards)?\s*(.+)$/.exec(lower) ??
    /^(?:where is|find)\s+(.+)$/.exec(lower);
  const place = fly?.[1] ?? text;
  const site = ctx.flyToSite(place);
  if (site) return site;
  return ctx.flyToPlace(place);
}
