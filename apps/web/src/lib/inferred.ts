/**
 * Inferred layers: what an image model filled in where a scan never looked
 * (tools/captures/teacher_fill.py). Never mixed into the measured tileset: each is a tileset
 * of its own, declared on the measured one's root as `extras.inferredLayers`, carrying
 * `extras.evidence` on its own root. The viewer draws it beside the scan, says it is
 * inferred, and lets it be hidden.
 */

export interface InferredEvidence {
  kind: "inferred";
  /** The model that painted it, e.g. `nvidia-fixer`. */
  filler: string;
  views: number;
  gaussians: number;
  /** 0..1: how near, on average, its pixels were to something measured. */
  meanConfidence: number;
}

export interface InferredLayerRef {
  /** The layer's tileset.json, relative to the measured tileset's. */
  uri: string;
  evidence: InferredEvidence;
}

function finite(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

/** `extras.evidence` of an inferred tileset, or null when it is not one. */
export function evidenceOf(extras: unknown): InferredEvidence | null {
  const e = (extras as { evidence?: Record<string, unknown> } | null | undefined)?.evidence;
  if (e?.kind !== "inferred") return null;
  return {
    kind: "inferred",
    filler: typeof e.filler === "string" ? e.filler : "unknown",
    views: finite(e.views),
    gaussians: finite(e.gaussians),
    meanConfidence: Math.min(1, Math.max(0, finite(e.meanConfidence))),
  };
}

/** The inferred layers a measured tileset's root declares (malformed entries skipped). */
export function inferredLayersOf(extras: unknown): InferredLayerRef[] {
  const list = (extras as { inferredLayers?: unknown } | null | undefined)?.inferredLayers;
  if (!Array.isArray(list)) return [];
  const out: InferredLayerRef[] = [];
  for (const entry of list as { uri?: unknown; evidence?: unknown }[]) {
    if (typeof entry?.uri !== "string" || entry.uri.length === 0) continue;
    const evidence = evidenceOf({ evidence: entry.evidence });
    if (evidence) out.push({ uri: entry.uri, evidence });
  }
  return out;
}

/** The layer's URL beside the measured tileset's, keeping its query (a signed URL's). */
export function resolveLayerUrl(tilesetUrl: string, uri: string): string {
  const parent = new URL(tilesetUrl, globalThis.location?.href ?? "http://localhost/");
  const resolved = new URL(uri, parent);
  if (!resolved.search && resolved.origin === parent.origin) resolved.search = parent.search;
  return resolved.href;
}

/**
 * How inferred layers are drawn (a viewer's setting, `inferredStyle`): as they are, marked
 * unmistakably (`INFERRED_HIGHLIGHT`), or not at all. The measured splats are never touched.
 */
export type InferredStyle = "show" | "highlight" | "hide";

/** Every style, in the order the viewer offers them. */
export const INFERRED_STYLES: readonly InferredStyle[] = ["show", "highlight", "hide"];

export const INFERRED_STYLE_LABELS: Record<InferredStyle, string> = {
  show: "Show",
  highlight: "Highlight",
  hide: "Hide",
};

/** The one line that says what inferred is, wherever it can be seen. */
export const INFERRED_LEGEND = "Inferred: generated where no camera saw. Not measured.";

/** The purple accent of inferred content, sRGB (the legend's swatch is the same colour). */
export const INFERRED_PURPLE = "#b36bff";

/**
 * What Highlight does to an inferred splat (cesium/inferredLayers.ts): its colour pulled toward
 * the purple (`tint`: `INFERRED_PURPLE` as the splats' own colours are stored, by `a`), every
 * other band of `stripeM` metres across the layer darkened to `stripeDark`, and its opacity
 * times `opacity` -- so a fill reads as hatched, see-through purple whatever colour it was
 * painted.
 */
export const INFERRED_HIGHLIGHT = {
  tint: [0.7, 0.42, 1.0, 0.7] as const,
  stripeM: 0.35,
  stripeDark: 0.45,
  opacity: 0.8,
};

/**
 * What Highlight does to one inferred splat's colour (straight alpha) at `position` (metres, in
 * its layer's frame): the reference every renderer's shader follows (`INFERRED_COLOR_GLSL`, the
 * overlay's `LAYER_LOOK_GLSL`), for tests. As painted when `highlight` is off.
 */
export function inferredHighlightColor(
  color: readonly [number, number, number, number],
  position: readonly [number, number, number],
  highlight: boolean,
): [number, number, number, number] {
  if (!highlight) return [color[0], color[1], color[2], color[3]];
  const { tint, stripeM, stripeDark, opacity } = INFERRED_HIGHLIGHT;
  const along = (position[0] + position[1] + position[2]) * 0.57735027;
  const band = along / stripeM - Math.floor(along / stripeM);
  const shade = band < 0.5 ? 1 : stripeDark;
  const mix = (c: number, t: number): number => (c + (t - c) * tint[3]) * shade;
  return [
    mix(color[0], tint[0]),
    mix(color[1], tint[1]),
    mix(color[2], tint[2]),
    color[3] * opacity,
  ];
}

/** One line for a reader: what painted it and from how much. */
export function describeEvidence(e: InferredEvidence): string {
  const confidence = Math.round(e.meanConfidence * 100);
  return `Inferred by ${e.filler} from ${e.views} view${e.views === 1 ? "" : "s"}, ${confidence}% mean confidence. Not measured.`;
}
