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

/** One line for a reader: what painted it and from how much. */
export function describeEvidence(e: InferredEvidence): string {
  const confidence = Math.round(e.meanConfidence * 100);
  return `Inferred by ${e.filler} from ${e.views} view${e.views === 1 ? "" : "s"}, ${confidence}% mean confidence. Not measured.`;
}
