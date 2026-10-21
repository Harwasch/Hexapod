/**
 * Where a per-cell view rule (cesium/compositorSession.ts; ADR 0017 "Per-cell view rules")
 * meets PlayCanvas when PlayCanvas draws the site's scan: per asset id -- the scan the renderer
 * draws (SiteManager `scanTarget`) -- the rule's owner plane and what each tileset it names is,
 * or none. The rule puts it here as it changes; the renderer drawing that asset watches it
 * (playcanvasBackend.ts) and draws every date in its one sort.
 *
 * **What the renderer draws from it.** Every tile it draws comes from a tileset (the scan's,
 * another date's, an inferred layer's, a split object's), and `sources` says, by that tileset's
 * URL, which source of the plane it is. Each splat is kept only where its source owns the cell
 * its centre sits in, faded by its own spread near a seam as CesiumJS fades it
 * (`RULE_WEIGHT_GLSL`, lib/compositor.ts `compositorWeight`); a split object is drawn whole, or
 * not at all, by the cell under its centre. A tile of no listed source draws nothing: the rule
 * fails closed. The other dates are streamed by the renderer beside its scan (`dates`,
 * ruleDates.ts), each with its inferred layers and split objects.
 *
 * Both sides load on demand (the rule's session with the first rule, the renderer when chosen),
 * and so does this: nothing of it is on the globe's first load.
 */

import type { CellLookView } from "@/lib/cellLook";
import { NO_OWNER, type OwnerPlane } from "@/lib/compositor";

/** Cells a row of the owner texture holds (R8), as CesiumJS's (splatCompositor.ts). */
export const RULE_OWNER_WIDTH = 4096;

/** What a tileset the rule names is. */
export interface RuleSource {
  /** Its index among the plane's sources (lib/compositor.ts `CompositorSource.source`). */
  source: number;
  /** A split object: drawn whole by the date owning the cell under its centre, or not at all. */
  whole?: boolean;
}

/** A date the renderer streams beside its scan. */
export interface RuleDate {
  /** Its tileset's URL, absolute, as the renderer fetches it and `sources` names it. */
  url: string;
  assetId: string;
  /** Its tileset's root extras: its inferred layers and split objects. */
  extras: unknown;
  /** Where its tiles go in the scan's frame (column-major 4x4), or null where they are. */
  placement: number[] | null;
  /**
   * The root its inferred layers and split objects are placed from, `S · P⁻¹` (its own root
   * `S`, its placement `P`): a layer placed from it lands where the date's tiles do.
   */
  root: number[];
}

/** A rule, as the renderer drawing the scan is handed it. */
export interface ScanRuleView {
  /** The owner plane's cells in the scan's frame: (i0, j0, ni, nj), and their size (m). */
  plane: readonly [number, number, number, number];
  cellM: number;
  /** The plane's bytes as the owner texture holds them (`ruleOwnerRows`). */
  owners: { readonly data: Uint8Array; readonly height: number };
  /** By tileset URL: the scan's, each date's, each inferred layer's, each split object's. */
  sources: ReadonlyMap<string, RuleSource>;
  /** The source of a scan streamed natively (one entity, no tiles of a tileset), or `NO_OWNER`. */
  native: number;
  /** The inferred sources tinted as invented (Highlight invented): bit k for source k. */
  tinted: number;
  /** How far from a seam a splat is drawn whole, in its own sigmas (`SEAM_REACH`; 0: centres). */
  reach: number;
  /** False draws every splat as it was: a harness's comparison. */
  enabled: boolean;
  /** The site's other dates the rule draws, streamed beside the scan. */
  dates: readonly RuleDate[];
  /** Data age under the rule: each cell by its owner's age (`ownerAgeGrid`), else null. */
  age: CellLookView | null;
}

/** The owner plane's bytes in whole rows of `RULE_OWNER_WIDTH`, `NO_OWNER` after. */
export function ruleOwnerRows(plane: Pick<OwnerPlane, "owners">): {
  data: Uint8Array;
  height: number;
} {
  const height = Math.max(1, Math.ceil(plane.owners.length / RULE_OWNER_WIDTH));
  const data = new Uint8Array(height * RULE_OWNER_WIDTH).fill(NO_OWNER);
  data.set(plane.owners);
  return { data, height };
}

/**
 * The rule's weight on a splat, for PlayCanvas's work-buffer modifier (playcanvasBackend.ts):
 * lib/compositor.ts `compositorWeight` in GLSL, as CesiumJS's visibility part draws it
 * (splatCompositor.ts). It depends on no eye, so the copy pass applies it once per copy of a
 * tile, on the opacity. A splat's spread across the plane is its covariance's, from the
 * rotation (x, y, z, w) and scales the copy pass hands `modifySplatRotationScale` -- kept in
 * `hexapodRuleRotation` and `hexapodRuleScale` -- in the work buffer's frame, which is the
 * scan's and the plane's. Uniforms: `uRuleOwners` (R8), `uRulePlane` (i0, j0, ni, nj),
 * `uRuleParams` (on, cell m, reach in sigmas, -), and per tile `uRuleSource` (source, whole,
 * and for a split object its centre's x and y).
 */
export const RULE_WEIGHT_GLSL = `
uniform highp sampler2D uRuleOwners;
uniform vec4 uRulePlane;
uniform vec4 uRuleParams;
uniform vec4 uRuleSource;
vec4 hexapodRuleRotation = vec4(0.0, 0.0, 0.0, 1.0);
vec3 hexapodRuleScale = vec3(1.0);
int hexapodRuleOwner(ivec2 c) {
    int ni = int(uRulePlane.z + 0.5);
    if (c.x < 0 || c.y < 0 || c.x >= ni || c.y >= int(uRulePlane.w + 0.5)) {
        return ${String(NO_OWNER)};
    }
    int linear = c.y * ni + c.x;
    return int(texelFetch(uRuleOwners, ivec2(linear & ${String(RULE_OWNER_WIDTH - 1)}, linear >> ${String(Math.log2(RULE_OWNER_WIDTH))}), 0).r * 255.0 + 0.5);
}
// Whether cell c is across a seam from owner: another date owns it (nobody is no seam).
bool hexapodRuleSeam(ivec2 c, int owner) {
    int other = hexapodRuleOwner(c);
    return other != owner && other != ${String(NO_OWNER)};
}
float hexapodRuleWeight(vec3 center) {
    if (uRuleParams.x < 0.5) {
        return 1.0;
    }
    int source = int(uRuleSource.x + 0.5);
    if (source == ${String(NO_OWNER)}) {
        return 0.0;
    }
    bool whole = uRuleSource.y > 0.5;
    float cell = uRuleParams.y;
    vec2 g = (whole ? uRuleSource.zw : center.xy) / cell;
    vec2 f = floor(g);
    ivec2 c = ivec2(f) - ivec2(uRulePlane.xy);
    int owner = hexapodRuleOwner(c);
    if (owner != source) {
        return 0.0;
    }
    if (whole) {
        return 1.0;
    }
    // The splat's spread across the plane: the x and y diagonal of R diag(s^2) R^T.
    vec4 q = hexapodRuleRotation;
    vec3 rx = vec3(1.0 - 2.0 * (q.y * q.y + q.z * q.z), 2.0 * (q.x * q.y - q.w * q.z), 2.0 * (q.x * q.z + q.w * q.y));
    vec3 ry = vec3(2.0 * (q.x * q.y + q.w * q.z), 1.0 - 2.0 * (q.x * q.x + q.z * q.z), 2.0 * (q.y * q.z - q.w * q.x));
    vec3 s2 = hexapodRuleScale * hexapodRuleScale;
    vec2 sigma = sqrt(max(vec2(dot(rx * rx, s2), dot(ry * ry, s2)), vec2(1.0e-12)));
    vec2 t = (g - f) * cell;
    float d = 1.0e9;
    if (hexapodRuleSeam(c - ivec2(1, 0), owner)) {
        d = min(d, t.x / sigma.x);
    }
    if (hexapodRuleSeam(c + ivec2(1, 0), owner)) {
        d = min(d, (cell - t.x) / sigma.x);
    }
    if (hexapodRuleSeam(c - ivec2(0, 1), owner)) {
        d = min(d, t.y / sigma.y);
    }
    if (hexapodRuleSeam(c + ivec2(0, 1), owner)) {
        d = min(d, (cell - t.y) / sigma.y);
    }
    return uRuleParams.z > 0.0 ? smoothstep(0.5 * uRuleParams.z, uRuleParams.z, d) : 1.0;
}
`;

/**
 * A splat's spread across the plane (metres: east, north) from its rotation (x, y, z, w) and
 * scales, as `RULE_WEIGHT_GLSL` works it out: for tests, beside lib/compositor.ts.
 */
export function ruleSigma(
  rotation: readonly [number, number, number, number],
  scale: readonly [number, number, number],
): [number, number] {
  const [x, y, z, w] = rotation;
  const rx = [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)];
  const ry = [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)];
  const s2 = scale.map((s) => s * s);
  const dot = (r: number[]) => r.reduce((sum, v, k) => sum + v * v * (s2[k] ?? 0), 0);
  return [Math.sqrt(Math.max(dot(rx), 1e-12)), Math.sqrt(Math.max(dot(ry), 1e-12))];
}

type Listener = (view: ScanRuleView | null) => void;

const views = new Map<string, ScanRuleView>();
const listeners = new Map<string, Set<Listener>>();

/** Puts asset `assetId`'s rule (null: none) where the renderer drawing it reads it. */
export function publishScanRule(assetId: string, view: ScanRuleView | null): void {
  if ((views.get(assetId) ?? null) === view) return;
  if (view) views.set(assetId, view);
  else views.delete(assetId);
  for (const listener of listeners.get(assetId) ?? []) listener(view);
}

/** Calls `listener` with asset `assetId`'s rule now and on every change; returns the remover. */
export function watchScanRule(assetId: string, listener: Listener): () => void {
  let set = listeners.get(assetId);
  if (!set) listeners.set(assetId, (set = new Set()));
  set.add(listener);
  listener(views.get(assetId) ?? null);
  return () => {
    set.delete(listener);
    if (set.size === 0) listeners.delete(assetId);
  };
}

/** What the renderer drew of a rule, last time it was asked. */
export interface ScanRuleReading {
  renderer: string;
  /** Whether it draws a rule now. */
  on: boolean;
  /** Its tiles drawn under the rule, by source (`NO_OWNER`: a tile of no listed source). */
  sources: Record<number, number>;
  /** The other dates it streams: how many, their tiles on screen, gaussians, fetches. */
  dates: { dates: number; tiles: number; gaussians: number; loading: number };
}

const readings = new Map<string, () => ScanRuleReading>();

/** Registers how asset `assetId`'s renderer reads out its rule; returns the remover. */
export function reportScanRule(assetId: string, read: () => ScanRuleReading): () => void {
  readings.set(assetId, read);
  return () => {
    if (readings.get(assetId) === read) readings.delete(assetId);
  };
}

/** What the renderer drawing asset `assetId` says of its rule, or null when none does. */
export function scanRuleReading(assetId: string): ScanRuleReading | null {
  return readings.get(assetId)?.() ?? null;
}
