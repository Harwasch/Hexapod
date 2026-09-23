/**
 * The motion rig: a skeleton of a few dozen nodes, deliberately *not* a list of splat indices.
 *
 * Keying group membership by splat index would couple the runtime to the tiler — PLY order, the
 * keep-mask, opacity filtering, SPZ encoding, decode and snapshot aggregation all sit between
 * the two, and any change there desynchronises the mapping with no exception thrown. A skeleton
 * is geometric, so it survives re-tiling: the runtime re-derives membership with
 * `assignSplatsToNodes`.
 */

import { type Vec3 } from "./vec";

/** Where a node sits in the tree's structural hierarchy. Drives the default angular limit. */
export type SkeletonBand = "trunk" | "branch" | "leaf";

export const SKELETON_BANDS: readonly SkeletonBand[] = ["trunk", "branch", "leaf"];

/**
 * Default per-band cap on a node's *local* bend angle, radians. A node can never exceed its cap
 * (the saturation is smooth, so it approaches but never reaches it); a chain of nodes can of
 * course accumulate more than any single cap, which is what `maxNodeDisplacements` accounts for.
 */
export const DEFAULT_BAND_ANGLE_LIMIT_RAD: Readonly<Record<SkeletonBand, number>> = {
  trunk: 0.1,
  branch: 0.25,
  leaf: 0.35,
};

/** Largest node count a rig may have: assignments are returned as a `Uint16Array`. */
export const MAX_RIG_NODES = 65535;

/** One joint of the skeleton. JSON-serialisable: plain numbers, strings and arrays only. */
export interface SkeletonNode {
  /** Stable, human-readable identity. Also seeds this node's noise phase, so it affects motion. */
  readonly id: string;
  /** Index into `MotionRig.nodes` of the parent, or `-1` for the root. Always less than own index. */
  readonly parent: number;
  /** Rest position in the rig's local ENU frame: `[east, north, up]` metres, +Z up. */
  readonly position: Vec3;
  /**
   * Approximate **woody** cross-sectional radius at this joint, metres.
   *
   * No longer merely descriptive. `modes.ts` reads it for natural frequency (`ω ∝ radius /
   * length²`) and damping, and `flutter.ts` reads it to decide how hard a node's splats
   * shimmer and whether they shimmer at all. A rig whose radii are wrong will have
   * systematically wrong frequencies, and nothing downstream can tell.
   *
   * **`skeleton.py` measures this quantity as of C2, but only where a cloud can carry it.**
   * `woody_radius` projects a cluster onto the plane across its limb and takes the RMS radius
   * of the dense ring in that cross-section — a limb is a hollow shell in a splat cloud, so its
   * cross-section is a ring and foliage is a diffuse halo around the same axis. Where the
   * cross-section shows no such ring, which on the synthetic fixture is 130 nodes of 189, it
   * reports the cloud's own resolution limit (about half the point spacing) instead, and that
   * value scales with the capture's density rather than with the tree. Median recovered/true
   * radius on the fixture is 1.18 where the old `_rms_radius` was 5.56; the extracted crown now
   * sits at a median 15 Hz with 75 of 189 nodes on the clamp and 164 fluttering, against 30 Hz,
   * 123 and 9 before, and 11 Hz, 58 and 173 for the same skeleton carrying the true radii.
   *
   * None of that has been checked against a real capture, because there is not one in this
   * repository. See docs/LIVING_SURVEY.md, "What would break on a rig that is not this one".
   */
  readonly radius: number;
  /** Dimensionless resistance to bending, `> 0`. Higher bends less; bend angle scales as `1/stiffness`. */
  readonly stiffness: number;
  readonly band: SkeletonBand;
  /** Optional override of the band's default angular limit, radians. Omit to use the default. */
  readonly maxAngleRad?: number;
}

/** A complete rig. Round-trips losslessly through `JSON.stringify` / `JSON.parse`. */
export interface MotionRig {
  /**
   * Nodes in topological order: `nodes[i].parent < i` always, and `nodes[0]` is the sole root.
   * The root is the anchor — its transform is exactly identity at every time and every wind.
   */
  readonly nodes: readonly SkeletonNode[];
  /**
   * Checksum of the canonical splat positions this rig was derived from, from
   * `checksumPositions`. An exact byte identity check, not a tolerant one: the runtime refuses
   * to deform rather than deform the wrong splats.
   */
  readonly canonicalChecksum: string;
  /** Length unit of `position` and `radius`. Metres, always; present so a reader need not guess. */
  readonly units: "meters";
  /** Free text: what produced this rig, from what capture. Carried into the UI provenance panel. */
  readonly sourceNote: string;
}

/** The angular limit in force for a node, radians. */
export function nodeAngleLimit(node: SkeletonNode): number {
  const override = node.maxAngleRad;
  if (override !== undefined && Number.isFinite(override) && override > 0) return override;
  return DEFAULT_BAND_ANGLE_LIMIT_RAD[node.band];
}

/** Indices from `index` up to (but excluding) the root's parent, nearest ancestor first. */
export function ancestorsOf(rig: MotionRig, index: number): number[] {
  const chain: number[] = [];
  let cursor = index;
  let guard = 0;
  while (cursor >= 0 && guard <= rig.nodes.length) {
    chain.push(cursor);
    const node = rig.nodes[cursor];
    if (node === undefined) break;
    cursor = node.parent;
    guard += 1;
  }
  return chain;
}

/** Height of a node above the root node, metres. Negative below it. */
export function heightAboveRoot(rig: MotionRig, index: number): number {
  const node = rig.nodes[index];
  const root = rig.nodes[0];
  if (node === undefined || root === undefined) return 0;
  return node.position[2] - root.position[2];
}

function isVec3(value: unknown): value is Vec3 {
  return (
    Array.isArray(value) &&
    value.length === 3 &&
    value.every((v) => typeof v === "number" && Number.isFinite(v))
  );
}

function isBand(value: unknown): value is SkeletonBand {
  return value === "trunk" || value === "branch" || value === "leaf";
}

/**
 * Every structural problem with a rig, as human-readable strings. Empty means valid. Returning
 * all of them (rather than throwing on the first) is what makes a tool's output diagnosable.
 */
export function validateRig(rig: MotionRig): string[] {
  const issues: string[] = [];
  const { nodes } = rig;
  if (nodes.length === 0) {
    issues.push("rig has no nodes");
    return issues;
  }
  if (nodes.length > MAX_RIG_NODES) {
    issues.push(`rig has ${nodes.length} nodes, more than the ${MAX_RIG_NODES} a Uint16 can index`);
  }
  if (rig.units !== "meters")
    issues.push(`units must be "meters", got ${JSON.stringify(rig.units)}`);
  if (typeof rig.canonicalChecksum !== "string" || rig.canonicalChecksum.length === 0) {
    issues.push("canonicalChecksum must be a non-empty string");
  }
  const seenIds = new Set<string>();
  for (let i = 0; i < nodes.length; i += 1) {
    const node = nodes[i];
    if (node === undefined) {
      issues.push(`node ${i} is missing`);
      continue;
    }
    const where = `node ${i} (${node.id})`;
    if (typeof node.id !== "string" || node.id.length === 0) issues.push(`${where}: empty id`);
    if (seenIds.has(node.id)) issues.push(`${where}: duplicate id`);
    seenIds.add(node.id);
    if (i === 0) {
      if (node.parent !== -1) issues.push(`${where}: the first node must be the root (parent -1)`);
    } else if (!Number.isInteger(node.parent) || node.parent < 0 || node.parent >= i) {
      issues.push(`${where}: parent ${node.parent} must be an earlier index (topological order)`);
    }
    if (!isVec3(node.position)) issues.push(`${where}: position must be three finite numbers`);
    if (!Number.isFinite(node.radius) || node.radius < 0)
      issues.push(`${where}: radius must be >= 0`);
    if (!Number.isFinite(node.stiffness) || node.stiffness <= 0) {
      issues.push(`${where}: stiffness must be > 0`);
    }
    if (!isBand(node.band)) issues.push(`${where}: band must be trunk, branch or leaf`);
    const limit = node.maxAngleRad;
    if (limit !== undefined && (!Number.isFinite(limit) || limit <= 0)) {
      issues.push(`${where}: maxAngleRad, when present, must be > 0`);
    }
  }
  return issues;
}

/** Throws with every problem listed, or returns the rig unchanged. */
export function assertValidRig(rig: MotionRig): MotionRig {
  const issues = validateRig(rig);
  if (issues.length > 0) throw new Error(`invalid motion rig:\n  ${issues.join("\n  ")}`);
  return rig;
}

/** Canonical JSON for a rig. Stable key order, so two equal rigs give equal text. */
export function serializeRig(rig: MotionRig): string {
  return JSON.stringify({
    units: rig.units,
    canonicalChecksum: rig.canonicalChecksum,
    sourceNote: rig.sourceNote,
    nodes: rig.nodes.map((node) => {
      const out: Record<string, unknown> = {
        id: node.id,
        parent: node.parent,
        position: [node.position[0], node.position[1], node.position[2]],
        radius: node.radius,
        stiffness: node.stiffness,
        band: node.band,
      };
      if (node.maxAngleRad !== undefined) out.maxAngleRad = node.maxAngleRad;
      return out;
    }),
  });
}

function asRecord(value: unknown, what: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`motion rig: ${what} must be an object`);
  }
  return value as Record<string, unknown>;
}

/** Reads a string field. Anything else becomes the fallback rather than "[object Object]". */
function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

/** Reads a numeric field. Anything else becomes NaN, which `validateRig` then reports. */
function asNumber(value: unknown): number {
  return typeof value === "number" ? value : Number.NaN;
}

/** Parses rig JSON, validating structure. Throws rather than returning a half-trusted rig. */
export function parseRig(text: string): MotionRig {
  const root = asRecord(JSON.parse(text), "the top level");
  const rawNodes = root.nodes;
  if (!Array.isArray(rawNodes)) throw new Error("motion rig: nodes must be an array");
  const nodes: SkeletonNode[] = rawNodes.map((raw, i) => {
    const node = asRecord(raw, `nodes[${i}]`);
    const position = node.position;
    if (!isVec3(position)) throw new Error(`motion rig: nodes[${i}].position must be [x, y, z]`);
    const band = node.band;
    if (!isBand(band)) throw new Error(`motion rig: nodes[${i}].band is not a known band`);
    const limit = node.maxAngleRad;
    const base: SkeletonNode = {
      id: asString(node.id),
      parent: asNumber(node.parent),
      position: [position[0], position[1], position[2]],
      radius: asNumber(node.radius),
      stiffness: asNumber(node.stiffness),
      band,
    };
    return limit === undefined ? base : { ...base, maxAngleRad: asNumber(limit) };
  });
  const units = root.units;
  if (units !== "meters") throw new Error('motion rig: units must be "meters"');
  return assertValidRig({
    nodes,
    canonicalChecksum: asString(root.canonicalChecksum),
    units,
    sourceNote: asString(root.sourceNote),
  });
}

/**
 * FNV-1a over the raw bytes of the canonical positions, plus the splat count. Exact: any
 * differing bit changes the digest. Cheap enough to run on every re-attach (~150k splats is a
 * single linear pass over 1.8 MB).
 */
export function checksumPositions(positions: Float32Array): string {
  const bytes = new Uint8Array(positions.buffer, positions.byteOffset, positions.byteLength);
  let h = 0x811c9dc5;
  for (const byte of bytes) {
    h = Math.imul(h ^ byte, 0x01000193) >>> 0;
  }
  const count = Math.floor(positions.length / 3);
  return `fnv1a32:${count}:${h.toString(16).padStart(8, "0")}`;
}

/** Whether `positions` are bit-identical to the ones the rig was built from. */
export function positionsMatchChecksum(rig: MotionRig, positions: Float32Array): boolean {
  return checksumPositions(positions) === rig.canonicalChecksum;
}

/** Golden angle, radians: spreads branch azimuths without any two lining up. */
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

/**
 * The fixture tree's proportions. Mirrors the constants of the same names in
 * `tools/captures/synthetic_tree.py`, which generates the splats these nodes own.
 *
 * The numbers that matter are the radii, and they are not free. The first version of this
 * fixture gave a 1.15 m branch a 12 cm radius — a slenderness (length ÷ diameter) of 4.8, where
 * a real branch is 20–60 and a real twig 50–200. Every limb was 5–15× too thick. Since a
 * cantilever's fundamental goes as `radius / length²`, those branches came out ~10× too stiff,
 * landed above the forcing band, and rode rigid: the trunk did all of the visible work and the
 * crown, where a tree's shimmer actually comes from, did none. The motion model was behaving
 * correctly on a tree made of fence posts.
 */
/** The trunk ends at this fraction of the tree's height; the crown carries the rest. */
const TRUNK_TOP_FRACTION = 0.7;
/**
 * Radius of a terminal twig node, metres. Every other radius follows from it by da Vinci's
 * rule, so this one number sets the whole tree's thickness.
 *
 * One honest deviation, and it is a consequence of sampling rather than of taste: a real 6 m
 * tree carries thousands of twigs, this rig carries ~100, and `r_trunk = r_twig·√tips` means
 * the two ends cannot both be realistic at this node count. The tip radius is chosen so the
 * *trunk* lands where a real trunk is — slenderness ~15, fundamental ~0.5 Hz — and the twigs
 * come out near slenderness 40 rather than a real twig's 50–200. What the motion model reads is
 * the frequency, and that lands in the right band at every level.
 */
const TWIG_RADIUS_M = 0.0088;
/** Radius growth per segment towards a limb's base, on top of da Vinci's rule: limbs taper. */
const TRUNK_TAPER = 1.03;
const LIMB_TAPER = 1.06;
/** Tree height the limb lengths below are quoted at, metres; they scale with `heightM`. */
const HEIGHT_REFERENCE_M = 6;
/** Primary branch length at the lowest whorl and at the highest, metres at that height. */
const PRIMARY_LENGTH_M: readonly [number, number] = [1.6, 0.95];
/** Primary branch elevation above horizontal at the lowest and highest whorl, radians. */
const PRIMARY_ELEVATION_RAD: readonly [number, number] = [0.45, 0.82];
const SECONDARY_LENGTH_FRACTION = 0.56;
const TWIG_LENGTH_FRACTION = 0.85;
/** Elevation each generation adds to its parent's. Small, and negative at the twigs: elevation
 * accumulates down the chain, and generous gains put the tips past vertical. */
const SECONDARY_ELEVATION_GAIN_RAD = 0.1;
const TWIG_ELEVATION_GAIN_RAD = -0.05;
/** Half-width of the fan a limb's children spread through, in azimuth and in elevation. */
const CHILD_SPREAD_RAD = 1.15;
const CHILD_ELEVATION_SPREAD_RAD = 0.4;
/** Elevation a limb gains along its own length: limbs curve up rather than running straight. */
const LIMB_CURL_RAD = 0.16;
/** Stiffness runs between these, linearly in radius ÷ trunk-base radius. Invented. */
const STIFFNESS_MIN = 1;
const STIFFNESS_MAX = 8;
/**
 * Deterministic wobble on limb lengths and elevations, so no two limbs of a whorl are
 * congruent.
 *
 * Without it the three primaries of a whorl have the same length and the same elevation, hence
 * the same natural frequency, and differ only in a hashed phase — which is not enough to stop
 * them moving as a unit (sibling correlation measured 0.88). Real trees do not carry congruent
 * branches. The multipliers are irrational, so the sequence never repeats over any limb count,
 * and it is an integer sequence rather than a random draw: the rig has no seed.
 */
const LIMB_LENGTH_JITTER = 0.22;
const LIMB_ELEVATION_JITTER = 0.12;
const LENGTH_PHASE = 0.618033988749895;
const ELEVATION_PHASE = 0.7548776662466927;

/** A deterministic value in `[-1, 1)` from an integer. */
function wobble(index: number, phase: number): number {
  return 2 * (((index + 1) * phase) % 1) - 1;
}

/** A node under construction: everything but the radius, which is computed from the tips inward. */
interface DraftNode {
  id: string;
  parent: number;
  position: Vec3;
  band: SkeletonBand;
}

/**
 * Positions of a chain of `segments` nodes walking out from `start`.
 *
 * The elevation rises by `curl` over the whole chain, so a limb arcs upward instead of being a
 * straight spoke.
 */
function limbChain(
  start: Vec3,
  azimuth: number,
  elevation: number,
  length: number,
  segments: number,
  curl: number,
): Vec3[] {
  const out: Vec3[] = [];
  let x = start[0];
  let y = start[1];
  let z = start[2];
  const step = length / segments;
  for (let j = 0; j < segments; j += 1) {
    const el = elevation + (curl * (j + 1)) / segments;
    const horizontal = Math.cos(el) * step;
    x += horizontal * Math.cos(azimuth);
    y += horizontal * Math.sin(azimuth);
    z += Math.sin(el) * step;
    out.push([x, y, z]);
  }
  return out;
}

/** Where child `index` of `count` sits in its parent's fan, in `[-1, 1]`. */
function childFan(index: number, count: number): number {
  return (2 * index + 1) / count - 1;
}

/**
 * Radii from the tips inward: a node's cross-section is the sum of its children's.
 *
 * Da Vinci's rule, applied exactly, plus a per-segment taper so a chain of single-child nodes
 * is not a constant-radius tube.
 */
function daVinciRadii(nodes: readonly DraftNode[]): number[] {
  const children: number[][] = nodes.map(() => []);
  nodes.forEach((node, index) => {
    if (node.parent >= 0) children[node.parent]?.push(index);
  });
  const radii = new Array<number>(nodes.length).fill(0);
  for (let index = nodes.length - 1; index >= 0; index -= 1) {
    const kids = children[index] ?? [];
    if (kids.length === 0) {
      radii[index] = TWIG_RADIUS_M;
      continue;
    }
    let area = 0;
    for (const kid of kids) area += (radii[kid] ?? 0) ** 2;
    radii[index] = Math.sqrt(area) * (nodes[index]?.band === "trunk" ? TRUNK_TAPER : LIMB_TAPER);
  }
  return radii;
}

export interface SyntheticTreeOptions {
  /** Height of the tree, metres; the trunk itself reaches 70 % of it. Default 6. */
  readonly heightM?: number;
  /** Trunk joints including the root. Default 10. */
  readonly trunkSegments?: number;
  /** Trunk joints that carry branches, counted from the top. Default 4. */
  readonly whorls?: number;
  /** Primary branches per whorl. Default 3. */
  readonly branchesPerWhorl?: number;
  /** Secondary limbs per primary. Default 3. */
  readonly secondariesPerBranch?: number;
  /** Twigs (leaf-band tips) per secondary. Default 3. */
  readonly twigsPerSecondary?: number;
  readonly canonicalChecksum?: string;
}

/**
 * A deterministic synthetic tree rig: 214 nodes by default, root at the origin, trunk along +Z.
 *
 * This is the fixture the runtime is developed against — the one input where the correct
 * grouping is known exactly. No randomness: the same options always give the same rig.
 *
 * Four generations — trunk, primary, secondary, twig — rather than the three of the 33-node
 * version, because nine leaf clusters cannot rustle. 108 of them can, and each is an
 * independent oscillator with its own natural frequency, bending plane and gust delay.
 */
export function syntheticTreeRig(options: SyntheticTreeOptions = {}): MotionRig {
  const heightM = options.heightM ?? 6;
  const trunkSegments = Math.max(2, Math.floor(options.trunkSegments ?? 10));
  const whorls = Math.max(1, Math.floor(options.whorls ?? 4));
  const branchesPerWhorl = Math.max(1, Math.floor(options.branchesPerWhorl ?? 3));
  const secondariesPerBranch = Math.max(1, Math.floor(options.secondariesPerBranch ?? 3));
  const twigsPerSecondary = Math.max(1, Math.floor(options.twigsPerSecondary ?? 3));
  const trunkTopM = TRUNK_TOP_FRACTION * heightM;

  const draft: DraftNode[] = [];
  const trunkIndices: number[] = [];
  for (let i = 0; i < trunkSegments; i += 1) {
    const f = i / (trunkSegments - 1);
    trunkIndices.push(draft.length);
    draft.push({
      id: `trunk-${i}`,
      parent: i === 0 ? -1 : (trunkIndices[i - 1] ?? -1),
      position: [0, 0, f * trunkTopM],
      band: "trunk",
    });
  }

  let ordinal = 0;
  for (let w = 0; w < whorls; w += 1) {
    const trunkIndex = trunkIndices[trunkSegments - 1 - w];
    const anchor = trunkIndex === undefined ? undefined : draft[trunkIndex];
    if (trunkIndex === undefined || anchor === undefined) continue;
    // `w` counts down from the apex, so `f` is 0 at the top of the crown and 1 at its skirt:
    // long, shallow branches at the bottom and short, steep ones at the top.
    const f = whorls === 1 ? 1 : w / (whorls - 1);
    const scale = heightM / HEIGHT_REFERENCE_M;
    const whorlLen =
      (PRIMARY_LENGTH_M[1] + (PRIMARY_LENGTH_M[0] - PRIMARY_LENGTH_M[1]) * f) * scale;
    const whorlEl =
      PRIMARY_ELEVATION_RAD[1] + (PRIMARY_ELEVATION_RAD[0] - PRIMARY_ELEVATION_RAD[1]) * f;
    for (let b = 0; b < branchesPerWhorl; b += 1) {
      const azimuth = GOLDEN_ANGLE * ordinal;
      const primaryLen = whorlLen * (1 + LIMB_LENGTH_JITTER * wobble(ordinal, LENGTH_PHASE));
      const primaryEl = whorlEl + LIMB_ELEVATION_JITTER * wobble(ordinal, ELEVATION_PHASE);
      const first = draft.length;
      limbChain(anchor.position, azimuth, primaryEl, primaryLen, 2, LIMB_CURL_RAD).forEach(
        (position, j) => {
          draft.push({
            id: `branch-${ordinal}-${j}`,
            parent: j === 0 ? trunkIndex : first + j - 1,
            position,
            band: "branch",
          });
        },
      );
      const primaryTip = first + 1;
      const secondaryLen = primaryLen * SECONDARY_LENGTH_FRACTION;
      const secondaryEl = primaryEl + LIMB_CURL_RAD + SECONDARY_ELEVATION_GAIN_RAD;
      for (let s = 0; s < secondariesPerBranch; s += 1) {
        const fan = childFan(s, secondariesPerBranch);
        const wobbleIndex = ordinal * 7 + s;
        const secondaryLenS =
          secondaryLen * (1 + LIMB_LENGTH_JITTER * wobble(wobbleIndex, LENGTH_PHASE));
        const secondaryAz = azimuth + CHILD_SPREAD_RAD * fan;
        const secondaryElS =
          secondaryEl +
          CHILD_ELEVATION_SPREAD_RAD * fan +
          LIMB_ELEVATION_JITTER * wobble(wobbleIndex, ELEVATION_PHASE);
        const start = draft[primaryTip]?.position ?? anchor.position;
        const base = draft.length;
        limbChain(start, secondaryAz, secondaryElS, secondaryLenS, 2, LIMB_CURL_RAD).forEach(
          (position, j) => {
            draft.push({
              id: `twig-${ordinal}-${s}-${j}`,
              parent: j === 0 ? primaryTip : base + j - 1,
              position,
              band: "branch",
            });
          },
        );
        const secondaryTip = base + 1;
        const twigLen = secondaryLenS * TWIG_LENGTH_FRACTION;
        const twigEl = secondaryElS + LIMB_CURL_RAD + TWIG_ELEVATION_GAIN_RAD;
        for (let k = 0; k < twigsPerSecondary; k += 1) {
          const twigFan = childFan(k, twigsPerSecondary);
          const twigIndex = ordinal * 13 + s * 3 + k;
          const tip = limbChain(
            draft[secondaryTip]?.position ?? start,
            secondaryAz + CHILD_SPREAD_RAD * twigFan,
            twigEl +
              CHILD_ELEVATION_SPREAD_RAD * twigFan +
              LIMB_ELEVATION_JITTER * wobble(twigIndex, ELEVATION_PHASE),
            twigLen * (1 + LIMB_LENGTH_JITTER * wobble(twigIndex, LENGTH_PHASE)),
            1,
            LIMB_CURL_RAD,
          )[0];
          draft.push({
            id: `leaf-${ordinal}-${s}-${k}`,
            parent: secondaryTip,
            position: tip ?? start,
            band: "leaf",
          });
        }
      }
      ordinal += 1;
    }
  }

  const radii = daVinciRadii(draft);
  const rootRadius = radii[0] ?? TWIG_RADIUS_M;
  // Linear in thickness between the twigs and the trunk base, so the thinnest node in the rig
  // is exactly STIFFNESS_MIN whatever the tree's absolute scale.
  const span = Math.max(rootRadius - TWIG_RADIUS_M, 1e-9);
  const nodes: SkeletonNode[] = draft.map((node, index) => {
    const radius = radii[index] ?? TWIG_RADIUS_M;
    return {
      id: node.id,
      parent: node.parent,
      position: node.position,
      radius,
      stiffness:
        STIFFNESS_MIN + (STIFFNESS_MAX - STIFFNESS_MIN) * ((radius - TWIG_RADIUS_M) / span),
      band: node.band,
    };
  });

  return assertValidRig({
    nodes,
    canonicalChecksum: options.canonicalChecksum ?? "fnv1a32:0:00000000",
    units: "meters",
    sourceNote: `synthetic tree, ${heightM} m, ${nodes.length} nodes (@twin/world syntheticTreeRig)`,
  });
}
