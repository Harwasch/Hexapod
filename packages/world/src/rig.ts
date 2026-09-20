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
  /** Approximate cross-sectional radius at this joint, metres. Descriptive; not used by `deform`. */
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

export interface SyntheticTreeOptions {
  /** Height of the trunk, metres. Default 6. */
  readonly heightM?: number;
  /** Trunk joints including the root. Default 6. */
  readonly trunkSegments?: number;
  /** Trunk joints that carry branches, counted from the top. Default 3. */
  readonly whorls?: number;
  /** Branches per whorl. Default 3. */
  readonly branchesPerWhorl?: number;
  readonly canonicalChecksum?: string;
}

/**
 * A deterministic synthetic tree rig: 33 nodes by default, root at the origin, trunk along +Z.
 *
 * This is the fixture the runtime is developed against — the one input where the correct
 * grouping is known exactly. No randomness: the same options always give the same rig.
 */
export function syntheticTreeRig(options: SyntheticTreeOptions = {}): MotionRig {
  const heightM = options.heightM ?? 6;
  const trunkSegments = Math.max(2, Math.floor(options.trunkSegments ?? 6));
  const whorls = Math.max(1, Math.floor(options.whorls ?? 3));
  const branchesPerWhorl = Math.max(1, Math.floor(options.branchesPerWhorl ?? 3));
  const nodes: SkeletonNode[] = [];
  const trunkIndices: number[] = [];

  for (let i = 0; i < trunkSegments; i += 1) {
    const f = i / (trunkSegments - 1);
    trunkIndices.push(nodes.length);
    nodes.push({
      id: `trunk-${i}`,
      parent: i === 0 ? -1 : (trunkIndices[i - 1] ?? -1),
      position: [0, 0, f * heightM],
      radius: 0.35 - 0.23 * f,
      stiffness: 8 - 2 * f,
      band: "trunk",
    });
  }

  let branchOrdinal = 0;
  for (let w = 0; w < whorls; w += 1) {
    const trunkIndex = trunkIndices[trunkSegments - 1 - w];
    const anchor = trunkIndex === undefined ? undefined : nodes[trunkIndex];
    if (trunkIndex === undefined || anchor === undefined) continue;
    for (let b = 0; b < branchesPerWhorl; b += 1) {
      const azimuth = GOLDEN_ANGLE * branchOrdinal;
      const east = Math.cos(azimuth);
      const north = Math.sin(azimuth);
      const base = anchor.position;
      const primaryIndex = nodes.length;
      nodes.push({
        id: `branch-${branchOrdinal}-0`,
        parent: trunkIndex,
        position: [base[0] + east * 1.1, base[1] + north * 1.1, base[2] + 0.35],
        radius: 0.12,
        stiffness: 3.2,
        band: "branch",
      });
      const secondaryIndex = nodes.length;
      const primary = nodes[primaryIndex]?.position ?? base;
      nodes.push({
        id: `branch-${branchOrdinal}-1`,
        parent: primaryIndex,
        position: [primary[0] + east * 0.9, primary[1] + north * 0.9, primary[2] + 0.3],
        radius: 0.07,
        stiffness: 2,
        band: "branch",
      });
      const secondary = nodes[secondaryIndex]?.position ?? primary;
      nodes.push({
        id: `leaf-${branchOrdinal}`,
        parent: secondaryIndex,
        position: [secondary[0] + east * 0.6, secondary[1] + north * 0.6, secondary[2] + 0.25],
        radius: 0.04,
        stiffness: 1,
        band: "leaf",
      });
      branchOrdinal += 1;
    }
  }

  return assertValidRig({
    nodes,
    canonicalChecksum: options.canonicalChecksum ?? "fnv1a32:0:00000000",
    units: "meters",
    sourceNote: `synthetic tree, ${heightM} m, ${nodes.length} nodes (@twin/world syntheticTreeRig)`,
  });
}
