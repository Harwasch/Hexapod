import type { Articulation, ArticulationNode, ArticulationHeader } from "./livingArticulation";
interface Instance {
  id: number;
  parent?: number | null;
  category?: string;
  properties?: Record<string, number>;
  behaviour?: string;
  bounds?: { min: number[]; max: number[] };
}
type BoundInstance = Instance & { bounds: { min: number[]; max: number[] } };
/** Published segmentation -> clump rig and checksum-bound streams. Load-time only.
 * Native tile run order is preserved; there is no colour guessing or per-frame CPU splat work.
 */
export function tierAFromInstances(value: unknown): Articulation {
  const doc = value as {
    format?: string;
    instances?: Instance[];
    tiles?: Record<string, number[]>;
    tilesEncoding?: string;
  };
  if (doc?.format !== "hexapod.instances" || !Array.isArray(doc.instances) || !doc.tiles)
    throw new Error("Invalid vegetation segmentation");
  const classes = new Set([
    "trees",
    "shrubs",
    "grass",
    "crops",
    "flowers",
    "ferns",
    "moss",
    "vegetation",
  ]);
  const all = new Map(doc.instances.map((i) => [i.id, i]));
  const vegetation = (i: Instance, seen = new Set<number>()): boolean => {
    if (seen.has(i.id)) return false;
    seen.add(i.id);
    // A vegetation score never overrides a known ground/rigid/static class.
    if (
      i.behaviour === "static" ||
      (i.properties?.rigid ?? 0) >= 0.5 ||
      (i.properties?.static ?? 0) >= 0.8
    )
      return false;
    const parent = i.parent == null ? null : all.get(i.parent);
    // A finer shrub label inside a ground/table segment is not independent evidence.
    // Fail closed on the complete ancestry, including contradictory coarse labels.
    if (parent && !vegetation(parent, seen)) return false;
    if (i.category) return classes.has(i.category);
    if ((i.properties?.vegetation ?? 0) >= 0.5) return true;
    return parent !== null && parent !== undefined;
  };
  const candidates = doc.instances.filter(
    (i): i is BoundInstance =>
      vegetation(i) &&
      i.bounds !== undefined &&
      [...i.bounds.min, ...i.bounds.max].length === 6 &&
      [...i.bounds.min, ...i.bounds.max].every(Number.isFinite),
  );
  let cell = 2,
    groups: Map<string, BoundInstance[]>;
  do {
    groups = new Map();
    for (const i of candidates) {
      const b = i.bounds;
      const key = `${Math.floor(((b.min[0] ?? 0) + (b.max[0] ?? 0)) / 2 / cell)},${Math.floor(((b.min[1] ?? 0) + (b.max[1] ?? 0)) / 2 / cell)}`;
      const list = groups.get(key);
      if (list) list.push(i);
      else groups.set(key, [i]);
    }
    cell *= 2;
  } while (groups.size > 500);
  const nodes: ArticulationNode[] = [],
    owners = new Map<number, number>();
  let ground = Infinity;
  for (const list of groups.values()) {
    const low = [0, 1, 2].map((c) => Math.min(...list.map((i) => i.bounds.min[c] ?? 0))),
      high = [0, 1, 2].map((c) => Math.max(...list.map((i) => i.bounds.max[c] ?? 0)));
    const x = ((low[0] ?? 0) + (high[0] ?? 0)) / 2,
      y = ((low[1] ?? 0) + (high[1] ?? 0)) / 2,
      z = low[2] ?? 0;
    const root = nodes.length;
    nodes.push(
      { parent: -1, band: "trunk", position: [x, y, z] },
      {
        parent: root,
        position: [x, y, high[2] ?? z],
        groundM: z,
        canopyM: Math.max(0.15, (high[2] ?? z) - z),
      },
    );
    ground = Math.min(ground, z);
    for (const i of list) owners.set(i.id, root + 2);
  }
  if (nodes.length === 0) nodes.push({ parent: -1, band: "trunk", position: [0, 0, 0] });
  const tiles: ArticulationHeader["tiles"] = {};
  let count = 0;
  for (const [checksum, runs] of Object.entries(doc.tiles)) {
    if (
      !Array.isArray(runs) ||
      runs.length % 2 !== 0 ||
      !runs.every((n) => Number.isSafeInteger(n) && n >= 0)
    )
      throw new Error("Invalid instance runs");
    const size = runs.reduce((n, v, i) => n + (i % 2 ? v : 0), 0);
    if (size < 1 || size > 10000000 || count + size > 50000000)
      throw new Error("Invalid vegetation tile size");
    tiles[checksum] = { offset: count * 16, count: size };
    count += size;
  }
  const bindings = new Uint32Array(count * 4);
  let row = 0;
  for (const runs of Object.values(doc.tiles))
    for (let i = 0; i < runs.length; i += 2) {
      const owner = owners.get(runs[i] ?? 0) ?? 0;
      for (let n = 0; n < (runs[i + 1] ?? 0); n++, row++)
        if (owner > 0) bindings.set([owner, owner - 1, 65535, 0], row * 4);
    }
  const header: ArticulationHeader = {
    format: "hexapod.living-articulation",
    version: 1,
    motionEvidence: "simulated",
    nodes,
    frequenciesHz: [0],
    referenceBearingDeg: 90,
    motionBinary: "",
    leafBinary: "",
    bindingBinary: "",
    leafCount: 0,
    tiles,
    groundM: Number.isFinite(ground) ? ground : 0,
    detail: {
      vegetation: candidates.length > 0,
      fittedRig: false,
      separableCrownShare: 0,
      attachedCrownShare: 0,
      smallSplatShare: 0,
      leafBindings: false,
    },
  };
  return {
    header,
    coefficients: new Float32Array(nodes.length * 6),
    leaves: new Float32Array(0),
    bindings,
  };
}
