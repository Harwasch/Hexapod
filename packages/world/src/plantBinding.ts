/**
 * The plant binding: which gaussian of every tile belongs to which plant of a forest rig, and
 * which to no plant at all.
 *
 * A single tree's rig binds a gaussian to the joints nearest it, from its position alone
 * (`skinSplatsToNodes`), and that is right when everything in the tileset is the tree. A
 * capture of a park is not: a wall stands a metre from a crown, a lawn runs under it, and the
 * joints nearest a brick are the crown's. Which splat is part of which plant is what the scene
 * step measured (`tools/captures/scene_plants.py`: ground, greenness, instances), so it travels
 * with the tiles as a sidecar — `plants.json`, the rig's `binding` — and the runtime only
 * applies it:
 *
 * - **static** (label 0): the gaussian is bound, with weight 1, to the rig's static anchor,
 *   which the deformer pins (`staticAnchors`), so it is drawn from its canonical bytes on both
 *   motion paths at every wind — bit-exact, not merely still;
 * - **plant k** (label k ≥ 1): the gaussian is skinned exactly as `skinSplatsToNodes` would skin
 *   it against a rig made of plant k's joints alone — its four nearest *of that plant*, modified
 *   Shepard weights — so nothing of one plant ever follows another, or a building a plant.
 *
 * ### The file
 *
 * ```json
 * { "format": "hexapod.plants", "version": 1, "encoding": "rle",
 *   "tiles": { "fnv1a32:5804:a28c07b7": [0, 3120, 4, 17, 0, 88, ...], ... } }
 * ```
 *
 * One entry per tile, **keyed by the tile's own checksum** (`checksumPositions` over its
 * canonical positions, what `rig.tileChecksums` lists): run-length pairs `[label, count]` in the
 * tile's own gaussian order, which is Morton order within a cell, so a plant is a few runs. The
 * checksum carries the tile's gaussian count, and the runs must add up to it. A tile whose
 * digest has no entry is not bound — the deformer refuses — and every digest the rig lists must
 * have one, which is how a binding for another packing of the capture is caught at load.
 */

import { type MotionRig } from "./rig";
import { SKIN_INFLUENCES, SKIN_WEIGHT_TOTAL, skinSplatsToNodes, type SplatSkin } from "./skin";
import { staticAnchorNode } from "./rig";

export const PLANT_BINDING_FORMAT = "hexapod.plants";
export const PLANT_BINDING_VERSION = 1;

/** A parsed binding. Runs are decoded per tile, when the tile is bound. */
export interface PlantBinding {
  /** Plants the labels index, 1-based: the rig's `plants.length`. */
  readonly plantCount: number;
  /** Per tile checksum, run-length pairs `[label, count, ...]`. */
  readonly tiles: ReadonlyMap<string, Int32Array>;
}

const CHECKSUM = /^fnv1a32:(\d+):[0-9a-f]{8}$/;

/** Every problem with a parsed binding document against its rig. Empty means valid. */
export function validatePlantBinding(raw: unknown, rig: MotionRig): string[] {
  const issues: string[] = [];
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    return ["the top level must be an object"];
  }
  const doc = raw as Record<string, unknown>;
  if (doc.format !== PLANT_BINDING_FORMAT) issues.push(`format must be ${PLANT_BINDING_FORMAT}`);
  if (doc.version !== PLANT_BINDING_VERSION)
    issues.push(`version ${JSON.stringify(doc.version)} is not ${PLANT_BINDING_VERSION}`);
  if (doc.encoding !== "rle") issues.push('encoding must be "rle"');
  const plants = rig.plants;
  if (plants === undefined) issues.push("the rig lists no plants");
  const count = plants?.length ?? 0;
  const tiles = doc.tiles;
  if (typeof tiles !== "object" || tiles === null || Array.isArray(tiles)) {
    issues.push("tiles must be an object keyed by tile checksum");
    return issues;
  }
  for (const [key, value] of Object.entries(tiles as Record<string, unknown>)) {
    const match = CHECKSUM.exec(key);
    if (match === null) {
      issues.push(`${key}: not a checksumPositions digest`);
      continue;
    }
    if (!Array.isArray(value) || value.length % 2 !== 0) {
      issues.push(`${key}: runs must be [label, count] pairs`);
      continue;
    }
    let total = 0;
    for (let i = 0; i < value.length; i += 2) {
      const label: unknown = value[i];
      const run: unknown = value[i + 1];
      if (!Number.isInteger(label) || (label as number) < 0 || (label as number) > count) {
        issues.push(`${key}: label ${String(label)} is not 0 or a plant`);
        break;
      }
      if (!Number.isInteger(run) || (run as number) <= 0) {
        issues.push(`${key}: run length ${String(run)} must be a positive integer`);
        break;
      }
      total += run as number;
    }
    if (total !== Number(match[1]))
      issues.push(`${key}: runs cover ${total} gaussians, the tile holds ${match[1] ?? "?"}`);
  }
  const listed = rig.tileChecksums ?? [rig.canonicalChecksum];
  const keys = new Set(Object.keys(tiles));
  const missing = listed.filter((checksum) => !keys.has(checksum));
  if (missing.length > 0)
    issues.push(
      `${missing.length} of the rig's tiles have no binding (first: ${missing[0] ?? ""})`,
    );
  return issues;
}

/** Parses `plants.json` for `rig`. Throws with every problem listed. */
export function parsePlantBinding(text: string, rig: MotionRig): PlantBinding {
  const raw = JSON.parse(text) as unknown;
  const issues = validatePlantBinding(raw, rig);
  if (issues.length > 0) throw new Error(`invalid plant binding:\n  ${issues.join("\n  ")}`);
  const tiles = new Map<string, Int32Array>();
  for (const [key, value] of Object.entries((raw as { tiles: Record<string, number[]> }).tiles)) {
    tiles.set(key, Int32Array.from(value));
  }
  return { plantCount: rig.plants?.length ?? 0, tiles };
}

/** The label of every gaussian of the tile whose checksum is `checksum`, or `undefined`. */
export function plantLabels(binding: PlantBinding, checksum: string): Uint16Array | undefined {
  const runs = binding.tiles.get(checksum);
  if (runs === undefined) return undefined;
  let total = 0;
  for (let i = 1; i < runs.length; i += 2) total += runs[i] ?? 0;
  const out = new Uint16Array(total);
  let at = 0;
  for (let i = 0; i < runs.length; i += 2) {
    const run = runs[i + 1] ?? 0;
    out.fill(runs[i] ?? 0, at, at + run);
    at += run;
  }
  return out;
}

/** Per rig, each plant's joints as a rig of their own. Weak: a rig let go takes them with it. */
const PLANT_RIGS = new WeakMap<MotionRig, MotionRig[]>();

/** Plant `k`'s joints, re-indexed from 0, as a single-plant rig. */
export function plantRig(rig: MotionRig, k: number): MotionRig {
  let rigs = PLANT_RIGS.get(rig);
  if (rigs === undefined) {
    rigs = (rig.plants ?? []).map((plant) => ({
      nodes: rig.nodes.slice(plant.nodeStart, plant.nodeEnd).map((node) => ({
        ...node,
        parent: node.parent < 0 ? -1 : node.parent - plant.nodeStart,
      })),
      canonicalChecksum: rig.canonicalChecksum,
      units: "meters" as const,
      sourceNote: `${rig.sourceNote} / ${plant.id}`,
    }));
    PLANT_RIGS.set(rig, rigs);
  }
  const out = rigs[k];
  if (out === undefined) throw new Error(`plantRig: the rig has no plant ${k}`);
  return out;
}

/**
 * Skins every gaussian to its own plant's joints, or binds it to the static anchor.
 *
 * `labels[i]` is gaussian `i`'s label (`plantLabels`): 0 static, `k` plant `k - 1` of
 * `rig.plants`. A plant's gaussians get exactly what `skinSplatsToNodes` gives them against
 * that plant's joints alone, re-indexed into the forest rig; a static one — or any with a
 * non-finite position — gets the anchor with weight 1 in every slot.
 */
export function skinSplatsToPlants(
  positions: Float32Array,
  rig: MotionRig,
  labels: Uint16Array,
): SplatSkin {
  const count = Math.floor(positions.length / 3);
  if (labels.length !== count)
    throw new Error(`skinSplatsToPlants: ${labels.length} labels for ${count} gaussians`);
  const anchor = staticAnchorNode(rig);
  if (anchor < 0) throw new Error("skinSplatsToPlants: the rig has no static anchor");
  const nodes = new Uint16Array(count * SKIN_INFLUENCES).fill(anchor);
  const weights = new Uint16Array(count * SKIN_INFLUENCES);
  // Every gaussian starts static; each plant's are then gathered and skinned in one pass.
  const members = new Map<number, number[]>();
  for (let i = 0; i < count; i += 1) {
    weights[i * SKIN_INFLUENCES] = SKIN_WEIGHT_TOTAL;
    const label = labels[i] ?? 0;
    if (label === 0) continue;
    const x = positions[i * 3] ?? Number.NaN;
    const y = positions[i * 3 + 1] ?? Number.NaN;
    const z = positions[i * 3 + 2] ?? Number.NaN;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    let list = members.get(label);
    if (list === undefined) {
      list = [];
      members.set(label, list);
    }
    list.push(i);
  }
  const plants = rig.plants ?? [];
  for (const [label, list] of members) {
    const plant = plants[label - 1];
    if (plant === undefined) continue;
    const gathered = new Float32Array(list.length * 3);
    list.forEach((i, j) => {
      gathered[j * 3] = positions[i * 3] ?? 0;
      gathered[j * 3 + 1] = positions[i * 3 + 1] ?? 0;
      gathered[j * 3 + 2] = positions[i * 3 + 2] ?? 0;
    });
    const skin = skinSplatsToNodes(gathered, plantRig(rig, label - 1));
    list.forEach((i, j) => {
      for (let k = 0; k < SKIN_INFLUENCES; k += 1) {
        nodes[i * SKIN_INFLUENCES + k] =
          (skin.nodes[j * SKIN_INFLUENCES + k] ?? 0) + plant.nodeStart;
        weights[i * SKIN_INFLUENCES + k] = skin.weights[j * SKIN_INFLUENCES + k] ?? 0;
      }
    });
  }
  return { nodes, weights };
}
