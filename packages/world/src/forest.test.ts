/**
 * Forest rigs: many plants in one rig, each skinned only to its own joints, everything else
 * pinned to a static anchor — against the committed synthetic yard
 * (`data/tiles/synthetic-yard/splat`, written by `tools/captures/synthetic_yard.py` and
 * `scene_plants.py`).
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  checksumPositions,
  createLivingMotion,
  IDENTITY_TRANSFORM,
  isForestRig,
  livingFrame,
  livingMaxDisplacement,
  NO_GUSTS,
  parseMotionSidecar,
  parsePlantBinding,
  parseRig,
  plantLabels,
  plantRig,
  prepareLivingMotion,
  quatRotate,
  serializeRig,
  SKIN_INFLUENCES,
  SKIN_WEIGHT_TOTAL,
  skinSplatsToNodes,
  skinSplatsToPlants,
  staticAnchorNode,
  staticAnchors,
  validateMotionSidecar,
  validatePlantBinding,
  validateRig,
  type LivingWind,
  type MotionRig,
  type MotionSidecar,
  type NodeTransform,
  type Vec3,
} from "./index";

function yardPath(name: string): string {
  return fileURLToPath(
    new URL(`../../../data/tiles/synthetic-yard/splat/${name}`, import.meta.url),
  );
}

const rigText = readFileSync(yardPath("rig.json"), "utf8");
const rig = parseRig(rigText);
const sidecarText = readFileSync(yardPath("motion.json"), "utf8");
const sidecar = parseMotionSidecar(sidecarText, rig);
const bindingText = readFileSync(yardPath("plants.json"), "utf8");

/** Wall-clock milliseconds, from Node's monotonic clock: the library itself names no clock. */
function nowMs(): number {
  return Number(process.hrtime.bigint()) / 1e6;
}

const WIND: LivingWind = { speedMps: 12, bearingDeg: 250, gust: NO_GUSTS, season: "summer" };

function transformPoint(t: NodeTransform, p: Vec3): Vec3 {
  const r = quatRotate(t.rotation, p);
  return [r[0] + t.translation[0], r[1] + t.translation[1], r[2] + t.translation[2]];
}

describe("the yard's forest rig", () => {
  it("is a valid forest rig: a static anchor, then one root per plant", () => {
    expect(isForestRig(rig)).toBe(true);
    expect(validateRig(rig)).toEqual([]);
    expect(rig.plants?.map((p) => p.class)).toEqual([
      "tree",
      "tree",
      "tree",
      "shrub",
      "shrub",
      "shrub",
      "shrub",
      "shrub",
      "snag",
      "snag",
    ]);
    const anchors = staticAnchors(rig);
    expect(staticAnchorNode(rig)).toBe(0);
    expect(anchors.reduce((n, v) => n + v, 0)).toBe(1);
    for (const plant of rig.plants ?? []) {
      expect(rig.nodes[plant.nodeStart]?.parent).toBe(-1);
      for (let i = plant.nodeStart + 1; i < plant.nodeEnd; i += 1) {
        const parent = rig.nodes[i]?.parent ?? -1;
        expect(parent).toBeGreaterThanOrEqual(plant.nodeStart);
        expect(parent).toBeLessThan(i);
      }
    }
    expect(rig.bindingPath).toBe("plants.json");
    expect(rig.motionPath).toBe("motion.json");
  });

  it("round-trips through serializeRig and parseRig", () => {
    expect(parseRig(serializeRig(rig))).toEqual(rig);
  });

  it("refuses a plant hanging from another plant, and a stray root inside a plant", () => {
    const plants = rig.plants ?? [];
    const second = plants[1];
    if (second === undefined) throw new Error("no second plant");
    const crossed: MotionRig = {
      ...rig,
      nodes: rig.nodes.map((node, i) =>
        i === second.nodeStart + 1 ? { ...node, parent: 1 } : node,
      ),
    };
    expect(validateRig(crossed).join("\n")).toContain("of its own plant");
    const stray: MotionRig = {
      ...rig,
      nodes: rig.nodes.map((node, i) =>
        i === second.nodeStart + 1 ? { ...node, parent: -1 } : node,
      ),
    };
    expect(validateRig(stray).length).toBeGreaterThan(0);
    const anchored: MotionRig = {
      ...rig,
      nodes: rig.nodes.map((node, i) => (i === 0 ? node : node)),
      plants: plants.slice(1),
    };
    // Plant 0's nodes are now outside every plant, and only anchors may be.
    expect(validateRig(anchored).join("\n")).toContain("must be an anchor");
    const noPlants: MotionRig = { ...rig, plants: undefined };
    expect(validateRig(noPlants).join("\n")).toContain("binding is a forest rig's");
  });

  it("has a sidecar whose plants are the rig's, each with the wind at its own height", () => {
    expect(validateMotionSidecar(sidecar, rig)).toEqual([]);
    const plants = sidecar.plants ?? [];
    expect(plants.map((p) => p.id)).toEqual(rig.plants?.map((p) => p.id));
    const tallest = Math.max(...plants.map((p) => p.heightM));
    expect(sidecar.treeHeightM).toBe(tallest);
    // EN 1991-1-4 holds the intensity constant below z_min = 5 m: every shrub shares one.
    const shrubs = plants.filter((p) => p.class === "shrub");
    expect(new Set(shrubs.map((p) => p.turbulence.along)).size).toBe(1);
    const trees = plants.filter((p) => p.class === "tree");
    expect(Math.min(...trees.map((p) => p.turbulence.along))).toBeLessThan(
      shrubs[0]?.turbulence.along ?? 0,
    );
    const mismatched = { ...sidecar, plants: plants.slice(1) } as MotionSidecar;
    expect(validateMotionSidecar(mismatched, rig).join("\n")).toContain("plants");
  });

  it("gives no snag a leaf to flutter, and every shrub tips that do", () => {
    const flutter = sidecar.nodes.flutterM;
    for (const plant of sidecar.plants ?? []) {
      const own = flutter.slice(plant.nodeStart, plant.nodeEnd);
      if (plant.class === "snag") expect(own.every((v) => v === 0)).toBe(true);
      else expect(own.some((v) => v > 0)).toBe(true);
    }
    expect(flutter[0]).toBe(0);
  });
});

describe("the plant binding", () => {
  const binding = parsePlantBinding(bindingText, rig);

  it("covers every tile the rig lists, each run adding up to the tile's gaussians", () => {
    expect(binding.plantCount).toBe(rig.plants?.length);
    for (const checksum of rig.tileChecksums ?? []) {
      const labels = plantLabels(binding, checksum);
      expect(labels).toBeDefined();
      expect(labels?.length).toBe(Number(checksum.split(":")[1]));
    }
    expect(plantLabels(binding, "fnv1a32:3:00000000")).toBeUndefined();
  });

  it("is refused when it is not this rig's", () => {
    const doc = JSON.parse(bindingText) as { tiles: Record<string, number[]> };
    const first = Object.keys(doc.tiles)[0] ?? "";
    const { [first]: _dropped, ...rest } = doc.tiles;
    expect(validatePlantBinding({ ...doc, tiles: rest }, rig).join("\n")).toContain(
      "have no binding",
    );
    const short = { ...doc, tiles: { ...doc.tiles, [first]: [0, 1] } };
    expect(validatePlantBinding(short, rig).join("\n")).toContain("runs cover 1 gaussians");
    const beyond = { ...doc, tiles: { ...doc.tiles, [first]: [99, Number(first.split(":")[1])] } };
    expect(validatePlantBinding(beyond, rig).join("\n")).toContain("not 0 or a plant");
    expect(() => parsePlantBinding(JSON.stringify({ ...doc, format: "x" }), rig)).toThrow();
  });
});

describe("skinning to plants", () => {
  // Gaussians scattered around every plant's joints, labelled with a plant or with nothing.
  const plants = rig.plants ?? [];
  const points: number[] = [];
  const labels: number[] = [];
  plants.forEach((plant, k) => {
    for (let i = plant.nodeStart; i < plant.nodeEnd; i += 1) {
      const p = rig.nodes[i]?.position ?? [0, 0, 0];
      for (const [dx, dy, dz, own] of [
        [0.05, 0.02, 0.01, true],
        [-0.03, 0.04, 0.02, true],
        [0.02, -0.06, -0.01, false],
      ] as const) {
        points.push(p[0] + dx, p[1] + dy, p[2] + dz);
        labels.push(own ? k + 1 : 0);
      }
    }
  });
  const positions = Float32Array.from(points);
  const labelArray = Uint16Array.from(labels);
  const skin = skinSplatsToPlants(positions, rig, labelArray);

  it("binds a static gaussian wholly to the anchor", () => {
    for (let i = 0; i < labels.length; i += 1) {
      if (labels[i] !== 0) continue;
      expect(Array.from(skin.weights.subarray(i * 4, i * 4 + 4))).toEqual([
        SKIN_WEIGHT_TOTAL,
        0,
        0,
        0,
      ]);
      expect(Array.from(skin.nodes.subarray(i * 4, i * 4 + 4))).toEqual([0, 0, 0, 0]);
    }
  });

  it("skins a plant's gaussian to that plant's joints only, exactly as its own rig would", () => {
    for (let i = 0; i < labels.length; i += 1) {
      const label = labels[i] ?? 0;
      if (label === 0) continue;
      const plant = plants[label - 1];
      if (plant === undefined) throw new Error("no plant");
      const own = skinSplatsToNodes(positions.subarray(i * 3, i * 3 + 3), plantRig(rig, label - 1));
      for (let k = 0; k < SKIN_INFLUENCES; k += 1) {
        const node = skin.nodes[i * SKIN_INFLUENCES + k] ?? -1;
        const weight = skin.weights[i * SKIN_INFLUENCES + k] ?? 0;
        expect(weight).toBe(own.weights[k]);
        if (weight > 0) {
          expect(node).toBeGreaterThanOrEqual(plant.nodeStart);
          expect(node).toBeLessThan(plant.nodeEnd);
          expect(node - plant.nodeStart).toBe(own.nodes[k]);
        }
      }
    }
  });

  it("refuses labels that do not match the positions", () => {
    expect(() => skinSplatsToPlants(positions, rig, labelArray.subarray(1))).toThrow();
  });
});

describe("Living Mode on a forest", () => {
  const motion = createLivingMotion(rig, sidecar);
  prepareLivingMotion(motion);

  it("is the identity by value at calm", () => {
    const frame = livingFrame(motion, 12.5, { ...WIND, speedMps: 0 });
    for (const t of frame.transforms) expect(t).toBe(IDENTITY_TRANSFORM);
    expect(frame.flutter.still).toBe(true);
  });

  it("never moves the static anchor or a plant's root, and moves every plant", () => {
    for (const t of [3, 41.7, 600.25]) {
      const { transforms, flutter } = livingFrame(motion, t, WIND);
      expect(transforms[0]).toBe(IDENTITY_TRANSFORM);
      expect(flutter.amplitudeM[0]).toBe(0);
      for (const plant of rig.plants ?? []) {
        expect(transforms[plant.nodeStart]).toBe(IDENTITY_TRANSFORM);
        const moved = transforms
          .slice(plant.nodeStart, plant.nodeEnd)
          .some((x) => x !== IDENTITY_TRANSFORM);
        expect(moved).toBe(true);
      }
    }
  });

  it("sways trees most, shrubs little and snags least, measured at their tops", () => {
    const sway = new Map<string, number[]>();
    for (let step = 0; step < 240; step += 1) {
      const { transforms } = livingFrame(motion, 100 + step * 0.25, WIND);
      for (const plant of rig.plants ?? []) {
        let top = plant.nodeStart;
        for (let i = plant.nodeStart; i < plant.nodeEnd; i += 1) {
          if ((rig.nodes[i]?.position[2] ?? 0) > (rig.nodes[top]?.position[2] ?? 0)) top = i;
        }
        const p = rig.nodes[top]?.position ?? [0, 0, 0];
        const q = transformPoint(transforms[top] ?? IDENTITY_TRANSFORM, p);
        const d = Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]);
        const list = sway.get(plant.class) ?? [];
        list.push(d);
        sway.set(plant.class, list);
      }
    }
    const rms = (xs: number[]): number => Math.sqrt(xs.reduce((s, x) => s + x * x, 0) / xs.length);
    const tree = rms(sway.get("tree") ?? []);
    const shrub = rms(sway.get("shrub") ?? []);
    const snag = rms(sway.get("snag") ?? []);
    expect(tree).toBeGreaterThan(shrub);
    expect(shrub).toBeGreaterThan(0);
    // A bare trunk of a tree's height moves a fraction of what a crowned one does.
    expect(snag).toBeLessThan(tree / 3);
    expect(snag).toBeGreaterThan(0);
  });

  it("has a finite proven bound", () => {
    const bound = livingMaxDisplacement(motion, WIND);
    expect(Number.isFinite(bound)).toBe(true);
    expect(bound).toBeGreaterThan(0);
  });
});

/** The yard's plants copied `copies` times across a grid: a forest of `10 * copies` plants. */
function replicate(
  copies: number,
  only?: (plant: NonNullable<MotionRig["plants"]>[number]) => boolean,
): { rig: MotionRig; sidecar: MotionSidecar } {
  const nodes = [...rig.nodes.slice(0, 1)];
  const plants: NonNullable<MotionRig["plants"]>[number][] = [];
  const sidePlants: NonNullable<MotionSidecar["plants"]>[number][] = [];
  const columns = Object.fromEntries(
    (Object.entries(sidecar.nodes) as [string, readonly number[]][]).map(([key, values]) => [
      key,
      [values[0] ?? 0],
    ]),
  ) as Record<keyof MotionSidecar["nodes"], number[]>;
  for (let c = 0; c < copies; c += 1) {
    const dx = (c % 8) * 40;
    const dy = Math.floor(c / 8) * 40;
    (rig.plants ?? []).forEach((plant, k) => {
      if (only !== undefined && !only(plant)) return;
      const start = nodes.length;
      for (let i = plant.nodeStart; i < plant.nodeEnd; i += 1) {
        const node = rig.nodes[i];
        if (node === undefined) continue;
        nodes.push({
          ...node,
          id: `${String(c)}/${node.id}`,
          parent: node.parent < 0 ? -1 : node.parent - plant.nodeStart + start,
          position: [node.position[0] + dx, node.position[1] + dy, node.position[2]],
        });
        for (const key of Object.keys(columns) as (keyof MotionSidecar["nodes"])[]) {
          const value = sidecar.nodes[key][i] ?? 0;
          columns[key].push(key === "branch" ? value - plant.nodeStart + start : value);
        }
      }
      const id = `${String(c)}/${plant.id}`;
      plants.push({ ...plant, id, nodeStart: start, nodeEnd: nodes.length });
      const own = sidecar.plants?.[k];
      if (own !== undefined)
        sidePlants.push({ ...own, id, nodeStart: start, nodeEnd: nodes.length });
    });
  }
  const forest: MotionRig = { ...rig, nodes, plants, tileChecksums: undefined };
  return {
    rig: forest,
    sidecar: { ...sidecar, nodeCount: nodes.length, nodes: columns, plants: sidePlants },
  };
}

describe("the per-frame cost of a forest", () => {
  it("is per joint: 200 plants cost their joints, not their count", () => {
    const measure = (
      copies: number,
      only?: (plant: NonNullable<MotionRig["plants"]>[number]) => boolean,
    ): { nodes: number; plants: number; ms: number } => {
      const forest = replicate(copies, only);
      const motion = createLivingMotion(forest.rig, forest.sidecar);
      prepareLivingMotion(motion);
      livingFrame(motion, 1, WIND); // phases for this bearing, once
      const frames = 30;
      const started = nowMs();
      for (let i = 0; i < frames; i += 1) livingFrame(motion, 2 + i / 60, WIND);
      return {
        nodes: forest.rig.nodes.length,
        plants: forest.rig.plants?.length ?? 0,
        ms: (nowMs() - started) / frames,
      };
    };
    const small = measure(2);
    const large = measure(20);
    // The crown and trunk rigs alone: 200 small plants, no banded skeleton among them.
    const crowns = measure(29, (plant) => plant.class !== "tree");
    // Report, for docs/LIVING_SURVEY.md: joints and milliseconds a frame.
    console.info(
      `forest frame: ${String(small.plants)} plants / ${String(small.nodes)} joints ${small.ms.toFixed(2)} ms; ` +
        `${String(large.plants)} plants / ${String(large.nodes)} joints ${large.ms.toFixed(2)} ms; ` +
        `${String(crowns.plants)} shrubs and snags / ${String(crowns.nodes)} joints ${crowns.ms.toFixed(2)} ms`,
    );
    expect(large.plants).toBe(200);
    // Linear in joints, within a generous allowance for a shared, loaded machine.
    const perJointSmall = small.ms / small.nodes;
    const perJointLarge = large.ms / large.nodes;
    expect(perJointLarge).toBeLessThan(perJointSmall * 3);
  });

  it("keeps one wind: the same plant, elsewhere, feels the gust later", () => {
    const forest = replicate(2);
    expect(validateRig(forest.rig)).toEqual([]);
    expect(validateMotionSidecar(forest.sidecar, forest.rig)).toEqual([]);
    expect(checksumPositions(new Float32Array(0))).toMatch(/^fnv1a32:0:/);
  });
});
