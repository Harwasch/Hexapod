/**
 * `SplatDeformer` on a forest rig: the committed synthetic yard, many plants in one level-of-
 * detail tileset, beside a building and a lawn that must never move.
 *
 * What it holds the deformer to, against the real tiles, rig, sidecar and plant binding:
 *
 * - every gaussian the binding marks static is drawn from its canonical bytes, bit for bit, on
 *   the CPU path and — through the shader's transcription — on the GPU path, at every wind and
 *   under either motion model;
 * - a plant's gaussians follow that plant's joints and no other's;
 * - the GPU path's packing and arithmetic reproduce the CPU path on the plants;
 * - a forest rig without its binding, or a tile its binding does not know, is refused;
 * - and what binding a tile costs, per tile, is measured.
 */

import { beforeEach, describe, expect, it } from "vitest";

import {
  deform,
  deformPositions,
  flutterField,
  flutterHash,
  livingFrame,
  livingWindFromSettings,
  plantLabels,
  positionKeys,
  rigTileChecksums,
  checksumPositions,
  SKIN_INFLUENCES,
  staticAnchors,
  type FlutterField,
  type NodeTransform,
  type PlantBinding,
  type WindSettings,
} from "@twin/world";

import { BAKE_RESIDUAL_LIMIT_M, SplatDeformer } from "@/cesium/SplatDeformer";
import {
  clearSplatCaptures,
  digestSplatPositions,
  recordSplatCapture,
} from "@/cesium/splatCaptureRegistry";
import { transformPositions } from "@/cesium/splatFrames";
import { evaluateSplatMotion } from "@/cesium/splatGpuMotion";
import { positionWordOffset, splatTextureLayout } from "@/cesium/splatTexels";
import { bindTile } from "@/cesium/splatTiles";
import { bakeFixture } from "./splatFixture";

import {
  buildDrawCommand,
  fakeFactory,
  FakeHookedPrimitive,
  type FakeOwnedTexture,
} from "./splatGpuFixture";
import { FakeTile, FakeTiledPrimitive, FakeTiledTileset } from "./splatTilesFixture";
import {
  yardBake,
  yardBinding,
  yardChildrenOf,
  yardLeaves,
  yardMotion,
  yardRig,
  yardRootTransform,
  yardTiles,
} from "./splatYardFixture";

const GALE: WindSettings = { strength: 1, bearingDeg: 250 };
const BREEZE: WindSettings = { strength: 0.2, bearingDeg: 40 };
const CALM: WindSettings = { strength: 0, bearingDeg: 250 };

const ROOT = "splat.glb";
const SELECTIONS: readonly (readonly string[])[] = [[ROOT], yardChildrenOf(ROOT), yardLeaves];

function living(
  t: number,
  wind: WindSettings,
): { transforms: NodeTransform[]; flutter: FlutterField } {
  return livingFrame(yardMotion, t, livingWindFromSettings(wind, yardMotion.sidecar));
}

function tilesFor(uris: readonly string[]): FakeTile[] {
  return uris.map(
    (uri) => new FakeTile(uri, yardTiles.get(uri)?.local ?? new Float32Array(0), yardBake),
  );
}

function canonicalOf(uris: readonly string[]): Float32Array {
  const parts = uris.map((uri) => yardTiles.get(uri)?.local ?? new Float32Array(0));
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const part of parts) {
    out.set(part, at);
    at += part.length;
  }
  return out;
}

function labelsOf(uris: readonly string[]): Uint16Array {
  const parts = uris.map((uri) => {
    const local = yardTiles.get(uri)?.local ?? new Float32Array(0);
    const labels = plantLabels(yardBinding, checksumPositions(local));
    if (labels === undefined) throw new Error(`no binding for ${uri}`);
    return labels;
  });
  const out = new Uint16Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const part of parts) {
    out.set(part, at);
    at += part.length;
  }
  return out;
}

function commit(primitive: FakeTiledPrimitive, uris: readonly string[]): void {
  primitive._rootTransform = yardRootTransform;
  const packed = primitive.commit(tilesFor(uris));
  recordSplatCapture({
    count: primitive._numSplats,
    digest: digestSplatPositions(primitive._positions, primitive._numSplats),
    data: packed,
    at: 0,
  });
}

/** Splat `i`'s three position words from the last upload, as bits. */
function uploadedBits(primitive: FakeTiledPrimitive, i: number): number[] | undefined {
  const layout = splatTextureLayout(
    primitive._numSplats,
    primitive._splatRowMask,
    primitive._splatRowShift,
  );
  const upload = primitive.gaussianSplatTexture.uploads.at(-1);
  if (upload === undefined) return undefined;
  const word = positionWordOffset(i, layout) - upload.yOffset * layout.width * 4;
  if (word < 0 || word + 2 >= upload.words.length) return undefined;
  return [upload.words[word] ?? 0, upload.words[word + 1] ?? 0, upload.words[word + 2] ?? 0];
}

function bitsOf(value: number): number {
  return new Uint32Array(new Float32Array([value]).buffer)[0] ?? 0;
}

beforeEach(() => {
  clearSplatCaptures();
});

describe("the yard's plant binding against its tiles", () => {
  it("has an entry for every tile the rig lists, and the tiles are those tiles", () => {
    const accepted = rigTileChecksums(yardRig);
    for (const tile of yardTiles.values()) {
      const checksum = checksumPositions(tile.local);
      expect(accepted.has(checksum)).toBe(true);
      expect(plantLabels(yardBinding, checksum)?.length).toBe(tile.local.length / 3);
    }
  });
});

describe("a forest on the CPU path", () => {
  for (const uris of SELECTIONS) {
    it(`keeps every static gaussian at its bytes and moves the plants: ${String(uris.length)} tiles`, () => {
      const primitive = new FakeTiledPrimitive();
      commit(primitive, uris);
      const deformer = new SplatDeformer({
        tileset: new FakeTiledTileset(primitive),
        rig: yardRig,
        plantBinding: yardBinding,
      });
      const labels = labelsOf(uris);
      let staticCount = 0;
      for (const label of labels) if (label === 0) staticCount += 1;
      for (const [t, wind] of [
        [4, GALE],
        [4.05, GALE],
        [913.7, BREEZE],
      ] as const) {
        const frame = living(t, wind);
        const status = deformer.apply(frame.transforms, frame.flutter);
        expect(status.phase).toBe("ready");
        expect(status.motion).toBe("cpu");
        expect(status.plants).toBe(10);
        expect(status.staticSplats).toBe(staticCount);
        const audit = deformer.staticAudit();
        expect(audit?.staticSplats).toBe(staticCount);
        expect(audit?.staticMoved).toBe(0);
        expect(audit?.staticUnpinned).toBe(0);
        expect(audit?.plantMoved).toBeGreaterThan(0);
        // And in the words that went to the texture, not only in the buffer they came from.
        for (let i = 0; i < primitive._numSplats; i += 1) {
          if (labels[i] !== 0) continue;
          const words = uploadedBits(primitive, i);
          if (words === undefined) continue;
          for (let k = 0; k < 3; k += 1)
            expect(words[k]).toBe(bitsOf(primitive._positions[i * 3 + k] ?? 0));
        }
      }
      // Each plant's gaussians on its own joints; static ones on the pinned anchor.
      const skin = deformer.skin;
      const pinned = staticAnchors(yardRig);
      const plants = yardRig.plants ?? [];
      if (skin === undefined) throw new Error("no skin");
      for (let i = 0; i < labels.length; i += 1) {
        const label = labels[i] ?? 0;
        for (let k = 0; k < SKIN_INFLUENCES; k += 1) {
          const at = i * SKIN_INFLUENCES + k;
          if ((skin.weights[at] ?? 0) === 0) continue;
          const node = skin.nodes[at] ?? -1;
          if (label === 0) {
            expect(pinned[node]).toBe(1);
          } else {
            const plant = plants[label - 1];
            expect(node).toBeGreaterThanOrEqual(plant?.nodeStart ?? Infinity);
            expect(node).toBeLessThan(plant?.nodeEnd ?? -Infinity);
          }
        }
      }
      // Calm: the measured bytes, all of them.
      const calm = living(20, CALM);
      const status = deformer.apply(calm.transforms, calm.flutter);
      expect(status.displaced).toBe(false);
      for (let i = 0; i < primitive._numSplats; i += 1) {
        const words = uploadedBits(primitive, i);
        if (words === undefined) continue;
        for (let k = 0; k < 3; k += 1)
          expect(words[k]).toBe(bitsOf(primitive._positions[i * 3 + k] ?? 0));
      }
    }, 60_000);
  }

  it("pins the static anchor under the legacy model too, which knows nothing of plants", () => {
    const primitive = new FakeTiledPrimitive();
    commit(primitive, yardLeaves);
    const deformer = new SplatDeformer({
      tileset: new FakeTiledTileset(primitive),
      rig: yardRig,
      plantBinding: yardBinding,
    });
    const status = deformer.apply(deform(yardRig, 3, GALE), flutterField(yardRig, 3, GALE));
    expect(status.displaced).toBe(true);
    const audit = deformer.staticAudit();
    expect(audit?.staticMoved).toBe(0);
    expect(audit?.plantMoved).toBeGreaterThan(0);
  });
});

describe("a forest on the GPU path", () => {
  it("draws static gaussians exactly and plants as the CPU path does", () => {
    const factory = fakeFactory();
    const primitive = new FakeHookedPrimitive();
    const uris = yardLeaves;
    commit(primitive, uris);
    const deformer = new SplatDeformer({
      tileset: new FakeTiledTileset(primitive),
      rig: yardRig,
      gpu: factory,
      plantBinding: yardBinding,
    });
    expect(deformer.apply(living(0, CALM).transforms).motion).toBe("gpu");
    const uniforms = buildDrawCommand(primitive);
    const canonical = canonicalOf(uris);
    const labels = labelsOf(uris);
    const keys = positionKeys(canonical);
    const skin = deformer.skin;
    const assignment = deformer.assignment;
    if (skin === undefined || assignment === undefined) throw new Error("not attached");
    for (const [t, wind] of [
      [4, GALE],
      [3600.5, BREEZE],
    ] as const) {
      const frame = living(t, wind);
      const status = deformer.apply(frame.transforms, frame.flutter);
      expect(status.displaced).toBe(true);
      expect(uniforms.u_splatMotionActive?.()).toBe(1);
      const audit = deformer.staticAudit();
      expect(audit?.motion).toBe("gpu");
      expect(audit?.staticMoved).toBe(0);
      expect(audit?.plantMoved).toBeGreaterThan(0);
      const flutterTexture = factory.made.find(
        (texture: FakeOwnedTexture) =>
          texture.initial instanceof Float32Array &&
          texture.width === texture.height &&
          texture.width > 1,
      );
      const packed =
        flutterTexture === undefined
          ? undefined
          : { size: flutterTexture.width, data: flutterTexture.initial as Float32Array };
      // The CPU path's answer with the anchor held still, as the deformer holds it.
      const pinned = staticAnchors(yardRig);
      const transforms = frame.transforms.map((x, n) => ((pinned[n] ?? 0) === 1 ? undefined : x));
      const moved = deformPositions(
        canonical,
        assignment,
        transforms.map((x) => x ?? frame.transforms[0]) as NodeTransform[],
        undefined,
        frame.flutter,
        keys,
        skin,
      );
      const cpu = transformPositions(moved, yardBake, new Float32Array(moved.length));
      const motionData = deformer.gpuMotion?.motionData ?? new Float32Array(0);
      let worst = 0;
      for (let i = 0; i < labels.length; i += 1) {
        const rest = primitive._positions.subarray(i * 3, i * 3 + 3);
        const gpu = evaluateSplatMotion(
          motionData,
          skin.nodes.subarray(i * 4, i * 4 + 4),
          skin.weights.subarray(i * 4, i * 4 + 4),
          flutterHash(keys[i] ?? 0),
          [rest[0] ?? 0, rest[1] ?? 0, rest[2] ?? 0],
          packed,
        );
        if (labels[i] === 0) {
          for (let k = 0; k < 3; k += 1) expect(Object.is(gpu[k], rest[k])).toBe(true);
          continue;
        }
        for (let k = 0; k < 3; k += 1)
          worst = Math.max(worst, Math.abs((gpu[k] ?? 0) - (cpu[i * 3 + k] ?? 0)));
      }
      expect(worst).toBeLessThan(2e-5);
    }
    deformer.destroy();
  }, 60_000);
});

describe("refusals", () => {
  it("refuses a forest rig with no binding", () => {
    const primitive = new FakeTiledPrimitive();
    commit(primitive, yardLeaves);
    const deformer = new SplatDeformer({ tileset: new FakeTiledTileset(primitive), rig: yardRig });
    const status = deformer.apply(living(4, GALE).transforms);
    expect(status.phase).toBe("refused");
    expect(status.reason).toBe("binding");
  });

  it("refuses a tile its binding does not know", () => {
    const primitive = new FakeTiledPrimitive();
    commit(primitive, yardLeaves);
    const first = yardLeaves[0] ?? "";
    const missing = checksumPositions(yardTiles.get(first)?.local ?? new Float32Array(0));
    const tiles = new Map(yardBinding.tiles);
    tiles.delete(missing);
    const partial: PlantBinding = { ...yardBinding, tiles };
    const deformer = new SplatDeformer({
      tileset: new FakeTiledTileset(primitive),
      rig: yardRig,
      plantBinding: partial,
    });
    const status = deformer.apply(living(4, GALE).transforms);
    expect(status.phase).toBe("refused");
    expect(status.reason).toBe("checksum");
  });
});

describe("what binding a tile costs", () => {
  it("is measured per tile, and reported", () => {
    const accepted = rigTileChecksums(yardRig);
    const rows: string[] = [];
    let total = 0;
    let splats = 0;
    for (const tile of yardTiles.values()) {
      const baked = bakeFixture(tile.local, yardBake);
      // Warm, as a tile bound in a session is: candidate lists cached per plant rig.
      bindTile(baked, yardBake, yardRig, accepted, BAKE_RESIDUAL_LIMIT_M, yardBinding);
      const started = performance.now();
      const result = bindTile(
        baked,
        yardBake,
        yardRig,
        accepted,
        BAKE_RESIDUAL_LIMIT_M,
        yardBinding,
      );
      const ms = performance.now() - started;
      expect(result.kind).toBe("bound");
      total += ms;
      splats += tile.local.length / 3;
      rows.push(`${tile.uri} ${String(tile.local.length / 3)} gaussians ${ms.toFixed(2)} ms`);
    }
    console.info(
      `yard tile binding: ${String(yardTiles.size)} tiles, ${String(splats)} gaussians, ` +
        `${total.toFixed(1)} ms (${((total / splats) * 1000).toFixed(2)} µs a gaussian)\n  ${rows.join("\n  ")}`,
    );
    expect(total).toBeGreaterThan(0);
  });
});
