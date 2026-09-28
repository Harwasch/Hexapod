/**
 * The GPU motion path under Living Mode: the sidecar's modal sway **and** its advected leaf
 * flutter, evaluated from the textures exactly as the vertex shader reads them, against the CPU
 * path's `deformPositions` — on the single-tile synthetic tree and on the level-of-detail
 * fixture, both of which carry a `motion.json`.
 *
 * `evaluateSplatMotion` is the shader transcribed; what these tests prove is that the packing
 * (node rows, the per-frame flutter frame, the gathered flutter texture) and that arithmetic
 * reproduce the CPU path to micrometres, at an early time and at one where the advection has
 * carried the field thousands of texels, and that the per-frame upload stays per-node small.
 */

import { readFileSync } from "node:fs";

import { beforeEach, describe, expect, it } from "vitest";

import {
  assignSplatsToNodes,
  deformPositions,
  flutterHash,
  isAdvectedFlutter,
  livingFrame,
  livingWindFromSettings,
  loadLivingMotion,
  positionKeys,
  type LivingMotion,
  type MotionRig,
  type Season,
  type WindSettings,
} from "@twin/world";

import { SplatDeformer } from "@/cesium/SplatDeformer";
import { clearSplatCaptures } from "@/cesium/splatCaptureRegistry";
import { transformPositions, type Mat4 } from "@/cesium/splatFrames";
import {
  evaluateSplatMotion,
  FLUTTER_FRAME_TEXEL,
  FLUTTER_FRAME_TEXELS,
  MOTION_TEXTURE_WIDTH,
} from "@/cesium/splatGpuMotion";
import type { SplatTilesetLike } from "@/cesium/splatInternals";

import {
  bakeFixture,
  bakeMatrix,
  canonicalPositions,
  FakeSplatTileset,
  fixturePath,
  fixtureRig,
} from "./splatFixture";
import {
  buildDrawCommand,
  fakeFactory,
  FakeHookedPrimitive,
  FakeHookedSinglePrimitive,
  type FakeOwnedTexture,
} from "./splatGpuFixture";
import {
  childrenOf,
  FakeTile,
  FakeTiledTileset,
  lodBake,
  lodLeaves,
  lodPath,
  lodRig,
  lodTiles,
} from "./splatTilesFixture";

const GALE: WindSettings = { strength: 1, bearingDeg: 250 };
const BREEZE: WindSettings = { strength: 0.35, bearingDeg: 40 };
const CALM: WindSettings = { strength: 0, bearingDeg: 250 };

const treeMotion = loadLivingMotion(
  fixtureRig,
  readFileSync(fixturePath("source/motion.json"), "utf8"),
);
const lodMotion = loadLivingMotion(lodRig, readFileSync(lodPath("motion.json"), "utf8"));

beforeEach(() => {
  clearSplatCaptures();
});

interface Case {
  readonly name: string;
  readonly rig: MotionRig;
  readonly motion: LivingMotion;
  readonly bake: Mat4;
  /** A deformer on the GPU path over a hooked fake, and the snapshot it sees. */
  setup(factory: ReturnType<typeof fakeFactory>): {
    deformer: SplatDeformer;
    primitive: { vertexMotion: unknown; _positions: Float32Array };
    canonical: Float32Array;
  };
}

function lodSelection(uris: readonly string[]): Float32Array {
  const parts = uris.map((uri) => lodTiles.get(uri)?.local ?? new Float32Array(0));
  const out = new Float32Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

function lodCase(name: string, uris: readonly string[]): Case {
  return {
    name,
    rig: lodRig,
    motion: lodMotion,
    bake: lodBake,
    setup(factory) {
      const primitive = new FakeHookedPrimitive();
      primitive.commit(
        uris.map((uri) => new FakeTile(uri, lodTiles.get(uri)?.local ?? new Float32Array(0))),
      );
      const tileset: SplatTilesetLike = new FakeTiledTileset(primitive);
      const deformer = new SplatDeformer({ tileset, rig: lodRig, gpu: factory });
      return { deformer, primitive, canonical: lodSelection(uris) };
    },
  };
}

const CASES: readonly Case[] = [
  {
    name: "the synthetic tree (one tile)",
    rig: fixtureRig,
    motion: treeMotion,
    bake: bakeMatrix,
    setup(factory) {
      const primitive = new FakeHookedSinglePrimitive(bakeFixture(canonicalPositions));
      const deformer = new SplatDeformer({
        tileset: new FakeSplatTileset(primitive),
        rig: fixtureRig,
        gpu: factory,
      });
      return { deformer, primitive, canonical: canonicalPositions };
    },
  },
  lodCase("the LOD fixture's middle level", childrenOf("splat.glb")),
  lodCase("the LOD fixture's leaves", lodLeaves),
];

function frame(motion: LivingMotion, t: number, wind: WindSettings, season: Season = "summer") {
  return livingFrame(motion, t, livingWindFromSettings(wind, motion.sidecar, season));
}

/** The flutter texture the path uploaded: the one float texture that is square and large. */
function isFlutterUpload(texture: FakeOwnedTexture): boolean {
  return (
    texture.initial instanceof Float32Array && texture.width === texture.height && texture.width > 1
  );
}

function flutterUpload(made: readonly FakeOwnedTexture[]): FakeOwnedTexture | undefined {
  return made.find(isFlutterUpload);
}

describe("the GPU path under Living Mode", () => {
  for (const test of CASES) {
    it(`draws the advected leaf flutter as the CPU path does: ${test.name}`, () => {
      const factory = fakeFactory();
      const { deformer, primitive, canonical } = test.setup(factory);
      expect(deformer.apply(frame(test.motion, 0, CALM).transforms).motion).toBe("gpu");
      const uniforms = buildDrawCommand(
        primitive as { vertexMotion: FakeHookedPrimitive["vertexMotion"] },
      );

      // The deformer binds the splats as an independent reading of the rig does.
      const assignment = assignSplatsToNodes(canonical, test.rig);
      expect(deformer.assignment).toEqual(assignment);
      const keys = positionKeys(canonical);
      const count = canonical.length / 3;

      for (const [t, wind, season] of [
        [5, GALE, "summer"],
        [5.05, GALE, "summer"],
        // An hour and a half in: the field has been carried ~3·10⁵ texels downwind.
        [5400.25, GALE, "summer"],
        [71.3, BREEZE, "summer"],
        // Winter: sway without flutter, so the flutter frame reads as none.
        [12, GALE, "winter"],
      ] as const) {
        const living = frame(test.motion, t, wind, season);
        expect(isAdvectedFlutter(living.flutter)).toBe(true);
        expect(living.flutter.still).toBe(season === "winter");
        const status = deformer.apply(living.transforms, living.flutter);
        expect(status.displaced).toBe(true);
        expect(uniforms.u_splatMotionActive?.()).toBe(1);

        const texture = flutterUpload(factory.made);
        expect(texture).toBeDefined();
        if (texture === undefined) return;
        expect(texture.width).toBe(living.flutter.texture.size);
        expect(uniforms.u_splatFlutterTexture?.()).toBe(texture);
        const packed = { size: texture.width, data: texture.initial as Float32Array };

        const motionData = deformer.gpuMotion?.motionData ?? new Float32Array(0);
        const moved = deformPositions(
          canonical,
          assignment,
          living.transforms,
          undefined,
          living.flutter,
          keys,
        );
        const swayOnly = deformPositions(canonical, assignment, living.transforms);
        const cpu = transformPositions(moved, test.bake, new Float32Array(moved.length));
        const cpuSway = transformPositions(swayOnly, test.bake, new Float32Array(moved.length));
        let worst = 0;
        let flutter = 0;
        for (let i = 0; i < count; i += 1) {
          const rest = primitive._positions.subarray(i * 3, i * 3 + 3);
          const gpu = evaluateSplatMotion(
            motionData,
            assignment[i] ?? 0,
            flutterHash(keys[i] ?? 0),
            [rest[0] ?? 0, rest[1] ?? 0, rest[2] ?? 0],
            packed,
          );
          for (let k = 0; k < 3; k += 1) {
            const expected = cpu[i * 3 + k] ?? 0;
            worst = Math.max(worst, Math.abs((gpu[k] ?? 0) - expected));
            flutter = Math.max(flutter, Math.abs(expected - (cpuSway[i * 3 + k] ?? 0)));
          }
        }
        // float32 textures against the CPU path's float32 output: micrometres.
        expect(worst).toBeLessThan(2e-5);
        // …and the flutter is really there to be matched, millimetres to centimetres of it.
        if (season === "winter") expect(flutter).toBe(0);
        else expect(flutter).toBeGreaterThan(1e-3);

        // Per frame: the node rows and the flutter frame. The texture went up once.
        if (t !== 5) {
          const rows = Math.ceil((test.rig.nodes.length * 4) / MOTION_TEXTURE_WIDTH);
          expect(status.lastUploadWords).toBe(
            rows * MOTION_TEXTURE_WIDTH * 4 + FLUTTER_FRAME_TEXELS * 4,
          );
        }
      }
      // One seed, one upload of its texture, however many frames.
      expect(factory.made.filter(isFlutterUpload)).toHaveLength(1);

      // Calm: nothing drawn displaced, nothing uploaded.
      const calm = frame(test.motion, 20, CALM);
      const status = deformer.apply(calm.transforms, calm.flutter);
      expect(status.displaced).toBe(false);
      expect(status.lastUploadWords).toBe(0);
      expect(uniforms.u_splatMotionActive?.()).toBe(0);
      deformer.destroy();
      expect(factory.made.every((texture) => texture.destroyed)).toBe(true);
    }, 60_000);
  }

  it("keeps the flutter frame's numbers small whatever the advection has grown to", () => {
    const factory = fakeFactory();
    const setup = CASES[0]?.setup(factory);
    if (setup === undefined) throw new Error("no case");
    const { deformer, primitive } = setup;
    deformer.apply(frame(treeMotion, 0, CALM).transforms);
    buildDrawCommand(primitive as { vertexMotion: FakeHookedPrimitive["vertexMotion"] });
    const late = frame(treeMotion, 86_400, GALE);
    expect(late.flutter.advectionTexels).toBeGreaterThan(1e5);
    deformer.apply(late.transforms, late.flutter);
    const data = deformer.gpuMotion?.motionData ?? new Float32Array(0);
    const size = late.flutter.texture.size;
    for (let j = 0; j < 6; j += 1) {
      const offset = data[(FLUTTER_FRAME_TEXEL + 4 + j) * 4 + 3] ?? Number.NaN;
      expect(Math.abs(offset)).toBeLessThanOrEqual(size / 2);
    }
  });
});
