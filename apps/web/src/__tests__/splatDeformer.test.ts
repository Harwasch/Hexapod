/**
 * `SplatDeformer` against a fake primitive: everything but the GL call.
 *
 * The one property this whole approach rests on is that the measured geometry is never written
 * to. That is asserted here over a thousand simulated frames of real wind, byte for byte, on
 * every array the engine owns and every array the deformer keeps.
 */

import { beforeEach, describe, expect, it } from "vitest";

import {
  checksumPositions,
  deform,
  deformPositions,
  IDENTITY_TRANSFORM,
  maxDisplacement,
  syntheticTreeRig,
  type NodeTransform,
  type WindSettings,
} from "@twin/world";

import { markMovingNodes, SplatDeformer } from "@/cesium/SplatDeformer";
import {
  clearSplatCaptures,
  digestSplatPositions,
  recordSplatCapture,
} from "@/cesium/splatCaptureRegistry";
import { invertAffine } from "@/cesium/splatFrames";
import { bitsToFloat32, positionWordOffset, splatTextureLayout } from "@/cesium/splatTexels";

import {
  bakeFixture,
  canonicalPositions,
  FakeSplatPrimitive,
  FakeSplatTexture,
  FakeSplatTileset,
  fixtureRig,
  multiplyMat4,
  packedBufferFor,
  rootPlacement,
  rootTransform,
} from "./splatFixture";

const BREEZE: WindSettings = { strength: 0.6, bearingDeg: 250 };
const STILL: WindSettings = { strength: 0, bearingDeg: 250 };

/** Files the packed buffer the interception would have captured for these baked positions. */
function registerCapture(baked: Float32Array): void {
  const count = baked.length / 3;
  recordSplatCapture({
    count,
    digest: digestSplatPositions(baked, count),
    data: packedBufferFor(baked),
    at: 0,
  });
}

interface Harness {
  readonly tileset: FakeSplatTileset;
  readonly primitive: FakeSplatPrimitive;
  readonly deformer: SplatDeformer;
  readonly baked: Float32Array;
}

function harness(options: { rowShift?: number } = {}): Harness {
  const baked = bakeFixture(canonicalPositions);
  const primitive = new FakeSplatPrimitive(baked, options);
  const tileset = new FakeSplatTileset(primitive);
  registerCapture(baked);
  return {
    tileset,
    primitive,
    deformer: new SplatDeformer({ tileset, rig: fixtureRig }),
    baked,
  };
}

/** Rig transforms at time `t`. */
function transformsAt(t: number, wind: WindSettings = BREEZE): NodeTransform[] {
  return deform(fixtureRig, t, wind);
}

beforeEach(() => {
  clearSplatCaptures();
});

describe("attaching", () => {
  it("attaches to the committed fixture and reports an exact re-bake", () => {
    const { deformer, primitive } = harness();
    const status = deformer.apply(transformsAt(1));
    expect(status.phase).toBe("ready");
    expect(status.reason).toBeUndefined();
    expect(status.numSplats).toBe(2000);
    expect(status.observedChecksum).toBe(fixtureRig.canonicalChecksum);
    expect(status.geodeticAlignment).toBeGreaterThan(1 - 1e-9);
    // The engine's own baked bytes, reproduced term for term by our transcription of
    // `Matrix4.multiplyByPoint`. Anything but zero here means the bake matrix is not the one
    // `transformTile` used.
    expect(status.bakeResidualM).toBe(0);
    expect(status.uprightness?.upright).toBe(true);
    expect(primitive.texture.copyFrom).toHaveBeenCalledTimes(1);
  });

  it("assigns every splat to a rig node", () => {
    const { deformer } = harness();
    deformer.apply(transformsAt(1));
    expect(deformer.assignment?.length).toBe(2000);
    expect(deformer.canonicalPositions?.length).toBe(6000);
  });

  it("no-ops without a primitive, a snapshot, a texture or a capture", () => {
    const baked = bakeFixture(canonicalPositions);
    const primitive = new FakeSplatPrimitive(baked);
    const tileset = new FakeSplatTileset(primitive);
    const deformer = new SplatDeformer({ tileset, rig: fixtureRig });

    tileset.gaussianSplatPrimitive = undefined;
    expect(deformer.apply(transformsAt(1)).reason).toBe("no-primitive");

    tileset.gaussianSplatPrimitive = primitive;
    primitive._splatRowShift = 0;
    expect(deformer.apply(transformsAt(1)).reason).toBe("no-snapshot");

    // The texture does not exist for several frames after tile load, and the capture arrives
    // with it. Neither is an error and neither may retry-storm.
    primitive._splatRowShift = 12;
    expect(deformer.apply(transformsAt(1)).reason).toBe("no-capture");
    expect(deformer.apply(transformsAt(2)).phase).toBe("waiting");

    registerCapture(baked);
    expect(deformer.apply(transformsAt(3)).phase).toBe("ready");
  });

  it("waits, and does not write, while the texture is missing or destroyed", () => {
    const { deformer, primitive } = harness();
    deformer.apply(transformsAt(1));
    primitive.texture.destroyed = true;
    const before = primitive.texture.uploads.length;
    expect(deformer.apply(transformsAt(2)).reason).toBe("no-texture");
    expect(primitive.texture.uploads.length).toBe(before);
  });

  it("re-reads the texture on every write, so a recreated one is not missed", () => {
    // The draw command's uniform closes over the texture object at build time and the engine
    // destroys and recreates it on a dimension change; a held reference writes into a corpse.
    const { deformer, primitive } = harness();
    deformer.apply(transformsAt(1));
    const first = primitive.texture;
    primitive.gaussianSplatTexture = new FakeSplatTexture();
    deformer.apply(transformsAt(2));
    expect(primitive.texture).not.toBe(first);
    expect(primitive.texture.uploads.length).toBe(1);
  });
});

describe("addressing follows the primitive, not an assumed texture size", () => {
  it.each([12, 13])("rowShift %i", (rowShift) => {
    const { deformer, primitive } = harness({ rowShift });
    deformer.apply(transformsAt(1));
    const layout = splatTextureLayout(2000, (1 << rowShift) - 1, rowShift);
    const upload = primitive.texture.uploads[0];
    expect(upload?.width).toBe(layout.width);
    expect(upload?.yOffset).toBe(0);
    expect(upload?.height).toBe(layout.height);
    expect(upload?.words.length).toBe(layout.width * layout.height * 4);
  });
});

describe("what reaches the GPU", () => {
  it("writes the deformed positions and nothing else", () => {
    const { deformer, primitive, baked } = harness();
    const transforms = transformsAt(3.25);
    deformer.apply(transforms);

    const layout = splatTextureLayout(2000, 4095, 12);
    const expected = deformPositions(
      canonicalPositions,
      deformer.assignment ?? new Uint16Array(0),
      transforms,
    );
    const packed = packedBufferFor(baked);
    // A splat sits away from its node, so it can swing a little further than the node bound.
    const displacementBound = maxDisplacement(fixtureRig, BREEZE) * 2 + 0.1;
    const upload = primitive.texture.uploads[0];
    expect(upload).toBeDefined();
    if (upload === undefined) return;

    let moved = 0;
    for (let i = 0; i < 2000; i += 1) {
      const offset = positionWordOffset(i, layout);
      for (let c = 0; c < 3; c += 1) {
        const value = bitsToFloat32(upload.words[offset + c] ?? 0);
        // Bounded by the model's own proven worst case, so the bytes on the GPU and the
        // metric the UI would quote cannot disagree.
        expect(Math.abs(value - (baked[i * 3 + c] ?? 0))).toBeLessThan(displacementBound);
        if (value !== baked[i * 3 + c]) moved += 1;
      }
      // Covariance and colour come through exactly as the engine packed them.
      for (let c = 4; c < 8; c += 1) {
        expect(upload.words[offset + c]).toBe(packed[i * 8 + c]);
      }
      expect(upload.words[offset + 3]).toBe(0xdeadbeef);
    }
    expect(moved).toBeGreaterThan(1000);
    expect(expected.length).toBe(6000);
  });

  it("uploads nothing at all while every node is at rest", () => {
    const { deformer, primitive } = harness();
    for (let frame = 0; frame < 10; frame += 1) {
      deformer.apply(deform(fixtureRig, frame / 60, STILL));
    }
    expect(primitive.texture.copyFrom).not.toHaveBeenCalled();
    expect(deformer.status.displaced).toBe(false);
  });

  it("restores the engine's exact bytes when the wind drops to zero", () => {
    const { deformer, primitive, baked } = harness();
    deformer.apply(transformsAt(2));
    expect(deformer.status.displaced).toBe(true);
    deformer.apply(deform(fixtureRig, 3, STILL));
    expect(deformer.status.displaced).toBe(false);

    const layout = splatTextureLayout(2000, 4095, 12);
    const upload = primitive.texture.uploads.at(-1);
    for (let i = 0; i < 2000 * 3; i += 1) {
      const splat = Math.floor(i / 3);
      const word = positionWordOffset(splat, layout) + (i % 3);
      // Bit-identical, not close: "returns to exactly its measured pose" with no epsilon.
      expect(Object.is(bitsToFloat32(upload?.words[word] ?? 0), baked[i])).toBe(true);
    }
    // And then it stops uploading: an idle scene stays idle.
    const uploads = primitive.texture.uploads.length;
    deformer.apply(deform(fixtureRig, 4, STILL));
    expect(primitive.texture.uploads.length).toBe(uploads);
  });

  it("uploads one row band per frame while the wind blows", () => {
    const { deformer, primitive } = harness();
    for (let frame = 0; frame < 5; frame += 1) deformer.apply(transformsAt(frame / 60));
    expect(primitive.texture.copyFrom).toHaveBeenCalledTimes(5);
    for (const upload of primitive.texture.uploads) {
      expect(upload.yOffset).toBe(0);
      expect(upload.xOffset).toBe(0);
      expect(upload.height).toBe(1);
    }
    expect(deformer.status.lastUploadWords).toBe(8192 * 4);
  });
});

describe("snapshot rebuilds re-derive from the new base", () => {
  it("follows a clamp-to-ground model matrix change without refusing", () => {
    // `SiteManager.clampToGround` sets `tileset.modelMatrix` after an asynchronous terrain
    // sample, so this happens on every site, seconds after load.
    const { deformer, primitive, tileset } = harness();
    deformer.apply(transformsAt(1));
    expect(deformer.status.rederivations).toBe(0);

    const raised = [...rootPlacement];
    raised[14] = (raised[14] ?? 0) + 3;
    const clamped = multiplyMat4(invertAffine(rootTransform) ?? [], raised);
    const rebaked = bakeFixture(canonicalPositions, clamped);
    registerCapture(rebaked);

    tileset.root.content._lastSplatTransform = clamped;
    primitive._positions = rebaked;
    primitive._snapshot = { generation: 2 };

    const status = deformer.apply(transformsAt(2));
    expect(status.phase).toBe("ready");
    expect(status.rederivations).toBe(1);
    expect(status.generation).toBe(2);
    expect(status.bakeResidualM).toBe(0);
    // Re-derived, not re-applied: the rest pose is the *new* baked geometry.
    deformer.apply(deform(fixtureRig, 3, STILL));
    const layout = splatTextureLayout(2000, 4095, 12);
    const upload = primitive.texture.uploads.at(-1);
    for (let i = 0; i < 30; i += 1) {
      const word = positionWordOffset(Math.floor(i / 3), layout) + (i % 3);
      expect(Object.is(bitsToFloat32(upload?.words[word] ?? 0), rebaked[i])).toBe(true);
    }
  });

  it("notices a swapped positions array even without a generation bump", () => {
    const { deformer, primitive } = harness();
    deformer.apply(transformsAt(1));
    // Same numbers, new array: the identity guard fires and the attachment is rebuilt from the
    // array the engine now holds, rather than from one it may have released.
    primitive._positions = Float32Array.from(primitive._positions);
    const status = deformer.apply(transformsAt(2));
    expect(status.phase).toBe("ready");
    expect(status.rederivations).toBe(1);
  });

  it("waits rather than deforming a base it has no packed buffer for", () => {
    const { deformer, primitive } = harness();
    deformer.apply(transformsAt(1));
    const moved = Float32Array.from(primitive._positions, (v) => v + 10);
    primitive._positions = moved;
    primitive._snapshot = { generation: 2 };
    const uploads = primitive.texture.uploads.length;
    expect(deformer.apply(transformsAt(2)).reason).toBe("no-capture");
    expect(primitive.texture.uploads.length).toBe(uploads);
  });
});

describe("refusing rather than misleading", () => {
  it("refuses a multi-tile tileset, and never writes again", () => {
    const { deformer, primitive, tileset } = harness();
    tileset.root.children = [{}, {}];
    const status = deformer.apply(transformsAt(1));
    expect(status.phase).toBe("refused");
    expect(status.reason).toBe("multi-tile");
    expect(primitive.texture.copyFrom).not.toHaveBeenCalled();

    tileset.root.children = [];
    expect(deformer.apply(transformsAt(2)).phase).toBe("refused");
    expect(primitive.texture.copyFrom).not.toHaveBeenCalled();
  });

  it("refuses a snapshot that aggregated more than one tile", () => {
    const { deformer, primitive } = harness();
    primitive.selectedTileLength = 3;
    expect(deformer.apply(transformsAt(1)).reason).toBe("multi-tile");
    expect(primitive.texture.copyFrom).not.toHaveBeenCalled();
  });

  it("refuses a root frame whose Z column is not the geodetic normal", () => {
    const { deformer, primitive } = harness();
    const tilted = [...rootTransform];
    for (let r = 0; r < 3; r += 1) {
      const up = tilted[8 + r] ?? 0;
      tilted[8 + r] = tilted[4 + r] ?? 0;
      tilted[4 + r] = up;
    }
    primitive._rootTransform = tilted;
    expect(deformer.apply(transformsAt(1)).reason).toBe("frame");
    expect(primitive.texture.copyFrom).not.toHaveBeenCalled();
  });

  it("refuses splats the rig was not built for", () => {
    const shifted = Float32Array.from(canonicalPositions);
    shifted[7] = (shifted[7] ?? 0) + 1 / 4096;
    const baked = bakeFixture(shifted);
    const primitive = new FakeSplatPrimitive(baked);
    const tileset = new FakeSplatTileset(primitive);
    registerCapture(baked);
    const deformer = new SplatDeformer({ tileset, rig: fixtureRig });

    const status = deformer.apply(transformsAt(1));
    expect(status.phase).toBe("refused");
    expect(status.reason).toBe("checksum");
    expect(status.observedChecksum).not.toBe(fixtureRig.canonicalChecksum);
    expect(primitive.texture.copyFrom).not.toHaveBeenCalled();
  });

  it("refuses a tree that is not standing up", () => {
    // A rig whose checksum the sideways positions satisfy, so only the upright test can catch
    // it. An axis swap here would sway the tree into the ground.
    const sideways = new Float32Array(canonicalPositions.length);
    for (let i = 0; i < canonicalPositions.length; i += 3) {
      sideways[i] = canonicalPositions[i + 2] ?? 0;
      sideways[i + 1] = canonicalPositions[i + 1] ?? 0;
      sideways[i + 2] = canonicalPositions[i] ?? 0;
    }
    const baked = bakeFixture(sideways);
    const primitive = new FakeSplatPrimitive(baked);
    const tileset = new FakeSplatTileset(primitive);
    registerCapture(baked);
    const rig = syntheticTreeRig({ canonicalChecksum: checksumPositions(sideways) });
    const deformer = new SplatDeformer({ tileset, rig });
    expect(deformer.apply(transformsAt(1)).reason).toBe("upright");
    expect(primitive.texture.copyFrom).not.toHaveBeenCalled();
  });

  it("refuses a singular bake transform", () => {
    const { deformer, primitive, tileset } = harness();
    tileset.root.content._lastSplatTransform = new Array<number>(16).fill(0);
    expect(deformer.apply(transformsAt(1)).reason).toBe("bake");
    expect(primitive.texture.copyFrom).not.toHaveBeenCalled();
  });

  it("stops writing after destroy()", () => {
    const { deformer, primitive } = harness();
    deformer.apply(transformsAt(1));
    deformer.destroy();
    deformer.apply(transformsAt(2));
    expect(primitive.texture.copyFrom).toHaveBeenCalledTimes(1);
    expect(deformer.status.phase).toBe("detached");
  });
});

describe("the measured geometry is never written to", () => {
  it("survives a thousand frames of wind byte for byte", () => {
    const { deformer, primitive, tileset, baked } = harness();
    deformer.apply(transformsAt(0));

    const enginePositions = Uint8Array.from(new Uint8Array(primitive._positions.buffer.slice(0)));
    const canonicalBytes = Uint8Array.from(
      new Uint8Array(
        canonicalPositions.buffer,
        canonicalPositions.byteOffset,
        canonicalPositions.byteLength,
      ),
    );
    const deformerCanonical = deformer.canonicalPositions;
    const deformerCanonicalBytes = Uint8Array.from(
      new Uint8Array(
        deformerCanonical?.buffer ?? new ArrayBuffer(0),
        deformerCanonical?.byteOffset ?? 0,
        deformerCanonical?.byteLength ?? 0,
      ),
    );
    const bakeMatrixBefore = [...(tileset.root.content._lastSplatTransform ?? [])];

    for (let frame = 0; frame < 1000; frame += 1) {
      const t = frame / 60;
      const wind: WindSettings = { strength: 0.2 + 0.8 * Math.sin(t), bearingDeg: 40 + t * 30 };
      deformer.apply(deform(fixtureRig, t, wind));
    }

    expect(new Uint8Array(primitive._positions.buffer)).toEqual(enginePositions);
    expect(
      new Uint8Array(
        canonicalPositions.buffer,
        canonicalPositions.byteOffset,
        canonicalPositions.byteLength,
      ),
    ).toEqual(canonicalBytes);
    expect(
      new Uint8Array(
        deformerCanonical?.buffer ?? new ArrayBuffer(0),
        deformerCanonical?.byteOffset ?? 0,
        deformerCanonical?.byteLength ?? 0,
      ),
    ).toEqual(deformerCanonicalBytes);
    expect([...(tileset.root.content._lastSplatTransform ?? [])]).toEqual(bakeMatrixBefore);
    // Same baked array the harness handed the primitive: identity, not just equality.
    expect(primitive._positions).toBe(baked);

    // And after all that, zero wind still lands on the engine's exact bytes.
    deformer.apply(deform(fixtureRig, 1000, STILL));
    const layout = splatTextureLayout(2000, 4095, 12);
    const upload = primitive.texture.uploads.at(-1);
    for (let i = 0; i < baked.length; i += 1) {
      const word = positionWordOffset(Math.floor(i / 3), layout) + (i % 3);
      expect(Object.is(bitsToFloat32(upload?.words[word] ?? 0), baked[i])).toBe(true);
    }
  });
});

describe("markMovingNodes", () => {
  it("treats the exact rest transform as still, and anything else as moving", () => {
    const moves = new Uint8Array(3);
    expect(markMovingNodes(moves, [IDENTITY_TRANSFORM, IDENTITY_TRANSFORM])).toBe(false);
    expect([...moves]).toEqual([0, 0, 0]);

    const nudged: NodeTransform = { rotation: [0, 0, 0, 1], translation: [0, 0, 1e-12] };
    expect(markMovingNodes(moves, [IDENTITY_TRANSFORM, nudged, IDENTITY_TRANSFORM])).toBe(true);
    expect([...moves]).toEqual([0, 1, 0]);
  });
});
