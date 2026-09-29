/**
 * Switching a live deformer between the GPU and CPU paths ("Motion on GPU").
 *
 * The one property a switch must keep is the one the whole feature keeps: nothing displaced is
 * ever left behind. Leaving the CPU path, the attribute texture must get the engine's exact
 * bytes back before the shader takes over (it draws that texture as its rest pose); leaving
 * the GPU path, the hook must be uninstalled before the CPU path starts writing, so the two
 * never displace the same splat twice. Both are asserted bit for bit, mid-gust.
 */

import { beforeEach, describe, expect, it } from "vitest";

import {
  deform,
  flutterField,
  type FlutterField,
  type NodeTransform,
  type WindSettings,
} from "@twin/world";

import { SplatDeformer } from "@/cesium/SplatDeformer";
import {
  clearSplatCaptures,
  digestSplatPositions,
  recordSplatCapture,
} from "@/cesium/splatCaptureRegistry";
import { bitsToFloat32, positionWordOffset, splatTextureLayout } from "@/cesium/splatTexels";

import {
  bakeFixture,
  canonicalPositions,
  FakeSplatTileset,
  FIXTURE_SPLATS,
  fixtureRig,
  packedBufferFor,
} from "./splatFixture";
import { buildDrawCommand, fakeFactory, FakeHookedSinglePrimitive } from "./splatGpuFixture";

const BREEZE: WindSettings = { strength: 0.6, bearingDeg: 250 };
const STILL: WindSettings = { strength: 0, bearingDeg: 250 };

function frameAt(t: number, wind: WindSettings = BREEZE): [NodeTransform[], FlutterField] {
  return [deform(fixtureRig, t, wind), flutterField(fixtureRig, t, wind)];
}

function harness(gpu: boolean) {
  const baked = bakeFixture(canonicalPositions);
  const primitive = new FakeHookedSinglePrimitive(baked);
  const tileset = new FakeSplatTileset(primitive);
  recordSplatCapture({
    count: FIXTURE_SPLATS,
    digest: digestSplatPositions(baked, FIXTURE_SPLATS),
    data: packedBufferFor(baked),
    at: 0,
  });
  const factory = fakeFactory();
  const deformer = new SplatDeformer({
    tileset,
    rig: fixtureRig,
    gpu: gpu ? factory : undefined,
  });
  return { baked, primitive, factory, deformer };
}

/** Every splat's position words in `upload`, compared bit for bit with the engine's. */
function expectMeasuredPose(
  upload: { words: Uint32Array; yOffset: number } | undefined,
  baked: Float32Array,
): void {
  expect(upload).toBeDefined();
  if (upload === undefined) return;
  const layout = splatTextureLayout(FIXTURE_SPLATS, 4095, 12);
  for (let i = 0; i < FIXTURE_SPLATS * 3; i += 1) {
    const splat = Math.floor(i / 3);
    const word = positionWordOffset(splat, layout) - upload.yOffset * layout.width * 4 + (i % 3);
    if (word < 0 || word >= upload.words.length) continue;
    expect(Object.is(bitsToFloat32(upload.words[word] ?? 0), baked[i])).toBe(true);
  }
}

beforeEach(() => {
  clearSplatCaptures();
});

describe("switching paths", () => {
  it("reports why it is on the CPU path before and after attaching", () => {
    const { deformer } = harness(false);
    expect(deformer.status.cpuReason).toBe("no-factory");
    expect(deformer.apply(...frameAt(1)).cpuReason).toBe("no-factory");
    const gpu = harness(true).deformer;
    expect(gpu.status.cpuReason).toBeUndefined();
    const status = gpu.apply(...frameAt(1));
    expect(status.motion).toBe("gpu");
    expect(status.cpuReason).toBeUndefined();
  });

  it("CPU → GPU mid-gust writes the exact measured bytes back first", () => {
    const { baked, primitive, factory, deformer } = harness(false);
    deformer.apply(...frameAt(2));
    expect(deformer.status.displaced).toBe(true);
    const before = primitive.texture.uploads.length;
    const bindings = deformer.status.tileBindings;

    deformer.setGpu(factory);
    // One restoring upload, of the engine's own bytes, and the deformer holds nothing displaced.
    expect(primitive.texture.uploads.length).toBe(before + 1);
    expectMeasuredPose(primitive.texture.uploads.at(-1), baked);
    expect(deformer.status.displaced).toBe(false);
    expect(deformer.status.phase).toBe("waiting");

    // The next frames run on the shader, and the attribute texture is never written again.
    let status = deformer.apply(...frameAt(2.1));
    expect(status.motion).toBe("gpu");
    expect(primitive.vertexMotion).toBe(deformer.gpuMotion);
    const uniforms = buildDrawCommand(primitive);
    status = deformer.apply(...frameAt(2.2));
    expect(status.displaced).toBe(true);
    expect(uniforms.u_splatMotionActive?.()).toBe(1);
    expect(primitive.texture.uploads.length).toBe(before + 1);
    // Bindings are path-independent: the switch re-derived without binding a tile again.
    expect(status.tileBindings).toBe(bindings);
  });

  it("GPU → CPU mid-gust uninstalls the hook before the CPU path writes", () => {
    const { baked, primitive, factory, deformer } = harness(true);
    deformer.apply(...frameAt(0, STILL));
    const uniforms = buildDrawCommand(primitive);
    deformer.apply(...frameAt(2));
    expect(deformer.status.displaced).toBe(true);
    expect(uniforms.u_splatMotionActive?.()).toBe(1);

    deformer.setGpu(undefined);
    // Nothing written to the attribute texture (the GPU path never displaced it), the hook is
    // gone and its textures released: the engine's next draw is the measured pose.
    expect(primitive.texture.uploads).toHaveLength(0);
    expect(primitive.vertexMotion).toBeUndefined();
    expect(uniforms.u_splatMotionActive?.()).toBe(0);
    expect(factory.made.every((texture) => texture.destroyed)).toBe(true);
    expect(deformer.status.displaced).toBe(false);

    // The CPU path takes over, and calm restores the measured bytes exactly, as ever.
    let status = deformer.apply(...frameAt(2.1));
    expect(status.motion).toBe("cpu");
    expect(status.cpuReason).toBe("no-factory");
    expect(status.displaced).toBe(true);
    status = deformer.apply(...frameAt(2.2, STILL));
    expect(status.displaced).toBe(false);
    expectMeasuredPose(primitive.texture.uploads.at(-1), baked);
  });

  it("does not write old rows into a snapshot that rebuilt since the last frame", () => {
    const { primitive, factory, deformer } = harness(false);
    deformer.apply(...frameAt(2));
    const before = primitive.texture.uploads.length;
    // A rebuild: the engine packed a fresh texture from the measured positions.
    primitive._snapshot = { generation: primitive._snapshot.generation + 1 };
    deformer.setGpu(factory);
    expect(primitive.texture.uploads.length).toBe(before);
  });

  it("is a no-op for the path it is already on, and after destroy", () => {
    const { primitive, factory, deformer } = harness(true);
    deformer.apply(...frameAt(0, STILL));
    buildDrawCommand(primitive);
    deformer.apply(...frameAt(2));
    const hook = deformer.gpuMotion;
    deformer.setGpu(factory);
    expect(deformer.gpuMotion).toBe(hook);
    expect(deformer.status.displaced).toBe(true);
    deformer.destroy();
    deformer.setGpu(undefined);
    expect(deformer.status.phase).toBe("detached");
    expect(primitive.texture.uploads).toHaveLength(0);
  });
});
