import { describe, expect, it } from "vitest";

import type { LiveCameras, LiveState } from "@twin/contracts";

import {
  decodeCameras,
  decodePoints,
  describeLive,
  medianCentre,
  stageName,
  upOf,
} from "@/view/liveData";

function base64(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

/** Two cameras and two points, packed the way tools/pipeline/live.py packs them. */
function payload(): LiveCameras {
  const cameras = new DataView(new ArrayBuffer(24));
  // Camera 0 at origin + (0.5, 0, 0) * scale, looking +z, up -y.
  cameras.setInt16(0, 16384, true);
  cameras.setInt8(8, 127);
  cameras.setInt8(10, -127);
  // Camera 1 at origin + (0, -1, 0) * scale, looking +x, up +z.
  cameras.setInt16(14, -32767, true);
  cameras.setInt8(18, 127);
  cameras.setInt8(23, 127);
  const points = new DataView(new ArrayBuffer(18));
  points.setInt16(4, 32767, true);
  points.setUint8(6, 255);
  points.setUint8(7, 0);
  points.setUint8(8, 51);
  points.setInt16(9, -32767, true);
  return {
    stageId: "pose",
    seq: 3,
    final: false,
    registered: 2,
    frames: 40,
    origin: [1, 2, 3],
    scale: 10,
    up: [0, -2, 0],
    aspect: 1.5,
    cameraCount: 2,
    cameras: base64(new Uint8Array(cameras.buffer)),
    pointCount: 2,
    pointsTotal: 900,
    points: base64(new Uint8Array(points.buffer)),
  };
}

function state(overrides: Partial<LiveState> = {}): LiveState {
  return {
    jobId: "j",
    captureId: "c",
    captureName: "Garden tree",
    recipe: "photo-reconstruct",
    status: "in-progress",
    error: null,
    siteId: null,
    stage: null,
    stepsDone: 0,
    stepsStarted: 0,
    cameras: null,
    splat: null,
    ...overrides,
  };
}

describe("the live cameras payload", () => {
  it("decodes centres, directions and up against the origin and scale", () => {
    const decoded = decodeCameras(payload());
    expect(decoded.count).toBe(2);
    expect(decoded.centres[0]).toBeCloseTo(1 + 5, 3);
    expect(Array.from(decoded.centres.slice(3, 6))).toEqual([1, 2 - 10, 3]);
    expect(Array.from(decoded.forward.slice(0, 3))).toEqual([0, 0, 1]);
    expect(Array.from(decoded.up.slice(0, 3))).toEqual([0, -1, 0]);
    expect(Array.from(decoded.forward.slice(3, 6))).toEqual([1, 0, 0]);
    expect(Array.from(decoded.up.slice(3, 6))).toEqual([0, 0, 1]);
  });

  it("decodes the points with their colours", () => {
    const points = decodePoints(payload());
    expect(points.count).toBe(2);
    expect(Array.from(points.positions.slice(0, 3))).toEqual([1, 2, 13]);
    expect(points.colors[0]).toBe(1);
    expect(points.colors[2]).toBeCloseTo(0.2, 5);
    expect(points.positions[3]).toBe(-9);
  });

  it("never reads past the data it was given", () => {
    const short = { ...payload(), cameraCount: 50, pointCount: 50 };
    expect(decodeCameras(short).count).toBe(2);
    expect(decodePoints(short).count).toBe(2);
  });

  it("finds up from the payload, else falls back to COLMAP's -y", () => {
    expect(upOf({ cameras: payload(), splat: null })).toEqual([0, -1, 0]);
    expect(upOf({ cameras: null, splat: null })).toEqual([0, -1, 0]);
    const splat = {
      stageId: "train",
      step: 3000,
      total: 30000,
      count: 10,
      of: 20,
      bytes: 100,
      up: [0, 0, 3],
      name: "splat_003000_1.spz",
      url: null,
    };
    expect(upOf({ cameras: null, splat })).toEqual([0, 0, 1]);
  });

  it("centres on the median, not the mean a stray point would drag", () => {
    const positions = new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2, 1000, 1000, 1000, 1, 1, 1]);
    expect(medianCentre(positions)).toEqual([1, 1, 1]);
    expect(medianCentre(new Float32Array())).toBeNull();
  });
});

describe("the live status line", () => {
  it("says what the pose solve has placed", () => {
    const described = describeLive(
      state({
        stage: {
          stageId: "pose",
          impl: "colmap",
          status: "in-progress",
          ordinal: 1,
          startedAt: null,
          progress: null,
        },
        cameras: payload(),
      }),
    );
    expect(described.headline).toBe("Solving camera positions…");
    expect(described.detail).toBe("2 of 40 frames placed");
    expect(described.fraction).toBeNull();
    expect(described.ended).toBe(false);
  });

  it("gives training's step, percentage, time left and the splat on screen", () => {
    const described = describeLive(
      state({
        stage: {
          stageId: "train",
          impl: "gsplat",
          status: "in-progress",
          ordinal: 3,
          startedAt: null,
          progress: { done: 7500, total: 30000, elapsedS: 600, remainingS: 1800 },
        },
        splat: {
          stageId: "train",
          step: 3000,
          total: 30000,
          count: 100000,
          of: 400000,
          bytes: 1,
          up: null,
          name: "splat_003000_1.spz",
          url: null,
        },
      }),
    );
    expect(described.headline).toBe("Training the splat…");
    expect(described.detail).toBe(
      "Step 7,500 of 30,000 · 25% · about 30 min left · showing step 3,000 (on its way)",
    );
    expect(described.fraction).toBe(0.25);
  });

  it("ends with the scan, or with why it stopped", () => {
    expect(describeLive(state({ status: "complete", siteId: "s" }))).toMatchObject({
      headline: "Finished",
      ended: true,
    });
    expect(describeLive(state({ status: "error", error: "no frames" }))).toMatchObject({
      headline: "Stopped",
      detail: "no frames",
      ended: true,
    });
    expect(describeLive(state()).headline).toBe("Queued");
    expect(stageName("somethingNew")).toBe("somethingNew");
  });
});
