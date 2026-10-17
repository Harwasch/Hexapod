/**
 * The Time Lantern's geometry and input (cesium/timeLanternMath.ts): the lens in the drawing
 * buffer, its size, the ground window under it, the wheel as steps through dates, and which
 * record covers a point.
 */
import {
  Cartesian2,
  Cartesian3,
  Ellipsoid,
  IntersectionTests,
  Math as CesiumMath,
  Matrix4,
  Plane,
  Ray,
  Transforms,
} from "cesium";
import { describe, expect, it } from "vitest";

import type { LanternPart } from "@/cesium/lanternTypes";
import {
  coverageAt,
  groundWindow,
  lensRadii,
  toDrawingBuffer,
  WHEEL_IDLE,
  WINDOW_REACH,
  wheelSteps,
  type PickRayCamera,
  type WheelState,
} from "@/cesium/timeLanternMath";

const ORIGIN = Cartesian3.fromDegrees(-120.04, 34.698, 300);
const UP = Ellipsoid.WGS84.geodeticSurfaceNormal(ORIGIN, new Cartesian3());

/**
 * A pinhole camera at `height` above ORIGIN's ground, looking down at `pitch` degrees towards
 * the north: rays through CSS pixels of a `width` x `height` canvas, as Camera.getPickRay.
 */
function camera(heightM: number, pitchDeg: number, size = { width: 1000, height: 800 }) {
  const frame = Transforms.eastNorthUpToFixedFrame(ORIGIN);
  const pitch = CesiumMath.toRadians(pitchDeg);
  // Looking north (local +y), tipped down by -pitch.
  const forwardLocal = new Cartesian3(0, Math.cos(pitch), Math.sin(pitch));
  const rightLocal = new Cartesian3(1, 0, 0);
  const upLocal = Cartesian3.cross(rightLocal, forwardLocal, new Cartesian3());
  const position = Matrix4.multiplyByPoint(frame, new Cartesian3(0, 0, heightM), new Cartesian3());
  const forward = Matrix4.multiplyByPointAsVector(frame, forwardLocal, new Cartesian3());
  const right = Matrix4.multiplyByPointAsVector(frame, rightLocal, new Cartesian3());
  const up = Matrix4.multiplyByPointAsVector(frame, upLocal, new Cartesian3());
  const fovy = CesiumMath.toRadians(60);
  const tanY = Math.tan(fovy / 2);
  const tanX = tanY * (size.width / size.height);
  const pickRay: PickRayCamera = {
    getPickRay(window: Cartesian2, result = new Ray()) {
      const nx = (2 * window.x) / size.width - 1;
      const ny = 1 - (2 * window.y) / size.height;
      const direction = Cartesian3.clone(forward);
      Cartesian3.add(
        direction,
        Cartesian3.multiplyByScalar(right, nx * tanX, new Cartesian3()),
        direction,
      );
      Cartesian3.add(
        direction,
        Cartesian3.multiplyByScalar(up, ny * tanY, new Cartesian3()),
        direction,
      );
      Cartesian3.normalize(direction, direction);
      result.origin = Cartesian3.clone(position, result.origin);
      result.direction = Cartesian3.clone(direction, result.direction);
      return result;
    },
  };
  return { pickRay, position, metresPerPixel: (2 * heightM * tanY) / size.height };
}

const plane = { point: ORIGIN, normal: UP };

/** A ring's extent along the local east and north axes (m). */
function extent(ring: Cartesian3[]) {
  const toLocal = Matrix4.inverseTransformation(
    Transforms.eastNorthUpToFixedFrame(ORIGIN),
    new Matrix4(),
  );
  const local = ring.map((p) => Matrix4.multiplyByPoint(toLocal, p, new Cartesian3()));
  const xs = local.map((p) => p.x);
  const ys = local.map((p) => p.y);
  return {
    width: Math.max(...xs) - Math.min(...xs),
    depth: Math.max(...ys) - Math.min(...ys),
  };
}

describe("the lens on screen", () => {
  it("is in drawing-buffer pixels with the GL origin at the bottom", () => {
    // Device pixel ratio 2 and a resolution scale of 0.75: 1.5 buffer pixels per CSS pixel.
    const canvas = {
      clientWidth: 1000,
      clientHeight: 800,
      drawingBufferWidth: 1500,
      drawingBufferHeight: 1200,
    };
    const point = toDrawingBuffer(100, 200, canvas);
    expect(point.x).toBe(150);
    expect(point.y).toBe(1200 - 300);
    expect(point.scale).toBe(1.5);
  });

  it("is a sixth of the shorter side, between 110 and 200 px", () => {
    expect(lensRadii(1440, 900).outer).toBeCloseTo(144);
    expect(lensRadii(390, 600).outer).toBe(110);
    expect(lensRadii(3840, 2160).outer).toBe(200);
    expect(lensRadii(1440, 900).inner).toBeCloseTo(0.72 * 144);
  });
});

describe("the ground window", () => {
  it("is a circle at nadir, a lens's radius times the margin across", () => {
    const { pickRay, metresPerPixel } = camera(500, -90);
    const ring = groundWindow(pickRay, { x: 500, y: 400 }, 100, plane, 32, 1.35);
    expect(ring).toHaveLength(32);
    const { width, depth } = extent(ring ?? []);
    const expected = 2 * 100 * metresPerPixel * 1.35;
    expect(width / expected).toBeGreaterThan(0.98);
    expect(width / expected).toBeLessThan(1.02);
    expect(depth / expected).toBeGreaterThan(0.98);
    expect(depth / expected).toBeLessThan(1.02);
  });

  it("stretches away from the camera on an oblique view", () => {
    const { pickRay } = camera(500, -35);
    const { width, depth } = extent(groundWindow(pickRay, { x: 500, y: 500 }, 100, plane) ?? []);
    expect(depth).toBeGreaterThan(width * 1.3);
  });

  it("stops rays over the horizon at a reach of five cursor distances", () => {
    const { pickRay, position } = camera(500, -8);
    const cursor = { x: 500, y: 520 };
    const centreRay = pickRay.getPickRay(new Cartesian2(cursor.x, cursor.y));
    if (!centreRay) throw new Error("no ray");
    const hit = IntersectionTests.rayPlane(centreRay, Plane.fromPointNormal(ORIGIN, UP));
    if (!hit) throw new Error("cursor over the sky");
    const reach = WINDOW_REACH * Cartesian3.distance(position, hit);
    const ring = groundWindow(pickRay, cursor, 200, plane) ?? [];
    expect(ring).toHaveLength(32);
    const distances = ring.map((p) => Cartesian3.distance(position, p));
    // The rays over the horizon, and those meeting the ground too far off, stop at the reach.
    const clamped = distances.filter((d) => Math.abs(d - reach) < 0.5);
    expect(clamped.length).toBeGreaterThan(1);
    expect(Math.max(...distances)).toBeLessThan(reach + 0.5);
  });

  it("is nothing when the cursor is over the sky", () => {
    const { pickRay } = camera(500, -5);
    expect(groundWindow(pickRay, { x: 500, y: 10 }, 100, plane)).toBeNull();
  });
});

describe("the wheel through time", () => {
  const run = (events: [number, number, number][], state: WheelState = WHEEL_IDLE) => {
    let steps = 0;
    for (const [deltaY, deltaMode, at] of events) {
      const result = wheelSteps(state, deltaY, deltaMode, at);
      state = result.state;
      steps += result.steps;
    }
    return steps;
  };

  it("steps once per mouse notch, older when the wheel turns down", () => {
    expect(run([[3, 1, 1000]])).toBe(1);
    expect(run([[-3, 1, 1000]])).toBe(-1);
    expect(run([[100, 0, 1000]])).toBe(1);
  });

  it("adds up a trackpad's pixels, a step per 60", () => {
    const events: [number, number, number][] = Array.from({ length: 10 }, (_, i) => [
      12,
      0,
      1000 + i * 30,
    ]);
    expect(run(events)).toBe(2);
  });

  it("does not step again within the cooldown", () => {
    expect(
      run([
        [3, 1, 1000],
        [3, 1, 1050],
      ]),
    ).toBe(1);
    expect(
      run([
        [3, 1, 1000],
        [3, 1, 1200],
      ]),
    ).toBe(2);
  });

  it("starts again when the trackpad turns back", () => {
    expect(
      run([
        [40, 0, 1000],
        [-40, 0, 1010],
        [-25, 0, 1020],
      ]),
    ).toBe(-1);
  });
});

describe("coverage", () => {
  const part: LanternPart = {
    footprint: {
      type: "Polygon",
      coordinates: [
        [
          [-120.06, 34.68],
          [-120.02, 34.68],
          [-120.02, 34.72],
          [-120.06, 34.72],
          [-120.06, 34.68],
        ],
      ],
    },
    dateLabel: "15 Jun 2016",
    accuracyLabel: "x",
    recordUrl: "https://example.com",
  };

  it("finds the record under a point, and none outside every record", () => {
    expect(coverageAt([part], -120.04, 34.698)?.dateLabel).toBe("15 Jun 2016");
    expect(coverageAt([part], -120.1, 34.698)).toBeNull();
  });
});
