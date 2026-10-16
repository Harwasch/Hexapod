import { describe, expect, it } from "vitest";

import { destination, haversineDistance } from "@twin/geo";

import {
  eyeOf,
  frameGround,
  framingContains,
  hudInsets,
  HUD_INSETS,
  PHONE_HUD_INSETS,
} from "@/missions/framing";
import type { LonLat } from "@/missions/types";

const CENTER: LonLat = { longitude: -120.04, latitude: 34.7 };
const NONE = { top: 0, right: 0, bottom: 0, left: 0 };

/** The corners of a square `sideM` across, centred on CENTER. */
function square(sideM: number): LonLat[] {
  const half = (sideM / 2) * Math.SQRT2;
  return [45, 135, 225, 315].map((b) => destination(CENTER, b, half));
}

describe("frameGround", () => {
  it("fits a 1 km square seen straight down at half its width over tan 30°", () => {
    const framing = frameGround(square(1000), {
      headingDeg: 0,
      pitchDeg: -90,
      aspect: 1,
      insets: NONE,
      margin: 1,
    });
    expect(framing.rangeM).toBeCloseTo(500 / Math.tan(Math.PI / 6), -1);
    expect(haversineDistance(framing.target, CENTER)).toBeLessThan(5);
    const margined = frameGround(square(1000), {
      headingDeg: 0,
      pitchDeg: -90,
      aspect: 1,
      insets: NONE,
    });
    expect(margined.rangeM).toBeCloseTo(1.08 * framing.rangeM, 0);
  });

  it("puts every point inside the HUD-safe box, centred in it", () => {
    const points = [...square(2000), destination(CENTER, 10, 1600), destination(CENTER, 200, 900)];
    const options = { headingDeg: 15, pitchDeg: -38, aspect: 1440 / 900 };
    const framing = frameGround(points, options);
    const fit = framingContains(framing, points, options);
    expect(fit.inside).toBe(true);
    // Without the margin the outermost point touches the box: the fit is tight.
    const tight = framingContains({ ...framing, rangeM: framing.rangeM / 1.08 }, points, options);
    expect(tight.worst.excess).toBeGreaterThan(-0.01);
    expect(tight.worst.excess).toBeLessThan(0.01);
    // Centred: the points' extent is as far from the box's left edge as from its right.
    const xs = [];
    const ys = [];
    for (const point of points) {
      const one = framingContains({ ...framing, rangeM: framing.rangeM / 1.08 }, [point], {
        ...options,
        insets: NONE,
      });
      xs.push(one.worst.x);
      ys.push(one.worst.y);
    }
    const boxX = (-1 + 2 * HUD_INSETS.left + (1 - 2 * HUD_INSETS.right)) / 2;
    const boxY = (1 - 2 * HUD_INSETS.top + (-1 + 2 * HUD_INSETS.bottom)) / 2;
    expect(Math.abs((Math.min(...xs) + Math.max(...xs)) / 2 - boxX)).toBeLessThan(0.02);
    expect(Math.abs((Math.min(...ys) + Math.max(...ys)) / 2 - boxY)).toBeLessThan(0.02);
  });

  it("uses the narrower horizontal field of view in a portrait window", () => {
    // An east-west line seen from above: its width is what the range has to fit.
    const line = [destination(CENTER, 90, 1000), destination(CENTER, 270, 1000)];
    const down = { headingDeg: 0, pitchDeg: -90, insets: NONE, margin: 1 };
    const square = frameGround(line, { ...down, aspect: 1 });
    const portrait = frameGround(line, { ...down, aspect: 0.5 });
    const landscape = frameGround(line, { ...down, aspect: 2 });
    expect(portrait.rangeM / square.rangeM).toBeCloseTo(2, 2);
    // Landscape: the field of view spans the width, as in a square window.
    expect(landscape.rangeM / square.rangeM).toBeCloseTo(1, 2);
  });

  it("refuses to frame nothing", () => {
    expect(() => frameGround([], { headingDeg: 0, pitchDeg: -45, aspect: 1 })).toThrow();
  });
});

describe("eyeOf", () => {
  it("stands range back along the heading and up by the pitch", () => {
    const eye = eyeOf({ target: CENTER, headingDeg: 0, pitchDeg: -45, rangeM: 1000 }, 300);
    // Looking north, so the eye is south of the target.
    expect(haversineDistance(eye, CENTER)).toBeCloseTo(707.1, 0);
    expect(eye.latitude).toBeLessThan(CENTER.latitude);
    expect(eye.longitude).toBeCloseTo(CENTER.longitude, 9);
    expect(eye.height).toBeCloseTo(300 + 707.1, 0);
    expect([eye.heading, eye.pitch]).toEqual([0, -45]);
  });
});

describe("hudInsets", () => {
  it("is the phone's at mission.css's breakpoint and the desktop's above it", () => {
    expect(hudInsets(390)).toBe(PHONE_HUD_INSETS);
    expect(hudInsets(640)).toBe(PHONE_HUD_INSETS);
    expect(hudInsets(1440)).toBe(HUD_INSETS);
  });
});
