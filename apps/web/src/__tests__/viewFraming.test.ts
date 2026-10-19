/**
 * Framing a place in the part of the map a panel leaves open (`lib/viewFraming.ts`): on a phone
 * the Land sheet covers the lower half of the screen, so a boundary under review is framed
 * above it; on a desktop the panel is on the left and the land goes to its right.
 */
import { describe, expect, it } from "vitest";

import {
  NO_INSETS,
  panelInsets,
  pointsInOpenView,
  rectangleInView,
  topDownDistance,
  type ViewInsets,
} from "@/lib/viewFraming";

const PHONE = { left: 0, top: 0, right: 390, bottom: 844 };
const DESKTOP = { left: 0, top: 0, right: 1440, bottom: 900 };
const RANCH = { west: -120.06, south: 34.68, east: -120.02, north: 34.72 };

/** Where a lon/lat lands in a view looking straight down at `r`, in CSS pixels. */
function project(
  r: { west: number; south: number; east: number; north: number },
  view: { width: number; height: number },
  lon: number,
  lat: number,
) {
  return {
    x: ((lon - r.west) / (r.east - r.west)) * view.width,
    y: ((r.north - lat) / (r.north - r.south)) * view.height,
  };
}

function corners(b: typeof RANCH) {
  return [
    [b.west, b.north],
    [b.east, b.north],
    [b.east, b.south],
    [b.west, b.south],
  ] as const;
}

describe("panelInsets", () => {
  it("reads a wide panel low on the screen as a bottom sheet, and covers from its top down", () => {
    const sheet = { left: 10, top: 200, right: 380, bottom: 690 };
    expect(panelInsets(sheet, PHONE, 70)).toEqual({ ...NO_INSETS, top: 70, bottom: 644 });
  });

  it("reads a panel on the left as covering the left, up to its right edge", () => {
    const panel = { left: 90, top: 90, right: 560, bottom: 820 };
    expect(panelInsets(panel, DESKTOP, 80)).toEqual({ ...NO_INSETS, top: 80, left: 560 });
    const right = { left: 1000, top: 90, right: 1430, bottom: 820 };
    expect(panelInsets(right, DESKTOP).right).toBe(440);
  });

  it("covers only the top row when no panel is open", () => {
    expect(panelInsets(null, PHONE, 64)).toEqual({ ...NO_INSETS, top: 64 });
  });
});

describe("rectangleInView", () => {
  it("frames the land above a phone's sheet, inside the open strip, with a margin", () => {
    const view = { width: 390, height: 844 };
    const insets: ViewInsets = { ...NO_INSETS, top: 70, bottom: 644 };
    const r = rectangleInView(RANCH, view, insets);
    for (const [lon, lat] of corners(RANCH)) {
      const p = project(r, view, lon, lat);
      expect(p.y).toBeGreaterThanOrEqual(70);
      expect(p.y).toBeLessThanOrEqual(844 - 644);
      expect(p.x).toBeGreaterThanOrEqual(0);
      expect(p.x).toBeLessThanOrEqual(390);
    }
    // Centred in the open strip, not in the whole screen.
    const centre = project(r, view, -120.04, 34.7);
    expect(centre.y).toBeCloseTo(70 + (844 - 644 - 70) / 2, 0);
  });

  it("frames the land to the right of a desktop panel", () => {
    const view = { width: 1440, height: 900 };
    const insets: ViewInsets = { ...NO_INSETS, top: 80, left: 560 };
    const r = rectangleInView(RANCH, view, insets);
    for (const [lon, lat] of corners(RANCH)) {
      const p = project(r, view, lon, lat);
      expect(p.x).toBeGreaterThanOrEqual(560);
      expect(p.y).toBeGreaterThanOrEqual(80);
    }
    expect(project(r, view, -120.04, 34.7).x).toBeCloseTo(560 + (1440 - 560) / 2, 0);
  });

  it("fills the view around the land when nothing covers it", () => {
    const view = { width: 1000, height: 1000 };
    const r = rectangleInView(RANCH, view, NO_INSETS, 0);
    expect((r.west + r.east) / 2).toBeCloseTo(-120.04, 6);
    expect((r.south + r.north) / 2).toBeCloseTo(34.7, 6);
    // The taller side fits exactly with no margin (degrees of longitude are shorter here).
    expect(r.north - r.south).toBeCloseTo(0.04, 6);
  });

  it("still frames into a sliver when a panel covers nearly everything", () => {
    const view = { width: 390, height: 844 };
    const r = rectangleInView(RANCH, view, { ...NO_INSETS, bottom: 840 });
    expect(Number.isFinite(r.west) && Number.isFinite(r.north)).toBe(true);
    expect(r.north).toBeGreaterThan(RANCH.north);
  });
});

describe("topDownDistance", () => {
  it("puts the camera where the rectangle fills the view, measured from the ground", () => {
    const view = { width: 400, height: 800 };
    const r = rectangleInView(RANCH, view, NO_INSETS, 0);
    const fovy = Math.PI / 3;
    const d = topDownDistance(r, view, fovy);
    // The view's half-height at that distance is the rectangle's half-height in metres.
    expect(d * Math.tan(fovy / 2)).toBeCloseTo(((r.north - r.south) / 2) * 111_320, 0);
    // A wider window needs no more height for the same (taller) rectangle's height.
    expect(topDownDistance(r, { width: 1600, height: 800 }, fovy)).toBeCloseTo(d, 0);
  });
});

describe("pointsInOpenView", () => {
  const view = { width: 390, height: 844 };
  const insets: ViewInsets = { ...NO_INSETS, top: 70, bottom: 644 };

  it("is true only when every point is clear of the panel and on screen", () => {
    expect(pointsInOpenView([{ x: 100, y: 120 }], view, insets)).toBe(true);
    expect(
      pointsInOpenView(
        [
          { x: 100, y: 120 },
          { x: 100, y: 400 },
        ],
        view,
        insets,
      ),
    ).toBe(false);
    expect(pointsInOpenView([{ x: -5, y: 120 }], view, insets)).toBe(false);
    expect(pointsInOpenView([{ x: 100, y: 20 }], view, insets)).toBe(false);
  });

  it("counts a point behind the globe as hidden", () => {
    expect(pointsInOpenView([{ x: 100, y: 120 }, null], view, insets)).toBe(false);
  });
});
