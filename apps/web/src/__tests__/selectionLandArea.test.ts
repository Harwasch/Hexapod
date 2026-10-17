/**
 * Which picks a ground layer taking taps (the fog, Show once) may answer for: the polygons of a
 * Land's inventory and research layers, never their lines or points, nor other entities.
 */

import { Cartesian3, Entity, PolygonHierarchy } from "cesium";
import { expect, it } from "vitest";

import { isLandAreaPick } from "@/cesium/SelectionManager";

const ring = Cartesian3.fromDegreesArray([-120.05, 34.69, -120.03, 34.69, -120.03, 34.7]);

it("claims a land area's polygon, not its lines, points or other entities", () => {
  const area = new Entity({
    id: "land-context:inventory/oak#0",
    polygon: { hierarchy: new PolygonHierarchy(ring) },
  });
  const fence = new Entity({
    id: "land-context:inventory/fence",
    polyline: { positions: ring },
  });
  const zone = new Entity({ id: "zone:Z-14", polygon: { hierarchy: new PolygonHierarchy(ring) } });
  expect(isLandAreaPick({ id: area })).toBe(true);
  expect(isLandAreaPick({ id: fence })).toBe(false);
  expect(isLandAreaPick({ id: zone })).toBe(false);
  expect(isLandAreaPick(undefined)).toBe(false);
  expect(isLandAreaPick({ id: "land-boundary:fill:0" })).toBe(false);
});
