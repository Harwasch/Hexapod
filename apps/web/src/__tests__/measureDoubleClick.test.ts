/**
 * A double-click finishing a distance (`cesium/MeasurementManager.ts`). The browser sends two
 * clicks before its dblclick; taken as two points, the first ended the distance and the second
 * started another on the same spot, so the audit found a "0 mm · ground 0 mm" measurement
 * beside every distance finished that way.
 */
import { Cartesian2, Cartesian3, EntityCollection } from "cesium";
import { describe, expect, it } from "vitest";

import { isRepeatClick, MeasurementManager } from "@/cesium/MeasurementManager";
import type { SceneEvents } from "@/cesium/types";
import { Emitter } from "@/lib/emitter";
import type { Measurement } from "@/state/measurements";

/** A manager over a flat stand-in for the scene: a pixel picks a point 1 m per pixel apart. */
function measuring(mode: "distance" | "height" | "area") {
  const events = new Emitter<SceneEvents>();
  const finished: Measurement[] = [];
  events.on("measurement", (m) => finished.push(m));
  const viewer = {
    scene: {
      requestRender: () => undefined,
      pickPositionSupported: true,
      pickPosition: (w: Cartesian2) =>
        Cartesian3.fromDegrees(-120.04 + w.x * 1.1e-5, 34.7 - w.y * 0.9e-5, 300),
      globe: { pick: () => undefined },
    },
    entities: new EntityCollection(),
    canvas: document.createElement("canvas"),
    camera: { getPickRay: () => undefined },
  };
  const manager = new MeasurementManager(viewer as never, events);
  manager.start(mode);
  const click = (x: number, y: number, at: number) =>
    (manager as unknown as { onClick(w: Cartesian2, at: number): void }).onClick(
      new Cartesian2(x, y),
      at,
    );
  return { manager, finished, click };
}

describe("isRepeatClick", () => {
  it("is a click soon after the last one, where it was", () => {
    const first = { x: 100, y: 100, at: 1000 };
    expect(isRepeatClick(first, { x: 102, y: 101, at: 1200 })).toBe(true);
    expect(isRepeatClick(first, { x: 100, y: 100, at: 1700 })).toBe(false);
    expect(isRepeatClick(first, { x: 140, y: 100, at: 1100 })).toBe(false);
    expect(isRepeatClick(null, first)).toBe(false);
  });
});

describe("finishing a distance with a double-click", () => {
  it("adds one measurement, never a second one of 0 mm", () => {
    const { manager, finished, click } = measuring("distance");
    click(100, 100, 0);
    // The double-click at the end: two clicks, a moment apart, on the same pixel.
    click(400, 300, 2000);
    click(400, 300, 2120);
    expect(finished).toHaveLength(1);
    expect(finished[0]?.distance3dM).toBeGreaterThan(1);
    // Nothing is left half-started on the end point, waiting to become a 0 mm distance.
    expect((manager as unknown as { draft: unknown }).draft).toBeNull();
    // The next click starts a new distance there, and the one after ends it.
    click(420, 300, 4000);
    click(600, 300, 5000);
    expect(finished).toHaveLength(2);
    expect(finished.every((m) => (m.distance3dM ?? 0) > 1)).toBe(true);
    manager.stop();
  });

  it("does not end a distance on its own first point", () => {
    const { finished, click } = measuring("distance");
    // A double-click to start: the repeat is ignored, and a slow second click on the same
    // spot is the same point, not a 0 mm distance.
    click(100, 100, 0);
    click(100, 100, 100);
    click(100, 100, 3000);
    expect(finished).toHaveLength(0);
    click(300, 100, 4000);
    expect(finished).toHaveLength(1);
    expect(finished[0]?.distance3dM).toBeGreaterThan(1);
  });

  it("keeps a height measurement from collapsing to nothing the same way", () => {
    const { finished, click } = measuring("height");
    click(100, 100, 0);
    click(100, 100, 2000);
    expect(finished).toHaveLength(0);
  });

  it("still takes quick clicks in different places as separate area vertices", () => {
    const { manager, click } = measuring("area");
    click(100, 100, 0);
    click(300, 100, 150);
    click(300, 300, 300);
    // The double-click on the last vertex adds it once.
    click(100, 300, 450);
    click(100, 300, 560);
    const draft = (manager as unknown as { draft: { points: unknown[] } | null }).draft;
    expect(draft?.points).toHaveLength(4);
  });
});
