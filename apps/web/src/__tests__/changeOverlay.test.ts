/**
 * The change map's outline over every renderer (cesium/changeOverlay.ts): an SVG laid over
 * the globe after any splat renderer's canvas, so splats cannot cover it, never in the way of a
 * pointer; each class outlined with a light fill, low confidence hatched, not comparable
 * dashed in grey; projected through the camera, a polygon with a point behind it left out;
 * redrawn only when the view changed, at most every so often; gone when destroyed.
 */
import { Cartesian3, Event, type Scene } from "cesium";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ChangeArea, ChangeMap } from "@twin/contracts";

import { ChangeOverlay, OVERLAY_MIN_INTERVAL_MS } from "@/cesium/changeOverlay";

const LON = -123.88;
const LAT = 46.13;

function fakeScene() {
  const host = document.createElement("div");
  const canvas = document.createElement("canvas");
  Object.defineProperty(canvas, "clientWidth", { value: 800 });
  Object.defineProperty(canvas, "clientHeight", { value: 600 });
  host.appendChild(canvas);
  // A splat renderer's canvas, laid right above the globe's (ScanRendererHost).
  const splats = document.createElement("canvas");
  canvas.insertAdjacentElement("afterend", splats);
  document.body.appendChild(host);
  const scene = {
    canvas,
    postRender: new Event(),
    camera: { viewMatrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] },
    requestRender: vi.fn(),
    isDestroyed: () => false,
  };
  return { scene: scene as unknown as Scene, raw: scene, host, splats };
}

/** Window x from longitude, y from latitude; undefined east of `behindEast` (behind the eye). */
function projector(behindEast = Infinity) {
  return vi.fn((p: Cartesian3) => {
    const lonLat = Cartesian3.clone(p);
    const longitude = (Math.atan2(lonLat.y, lonLat.x) * 180) / Math.PI;
    const latitude = (Math.atan2(lonLat.z, Math.hypot(lonLat.x, lonLat.y)) * 180) / Math.PI;
    if (longitude > behindEast) return undefined;
    return { x: (longitude - LON) * 1e5, y: (LAT + 0.001 - latitude) * 1e5 };
  });
}

const square = (x: number, d = 1e-5): number[][][] => [
  [
    [LON + x * d, LAT],
    [LON + (x + 1) * d, LAT],
    [LON + (x + 1) * d, LAT + d],
    [LON + x * d, LAT + d],
    [LON + x * d, LAT],
  ],
];

const area = (cls: ChangeArea["class"], lowConfidence: boolean, x: number) =>
  ({
    class: cls,
    footprint: { type: "MultiPolygon", coordinates: [square(x)] },
    cells: 1,
    lowConfidence,
  }) as unknown as ChangeArea;

const map = (areas: ChangeArea[]) => ({ areas }) as unknown as ChangeMap;

const paths = (host: HTMLElement) =>
  [...host.querySelectorAll("svg path")].map((p) => ({
    cls: p.getAttribute("data-class"),
    low: p.getAttribute("data-low-confidence"),
    d: p.getAttribute("d") ?? "",
    fill: p.getAttribute("fill"),
    stroke: p.getAttribute("stroke"),
    dash: p.getAttribute("stroke-dasharray"),
  }));

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
  document.body.innerHTML = "";
});

describe("the change overlay", () => {
  it("lies over the globe and above any splat renderer's canvas, out of the pointer's way", () => {
    const { scene, host, splats } = fakeScene();
    new ChangeOverlay(scene, { project: projector() });
    const svg = host.querySelector("svg");
    expect(svg).not.toBeNull();
    expect(svg?.style.position).toBe("absolute");
    expect(svg?.style.pointerEvents).toBe("none");
    // After the renderer's canvas, so drawn over it.
    expect(
      splats.compareDocumentPosition(svg as Node) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    // A renderer canvas made later goes right after the globe's: still under the overlay.
    const later = document.createElement("canvas");
    scene.canvas.insertAdjacentElement("afterend", later);
    expect(
      later.compareDocumentPosition(svg as Node) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("outlines each class with a light fill, hatches low confidence, dashes not comparable", () => {
    const { scene, host } = fakeScene();
    const overlay = new ChangeOverlay(scene, { project: projector() });
    overlay.show(
      map([
        area("raised", false, 0),
        area("raised", true, 2),
        area("lowered", false, 4),
        area("insufficient", false, 6),
      ]),
    );
    const drawn = paths(host);
    expect(drawn.map((p) => [p.cls, p.low])).toEqual([
      ["raised", "false"],
      ["raised", "true"],
      ["lowered", "false"],
      ["insufficient", "false"],
    ]);
    expect(drawn.every((p) => /^M[^Z]+Z/.test(p.d))).toBe(true);
    const [raised, raisedLow, lowered, insufficient] = drawn;
    expect(raised?.stroke).toBe("#f08c3a");
    expect(lowered?.stroke).toBe("#3a9cf0");
    expect(raised?.fill).toBe("#f08c3a");
    expect(raisedLow?.fill).toMatch(/^url\(#change-hatch-raised-/);
    expect(host.querySelector(`pattern${raisedLow?.fill?.slice(4, -1) ?? ""}`)).not.toBeNull();
    expect(insufficient?.dash).toBeTruthy();
    expect(insufficient?.stroke).toBe("#a3a7ad");
    expect(raised?.dash).toBeNull();
  });

  it("leaves out a polygon with a point behind the camera", () => {
    const { scene, host } = fakeScene();
    const overlay = new ChangeOverlay(scene, { project: projector(LON + 3.5e-5) });
    overlay.show(
      map([area("raised", false, 0), area("raised", false, 2), area("raised", false, 4)]),
    );
    // Squares 0 and 2 lie west of the cut; square 2 reaches 3e-5 exactly, square 4 is past it.
    const d = paths(host)[0]?.d ?? "";
    expect(d.match(/M/g)).toHaveLength(2);
  });

  it("redraws only when the view changed, at most every so often", () => {
    const { scene, raw } = fakeScene();
    const project = projector();
    const overlay = new ChangeOverlay(scene, { project });
    overlay.show(map([area("raised", false, 0)]));
    const after = project.mock.calls.length;
    expect(after).toBeGreaterThan(0);
    // A frame with the same view: nothing projected again.
    vi.advanceTimersByTime(OVERLAY_MIN_INTERVAL_MS + 1);
    raw.postRender.raiseEvent();
    expect(project.mock.calls.length).toBe(after);
    // The camera moves: drawn again.
    raw.camera.viewMatrix = [...raw.camera.viewMatrix.slice(0, 12), 5, 0, 0, 1];
    raw.postRender.raiseEvent();
    const moved = project.mock.calls.length;
    expect(moved).toBeGreaterThan(after);
    // It moves again at once: held until the interval has passed, then drawn once.
    raw.camera.viewMatrix = [...raw.camera.viewMatrix.slice(0, 12), 6, 0, 0, 1];
    raw.postRender.raiseEvent();
    raw.camera.viewMatrix = [...raw.camera.viewMatrix.slice(0, 12), 7, 0, 0, 1];
    raw.postRender.raiseEvent();
    expect(project.mock.calls.length).toBe(moved);
    vi.advanceTimersByTime(OVERLAY_MIN_INTERVAL_MS);
    expect(project.mock.calls.length).toBe(moved + (moved - after));
  });

  it("clears to nothing, and is gone with its listener when destroyed", () => {
    const { scene, raw, host } = fakeScene();
    const project = projector();
    const overlay = new ChangeOverlay(scene, { project });
    overlay.show(map([area("lowered", false, 0)]));
    overlay.clear();
    expect(paths(host)).toEqual([]);
    overlay.show(map([area("lowered", false, 0)]));
    overlay.destroy();
    expect(host.querySelector("svg")).toBeNull();
    const calls = project.mock.calls.length;
    raw.camera.viewMatrix = [...raw.camera.viewMatrix.slice(0, 12), 9, 0, 0, 1];
    raw.postRender.raiseEvent();
    vi.advanceTimersByTime(OVERLAY_MIN_INTERVAL_MS * 2);
    expect(project.mock.calls.length).toBe(calls);
    expect(raw.postRender.numberOfListeners).toBe(0);
    overlay.destroy();
  });
});
