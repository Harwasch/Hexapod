import { describe, expect, it } from "vitest";

import { resortDistance, RESORT_MAX_M, RESORT_MIN_M } from "@/cesium/splatSorter";
import { isInterfaceTarget } from "@/cesium/uiActivity";

describe("the interface comes first", () => {
  it("counts presses on controls, not on the globe or the page", () => {
    document.body.innerHTML = `<canvas id="globe"></canvas>
      <button id="b"><span id="inside">Layers</span></button>
      <div role="listbox"><div role="option" id="o">A</div></div>
      <p id="text">words</p>`;
    const canvas = document.getElementById("globe") as Element;
    for (const id of ["b", "inside", "o"])
      expect(isInterfaceTarget(document.getElementById(id), canvas)).toBe(true);
    expect(isInterfaceTarget(canvas, canvas)).toBe(false);
    expect(isInterfaceTarget(document.getElementById("text"), canvas)).toBe(false);
    expect(isInterfaceTarget(document.body, canvas)).toBe(false);
  });
});

describe("how far the camera moves before a splat re-sort", () => {
  it("is a share of the nearest splat's distance, within bounds", () => {
    expect(resortDistance(2)).toBeCloseTo(0.1, 6);
    expect(resortDistance(0.3)).toBe(RESORT_MIN_M);
    expect(resortDistance(500)).toBe(RESORT_MAX_M);
    expect(resortDistance(Number.POSITIVE_INFINITY)).toBe(RESORT_MIN_M);
  });
});
