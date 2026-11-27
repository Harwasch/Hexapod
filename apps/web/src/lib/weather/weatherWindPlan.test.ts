import { describe, expect, it } from "vitest";

import { planWeatherWind, type WindScan } from "./weatherWindPlan";

const SEG = "https://tiles.example/grove/seg/instances.json";
const scan = (patch: Partial<WindScan> = {}): WindScan => ({
  assetId: "a",
  name: "Grove",
  tileset: "https://tiles.example/grove/tileset.json",
  splat: true,
  rigged: false,
  instancesUrl: "",
  ...patch,
});

describe("which scan weather wind moves", () => {
  it("prefers a baked rig, whose detail score then picks B or C", () => {
    const plan = planWeatherWind([scan({ instancesUrl: SEG })], { a: true }, "playcanvas");
    expect(plan).toEqual({
      kind: "target",
      assetId: "a",
      name: "Grove",
      url: "https://tiles.example/grove/articulation.json",
      source: "rig",
    });
  });

  it("waits for the rig probe rather than settling for Tier A", () => {
    expect(planWeatherWind([scan({ instancesUrl: SEG })], {}, "playcanvas")).toEqual({
      kind: "pending",
    });
  });

  it("falls back to segmentation (Tier A) and never to colour alone", () => {
    expect(
      planWeatherWind([scan({ instancesUrl: SEG })], { a: false }, "playcanvas"),
    ).toMatchObject({
      kind: "target",
      url: "https://tiles.example/grove/seg/instances.json?tier-a=1",
      source: "segmentation",
    });
    expect(planWeatherWind([scan()], { a: false }, "playcanvas")).toEqual({
      kind: "none",
      reason: "no vegetation data for this scan",
    });
  });

  it("leaves Living Survey scans, meshes and other renderers alone", () => {
    expect(planWeatherWind([scan({ rigged: true })], {}, "playcanvas")).toMatchObject({
      kind: "none",
      reason: "this site moves with Simulated wind above",
    });
    expect(planWeatherWind([scan({ splat: false })], {}, "playcanvas")).toMatchObject({
      kind: "none",
    });
    for (const renderer of ["cesium", "spark", "playcanvas-webgpu"])
      expect(planWeatherWind([scan()], { a: true }, renderer).kind).toBe("none");
  });

  it("keeps a signed tileset's query on both sidecars", () => {
    const signed = scan({
      tileset: "https://tiles.example/grove/tileset.json?sig=abc",
      instancesUrl: "https://tiles.example/grove/instances.json?sig=abc",
    });
    expect(planWeatherWind([signed], { a: true }, "playcanvas")).toMatchObject({
      url: "https://tiles.example/grove/articulation.json?sig=abc",
    });
    expect(planWeatherWind([signed], { a: false }, "playcanvas")).toMatchObject({
      url: "https://tiles.example/grove/instances.json?sig=abc&tier-a=1",
    });
  });

  it("resolves a relative tileset against the page", () => {
    const plan = planWeatherWind(
      [scan({ tileset: "/tiles/grove/tileset.json" })],
      { a: true },
      "playcanvas",
    );
    expect(plan.kind === "target" && plan.url).toBe(
      `${window.location.origin}/tiles/grove/articulation.json`,
    );
  });
});
