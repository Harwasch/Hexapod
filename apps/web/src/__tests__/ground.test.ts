import { afterEach, describe, expect, it, vi } from "vitest";

import { footprintContains } from "@twin/geo";

import { projectOutline, resolveGround, ROUGH_HALF_M } from "@/missions/ground";
import { containingQuery, fetchOsmContaining } from "@/missions/osm";

const point = { longitude: -119.9, latitude: 36.6 };
const square = (d: number) => [
  { lat: 36.6 - d, lon: -119.9 - d },
  { lat: 36.6 - d, lon: -119.9 + d },
  { lat: 36.6 + d, lon: -119.9 + d },
  { lat: 36.6 + d, lon: -119.9 - d },
  { lat: 36.6 - d, lon: -119.9 - d },
];

function overpass(elements: unknown[]) {
  return vi.fn(() =>
    Promise.resolve({ ok: true, json: () => Promise.resolve({ elements }) } as Response),
  );
}

const resolver = {
  point: { x: 0.5, y: 0.5 },
  snapshot: () => Promise.resolve({ image: "data:image/jpeg;base64,AAAA", width: 100, height: 50 }),
  unproject: (x: number, y: number) => ({
    longitude: -119.9 + x / 1000,
    latitude: 36.6 - y / 1000,
  }),
  outline: () =>
    Promise.resolve({
      points: [
        { x: 0.1, y: 0.1 },
        { x: 0.9, y: 0.1 },
        { x: 0.9, y: 0.9 },
        { x: 0.1, y: 0.9 },
      ],
      label: "orchard block",
      confidence: 0.8,
      note: "Followed the fence lines.",
      source: "claude" as const,
    }),
  vision: true,
};

describe("ground under a click", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("asks OpenStreetMap what contains the point and takes the smallest area", async () => {
    const fetch = overpass([
      { type: "way", id: 1, tags: { landuse: "farmland" }, geometry: square(0.05) },
      {
        type: "way",
        id: 2,
        tags: { name: "Home Orchard", landuse: "orchard" },
        geometry: square(0.004),
      },
      {
        type: "way",
        id: 3,
        tags: { natural: "water" },
        geometry: square(0.002).map((g) => ({ ...g, lon: g.lon + 0.1 })),
      },
    ]);
    vi.stubGlobal("fetch", fetch);
    const found = await fetchOsmContaining(point);
    expect(found?.name).toBe("Home Orchard");
    const init = (fetch.mock.calls[0] as unknown as [string, { body: URLSearchParams }])[1];
    expect(init.body.get("data")).toContain("is_in(36.6,-119.9)");
    expect(containingQuery(point)).toContain("way(pivot.a)[landuse]");
    expect(footprintContains(found!.footprint, point)).toBe(true);

    const ground = await resolveGround(point, "A-01", "scan the orchard", resolver);
    expect(ground.source).toBe("osm");
    expect(ground.zone.name).toBe("Home Orchard");
    expect(ground.zone.attribution).toContain("OpenStreetMap");
  });

  it("falls back to the model's outline, then to a rough square", async () => {
    vi.stubGlobal("fetch", overpass([]));
    const seen = await resolveGround(point, "A-01", "scan", resolver);
    expect(seen.source).toBe("vision");
    expect(seen.zone.name).toBe("Orchard block");
    expect(seen.zone.shaped).toBe(true);
    expect(seen.note).toContain("Drag the corners");

    const blind = await resolveGround(point, "A-02", "scan", { ...resolver, vision: false });
    expect(blind.source).toBe("rough");
    expect(footprintContains(blind.zone.footprint, point)).toBe(true);
    expect(blind.zone.acres).toBeGreaterThan(10);

    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.reject(new Error("offline"))),
    );
    const offline = await resolveGround(point, "A-03", "scan", {
      ...resolver,
      outline: () => Promise.reject(new Error("no model")),
    });
    expect(offline.source).toBe("rough");
  });

  it("projects an outline onto the ground and drops one that lands on the sky", () => {
    const outline = {
      points: [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 1, y: 1 },
      ],
      label: "x",
      confidence: 1,
      note: "",
      source: "claude" as const,
    };
    const snapshot = { image: "", width: 1000, height: 1000 };
    expect(projectOutline(outline, snapshot, resolver.unproject)?.type).toBe("Polygon");
    expect(projectOutline(outline, snapshot, () => null)).toBeNull();
    expect(ROUGH_HALF_M).toBeGreaterThan(50);
  });
});
