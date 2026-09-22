/**
 * The geometry half of the Observed/Simulated split.
 *
 * Every case here is a claim the UI is allowed to make, or one it is not. The one that matters
 * most is the last: the synthetic tree's own catalog entry must never come out of this function
 * reading like a capture, because that is the exact failure the Living Survey badge and
 * inspector exist to prevent.
 */

import { describe, expect, it } from "vitest";

import type { Site, SiteAsset } from "@twin/contracts";

import {
  formatCaptureDate,
  formatResolution,
  geometryProvenance,
  placementProvenance,
} from "@/lib/provenance";

const NOW = "2026-01-01T00:00:00Z";

function asset(over: Partial<SiteAsset> = {}): SiteAsset {
  return {
    id: "asset-1",
    siteId: "site-1",
    provider: "3d-tiles-url",
    name: "Gaussian splat",
    representation: "gaussian-splat",
    source: { type: "3d-tiles-url", url: "https://example.invalid/splat/tileset.json" },
    footprint: null,
    observedAt: null,
    validFrom: null,
    validTo: null,
    resolution: null,
    crs: null,
    license: null,
    attribution: [],
    provenance: null,
    renderConfig: {
      maximumScreenSpaceError: 16,
      pointCloudShading: null,
      clipsWorld: true,
      clipFootprint: "catalog",
      heightOffsetM: 0,
    },
    defaultVisible: true,
    createdAt: NOW,
    updatedAt: NOW,
    ...over,
  };
}

function site(assets: SiteAsset[], metadata: Record<string, unknown> = {}): Site {
  return {
    id: "site-1",
    slug: "site",
    name: "Site",
    description: null,
    boundary: { type: "Polygon", coordinates: [] },
    centroid: { longitude: 0, latitude: 0, height: 0 },
    areaM2: 1,
    thumbnailUrl: null,
    metadata,
    attribution: [],
    license: null,
    assets,
    cameraBookmarks: [],
    createdAt: NOW,
    updatedAt: NOW,
  } as unknown as Site;
}

describe("formatCaptureDate", () => {
  it("shows a year as a year rather than inventing a day", () => {
    // `new Date("2016")` is January 1st, and a date formatter would print it. Nobody recorded
    // a day or a month; the manifest said "2016".
    expect(formatCaptureDate("2016")).toBe("2016");
    expect(formatCaptureDate("summer 2019")).toBe("summer 2019");
  });

  it("formats a full timestamp and reports nothing when there is nothing", () => {
    expect(formatCaptureDate("2021-06-04T00:00:00Z")).toMatch(/2021/);
    expect(formatCaptureDate(null)).toBeNull();
    expect(formatCaptureDate("  ")).toBeNull();
  });
});

describe("formatResolution", () => {
  it("names ground sample distance as a property of the source imagery", () => {
    const text = formatResolution(
      asset({ resolution: { groundSampleDistanceM: 0.0272, pointSpacingM: null } }),
      "metric",
    );
    // Not "2.7 cm detail": the splat does not resolve at its source GSD, and saying so would
    // promise a precision the reconstruction never claimed.
    expect(text).toBe("about 2.7 cm per pixel of source imagery");
  });

  it("falls back to point spacing and to nothing at all", () => {
    expect(
      formatResolution(
        asset({ resolution: { groundSampleDistanceM: null, pointSpacingM: 0.193 } }),
        "metric",
      ),
    ).toBe("points about 19.3 cm apart");
    expect(formatResolution(asset({ resolution: null }), "metric")).toBeNull();
    // A resolution record with no numbers in it is not a resolution.
    expect(
      formatResolution(
        asset({ resolution: { groundSampleDistanceM: null, pointSpacingM: null } }),
        "metric",
      ),
    ).toBeNull();
  });
});

describe("geometryProvenance", () => {
  it("calls a capture measured only when a date or a resolution is recorded", () => {
    const measured = geometryProvenance(
      site(
        [
          asset({
            resolution: { groundSampleDistanceM: 0.0272, pointSpacingM: null },
            provenance: {
              notes: "OpenDroneMap (mesh, point cloud) and OpenSplat (gaussian splat)",
              sourceOrganization: null,
              sourceUrl: null,
              publishedAt: null,
            },
          }),
        ],
        { captured: "2016" },
      ),
      "asset-1",
      "metric",
    );
    expect(measured.measured).toBe(true);
    expect(measured.summary).toBe(
      "Measured capture · 2016 · about 2.7 cm per pixel of source imagery",
    );
    expect(measured.note).toBe("OpenDroneMap (mesh, point cloud) and OpenSplat (gaussian splat)");
  });

  it("says plainly when the catalog records neither, instead of guessing either way", () => {
    const unknown = geometryProvenance(site([asset()]), "asset-1", "metric");
    expect(unknown.measured).toBe(false);
    expect(unknown.summary).toBe("No capture date or resolution recorded");
    // Absence of metadata is not evidence of being generated, and the copy must not imply it.
    expect(unknown.summary).not.toMatch(/simulat|generat|procedural/i);
  });

  it("names the half it has when only one of date and resolution is recorded", () => {
    const dated = geometryProvenance(site([asset()], { captured: "2016" }), "asset-1", "metric");
    expect(dated.summary).toBe("Measured capture · 2016 · resolution not recorded");
    const sized = geometryProvenance(
      site([asset({ resolution: { groundSampleDistanceM: 0.014, pointSpacingM: null } })]),
      "asset-1",
      "metric",
    );
    expect(sized.summary).toBe(
      "Measured capture · capture date not recorded · about 1.4 cm per pixel of source imagery",
    );
  });

  it("quotes the deformed asset, not whichever asset comes first", () => {
    const both = site([
      asset({ id: "mesh", resolution: { groundSampleDistanceM: 0.014, pointSpacingM: null } }),
      asset({ id: "splat", resolution: { groundSampleDistanceM: 0.0272, pointSpacingM: null } }),
    ]);
    expect(geometryProvenance(both, "splat", "metric").summary).toContain("2.7 cm");
  });

  it("carries the synthetic tree's own words through untouched", () => {
    // Exactly what the seed archive holds for the fixture: no capture date, no
    // ground sample distance, and a pipeline note that says it was not captured at all.
    const fixture = geometryProvenance(
      site(
        [
          asset({
            provenance: {
              notes: "tools/captures/synthetic_tree.py (procedural; no capture, no reconstruction)",
              sourceOrganization: "Hexapod synthetic fixture",
              sourceUrl: null,
              publishedAt: null,
            },
          }),
        ],
        { captured: null },
      ),
      "asset-1",
      "metric",
    );
    expect(fixture.measured).toBe(false);
    expect(fixture.summary).toBe("No capture date or resolution recorded");
    expect(fixture.note).toBe(
      "tools/captures/synthetic_tree.py (procedural; no capture, no reconstruction)",
    );
    // The whole point: a reader of this output cannot come away thinking the tree was scanned.
    expect(`${fixture.summary} ${fixture.note}`).toMatch(/procedural; no capture/);
    expect(fixture.summary).not.toMatch(/Measured/);
  });

  it("reports honestly when there is no site and no asset at all", () => {
    const nothing = geometryProvenance(null, null, "metric");
    expect(nothing).toEqual({
      measured: false,
      summary: "No capture date or resolution recorded",
      note: "",
    });
  });
});

/**
 * The placement half, which is the one B4 added.
 *
 * The case that matters is the first: a capture somebody dropped on a map at plus or minus ten
 * metres and one aligned to EXIF GPS must not read alike. `geometryProvenance` calls both of
 * them "Measured capture" and is right to — the geometry was measured either way — so this is
 * the only function that can tell them apart.
 */
describe("placementProvenance", () => {
  const placed = (over: NonNullable<SiteAsset["provenance"]>) =>
    site([asset({ provenance: over })]);

  it("does not let a hand placement read like an alignment", () => {
    const byHand = placementProvenance(
      placed({ georefMethod: "manual", scaleSource: "unresolved", uncertaintyM: 10 }),
      "asset-1",
      "metric",
    );
    const aligned = placementProvenance(
      placed({ georefMethod: "exif-gps", scaleSource: "exif-gps", uncertaintyM: 1.4 }),
      "asset-1",
      "metric",
    );
    expect(byHand?.summary).toContain("Placed by hand");
    expect(byHand?.measured).toBe(false);
    expect(byHand?.scaleUnresolved).toBe(true);
    expect(aligned?.summary).toContain("Aligned to EXIF GPS");
    expect(aligned?.measured).toBe(true);
    expect(aligned?.scaleUnresolved).toBe(false);
    expect(byHand?.summary).not.toBe(aligned?.summary);
  });

  it("prints the uncertainty rather than implying there is none", () => {
    const withUncertainty = placementProvenance(
      placed({ georefMethod: "manual", scaleSource: "unresolved", uncertaintyM: 10 }),
      "asset-1",
      "metric",
    );
    const without = placementProvenance(
      placed({ georefMethod: "manual", scaleSource: "unresolved", uncertaintyM: null }),
      "asset-1",
      "metric",
    );
    expect(withUncertainty?.summary).toContain("\u00b1");
    expect(without?.summary).toContain("uncertainty not recorded");
  });

  it("does not call a bare coordinate an alignment", () => {
    // `exif_gps` writes `georefMethod: exif-gps` for an iPhone video whose only coordinate
    // came off the container, with nothing aligned and no scale. That is a location, and
    // the only thing in the record that distinguishes it is the scale source.
    const located = placementProvenance(
      placed({ georefMethod: "exif-gps", scaleSource: "unresolved", uncertaintyM: 10 }),
      "asset-1",
      "metric",
    );
    expect(located?.summary).toContain("Located by EXIF GPS");
    expect(located?.summary).not.toContain("Aligned");
    expect(located?.measured).toBe(false);
  });

  it("says nothing at all about an asset that predates the pipeline", () => {
    // Every legacy capture and every seeded reference layer is this case. A row reading
    // "unknown" would look like a finding; no row is the truth.
    expect(placementProvenance(site([asset()]), "asset-1", "metric")).toBeNull();
    expect(placementProvenance(null, null, "metric")).toBeNull();
  });

  it("falls back to whichever asset of the site records a placement", () => {
    const mixed = site([
      asset({ id: "mesh", representation: "mesh" }),
      asset({
        id: "splat",
        provenance: { georefMethod: "arkit", scaleSource: "arkit", uncertaintyM: 0.5 },
      }),
    ]);
    expect(placementProvenance(mixed, "mesh", "metric")?.summary).toContain("ARKit");
  });
});
