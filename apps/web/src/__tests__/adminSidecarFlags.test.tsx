/**
 * The console's notice for what a republish could not carry onto new tiles.
 *
 * The worker flags an asset for every sidecar kind it drops (`sidecarFlags`,
 * docs/SCENE_OBJECTS.md section 8); without a notice the scan just stops showing its objects.
 * The rules (which assets, under which site) and the badges, here; admin.spec.ts drives the
 * page.
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { SiteAsset } from "@twin/contracts";

import { flagDetail, flaggedAssets, flaggedBySite } from "@/admin/sidecarFlags";
import { FlaggedAssets, SidecarNotice } from "@/admin/SidecarNotice";

const NOW = "2026-10-03T12:34:56Z";
const RUN = "77777777-7777-4777-8777-777777777777";

const OBJECTS = {
  kind: "instances",
  action: "Objects need re-segmenting",
  reason: "the run published new tiles whose positions are not all ones it was bound to",
  jobId: RUN,
  flaggedAt: NOW,
};
const LOD = {
  kind: "nativeLod",
  action: "Streamed LOD needs a backfill",
  reason: "the run published new tiles, and it was bound to the previous ones",
  jobId: null,
  flaggedAt: NOW,
};

function asset(over: Partial<SiteAsset> = {}): SiteAsset {
  return {
    id: "asset-1",
    siteId: "site-1",
    provider: "3d-tiles-url",
    name: "Camp splat",
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
    sidecarFlags: [],
    createdAt: NOW,
    updatedAt: NOW,
    ...over,
  };
}

describe("what a republish dropped", () => {
  it("lists only the flagged assets, by site, flags in kind order", () => {
    const assets = [
      asset({ id: "a", sidecarFlags: [LOD, OBJECTS] }),
      asset({ id: "b" }),
      asset({ id: "c", siteId: "site-2", sidecarFlags: [OBJECTS] }),
      asset({ id: "d", siteId: null, sidecarFlags: [LOD] }),
    ];
    const flagged = flaggedAssets(assets);
    expect(flagged.map((a) => a.id)).toEqual(["a", "c", "d"]);
    expect(flagged[0]?.flags.map((f) => f.kind)).toEqual(["instances", "nativeLod"]);
    const bySite = flaggedBySite(assets);
    expect(Object.keys(bySite).sort()).toEqual(["site-1", "site-2"]);
    expect(bySite["site-1"]?.map((a) => a.id)).toEqual(["a"]);
    expect(flaggedAssets(undefined)).toEqual([]);
  });

  it("says why and by which run in the tooltip", () => {
    expect(flagDetail(OBJECTS)).toBe(
      `${OBJECTS.reason} (dropped by run 77777777, 2026-10-03 12:34 UTC)`,
    );
    expect(flagDetail(LOD)).toBe(`${LOD.reason} (dropped, 2026-10-03 12:34 UTC)`);
  });

  it("draws a badge per dropped kind, in a person's words, and nothing when none", () => {
    const { container, rerender } = render(<SidecarNotice flags={[OBJECTS, LOD]} />);
    const badges = screen.getAllByTestId("sidecar-flag");
    expect(badges.map((badge) => badge.textContent?.trim())).toEqual([
      "Objects need re-segmenting",
      "Streamed LOD needs a backfill",
    ]);
    expect(badges[0]?.getAttribute("title")).toContain("positions are not all ones");
    rerender(<SidecarNotice flags={[]} />);
    expect(container.innerHTML).toBe("");
  });

  it("lists every flagged asset with its site, and is absent when nothing is flagged", () => {
    const { rerender } = render(
      <FlaggedAssets
        assets={flaggedAssets([asset({ sidecarFlags: [OBJECTS] })])}
        siteNames={{ "site-1": "Camp" }}
      />,
    );
    const card = screen.getByTestId("flagged-assets");
    expect(card.textContent).toContain("Camp splat");
    expect(card.textContent).toContain("Camp");
    expect(card.textContent).toContain("Objects need re-segmenting");
    rerender(<FlaggedAssets assets={[]} siteNames={{}} />);
    expect(screen.queryByTestId("flagged-assets")).toBeNull();
  });
});
