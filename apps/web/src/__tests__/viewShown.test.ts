import { describe, expect, it } from "vitest";

import type { Site } from "@twin/contracts";

import { shownScan } from "@/view/shown";

type SiteAsset = Site["assets"][number];

const COVERAGE = "https://tiles.example/runs/epoch-1/quality/coverage_enu.ply";

function splat(id: string, epochId: string | null): SiteAsset {
  return {
    id,
    representation: "gaussian-splat",
    provider: "3d-tiles-url",
    source: { type: "3d-tiles-url", url: `https://tiles.example/${id}/tileset.json` },
    epochId,
  } as unknown as SiteAsset;
}

function site(assets: SiteAsset[], epochs: unknown[] | undefined): Site {
  return {
    id: "site",
    assets,
    epochs,
    metadata: { captureId: "c1", coverageUrl: COVERAGE, keepVerifiedPct: 81.5 },
  } as unknown as Site;
}

const epoch = (id: string, ordinal: number, currentAssetId: string | null) => ({
  id,
  ordinal,
  status: "published",
  currentAssetId,
});

describe("which scan view.html shows, and whose coverage", () => {
  it("shows epoch 1's current revision, with the coverage the mirror holds for it", () => {
    // After a re-run of epoch 1, its new revision is the newest asset: epoch 2's comes first.
    const shown = shownScan(
      site(
        [splat("e2-asset", "e2"), splat("e1-rerun", "e1")],
        [epoch("e1", 1, "e1-rerun"), epoch("e2", 2, "e2-asset")],
      ),
    );

    expect(shown.asset?.id).toBe("e1-rerun");
    expect(shown.coverage).toEqual({ url: COVERAGE, verifiedPct: 81.5 });
  });

  it("drops the overlay when the scan it shows is not epoch 1's", () => {
    // Epoch 1 was withheld (no tiles), so the scan shown is epoch 2's; the mirror's
    // coverage was measured on epoch 1's run and must not lie over epoch 2's tiles.
    const shown = shownScan(
      site([splat("e2-asset", "e2")], [epoch("e1", 1, null), epoch("e2", 2, "e2-asset")]),
    );

    expect(shown.asset?.id).toBe("e2-asset");
    expect(shown.coverage).toBeNull();
  });

  it("keeps a legacy site's first splat and its coverage", () => {
    for (const epochs of [undefined, []]) {
      const shown = shownScan(site([splat("legacy", null), splat("later", null)], epochs));
      expect(shown.asset?.id).toBe("legacy");
      expect(shown.coverage?.url).toBe(COVERAGE);
    }
  });

  it("offers no coverage the metadata does not carry", () => {
    const bare: Site = { ...site([splat("legacy", null)], []), metadata: {} };
    expect(shownScan(bare).coverage).toBeNull();
    expect(shownScan(site([], [])).asset).toBeUndefined();
  });
});
