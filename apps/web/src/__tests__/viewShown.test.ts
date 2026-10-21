import { describe, expect, it } from "vitest";

import type { Site } from "@twin/contracts";

import { datesNotice, shownScan } from "@/view/shown";

type SiteAsset = Site["assets"][number];

const COVERAGE = "https://tiles.example/runs/epoch-1/quality/coverage_enu.ply";

function splat(id: string, epochId: string | null, extra: Record<string, unknown> = {}): SiteAsset {
  return {
    id,
    representation: "gaussian-splat",
    provider: "3d-tiles-url",
    source: { type: "3d-tiles-url", url: `https://tiles.example/${id}/tileset.json` },
    epochId,
    sourceKind: "observed",
    validTo: null,
    ...extra,
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
  observedAt: `2026-0${String(ordinal * 3)}-01T15:00:00Z`,
  status: "published",
  currentAssetId,
});

describe("which scan view.html shows, and whose coverage", () => {
  it("opens on the latest epoch, without epoch 1's coverage", () => {
    // After a re-run of epoch 1, its new revision is the newest asset: epoch 2's comes first.
    // The scan shown is the latest date, as on the globe; the mirror's coverage is epoch 1's.
    const shown = shownScan(
      site(
        [splat("e2-asset", "e2"), splat("e1-rerun", "e1")],
        [epoch("e1", 1, "e1-rerun"), epoch("e2", 2, "e2-asset")],
      ),
    );

    expect(shown.asset?.id).toBe("e2-asset");
    expect(shown.epoch?.id).toBe("e2");
    expect(shown.coverage).toBeNull();
  });

  it("says it shows the latest of several dates, and where the others are", () => {
    const two = shownScan(
      site(
        [splat("e1-asset", "e1"), splat("e2-asset", "e2")],
        [epoch("e1", 1, "e1-asset"), epoch("e2", 2, "e2-asset")],
      ),
    );
    expect(two.dates).toBe(2);
    expect(datesNotice(two)).toBe("Latest of 2 dates · the globe shows the others");
    // One date, or none (a legacy site), needs no notice.
    const one = shownScan(site([splat("e1-asset", "e1")], [epoch("e1", 1, "e1-asset")]));
    expect(one.dates).toBe(1);
    expect(datesNotice(one)).toBeNull();
    expect(datesNotice(shownScan(site([splat("legacy", null)], undefined)))).toBeNull();
  });

  it("shows epoch 1's coverage when epoch 1's current revision is the scan shown", () => {
    // Epoch 2 was rejected, so epoch 1 is the latest date offered: its re-run, with coverage.
    const shown = shownScan(
      site(
        [splat("e2-asset", "e2"), splat("e1-rerun", "e1")],
        [epoch("e1", 1, "e1-rerun"), { ...epoch("e2", 2, "e2-asset"), status: "rejected" }],
      ),
    );

    expect(shown.asset?.id).toBe("e1-rerun");
    expect(shown.coverage).toEqual({ url: COVERAGE, verifiedPct: 81.5 });
  });

  it("never lays epoch 1's coverage over a revision that is not current", () => {
    // A superseded revision of epoch 1 is never shown, so never gets the overlay either.
    const shown = shownScan(
      site(
        [splat("e1-old", "e1", { validTo: "2026-04-01T00:00:00Z" }), splat("e1-rerun", "e1")],
        [epoch("e1", 1, "e1-rerun")],
      ),
    );
    expect(shown.asset?.id).toBe("e1-rerun");
    expect(shown.coverage?.url).toBe(COVERAGE);
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
