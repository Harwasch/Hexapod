/**
 * A site with two dated captures (epochs), for the epoch tests: E1, the drone capture that
 * made the site's frame on 2026-03-01, and E2, a phone video joined to it on 2026-06-01, each
 * with its gaussian splat (A1, A2). Plain records, cast: only what the code reads is filled in.
 */
import type { Epoch, Site, SiteAsset } from "@twin/contracts";

export const SITE_ID = "site-epochs";

export function epoch(id: string, extra: Partial<Epoch> = {}): Epoch {
  return {
    id,
    siteId: SITE_ID,
    ordinal: 1,
    captureId: `${id}-capture`,
    jobId: null,
    observedAt: "2026-03-01T15:00:00Z",
    observedAtSource: "captured-at",
    platform: "drone",
    status: "published",
    joinMethod: "new-site",
    alignment: null,
    quality: null,
    currentAssetId: null,
    footprint: null,
    createdAt: "2026-03-02T00:00:00Z",
    updatedAt: "2026-03-02T00:00:00Z",
    ...extra,
  } as unknown as Epoch;
}

export function splat(id: string, extra: Partial<SiteAsset> = {}): SiteAsset {
  return {
    id,
    siteId: SITE_ID,
    name: `splat ${id}`,
    provider: "3d-tiles-url",
    representation: "gaussian-splat",
    source: { type: "3d-tiles-url", url: `https://tiles.example/${id}/tileset.json` },
    renderConfig: { maximumScreenSpaceError: 16, clipsWorld: false, heightOffsetM: 0 },
    defaultVisible: true,
    epochId: null,
    sourceKind: "observed",
    validFrom: null,
    validTo: null,
    observedAt: null,
    attribution: [],
    footprint: null,
    resolution: null,
    provenance: null,
    createdAt: "2026-03-02T00:00:00Z",
    updatedAt: "2026-03-02T00:00:00Z",
    ...extra,
  } as unknown as SiteAsset;
}

export const E1 = epoch("e1", { currentAssetId: "a1" });
export const E2 = epoch("e2", {
  ordinal: 2,
  observedAt: "2026-06-01T15:00:00Z",
  platform: "phone-video",
  joinMethod: "spatial",
  currentAssetId: "a2",
  createdAt: "2026-06-02T00:00:00Z",
});
export const A1 = splat("a1", { epochId: "e1", observedAt: E1.observedAt });
export const A2 = splat("a2", {
  epochId: "e2",
  observedAt: E2.observedAt,
  createdAt: "2026-06-02T00:00:00Z",
});

/** The two-epoch site, its record as the API gives it (`SiteRead.epochs`). */
export function twoEpochSite(extra: Partial<Site> = {}): Site {
  return {
    id: SITE_ID,
    slug: "yard",
    name: "Yard",
    assets: [A1, A2],
    epochs: [E1, E2],
    frame: null,
    attribution: [],
    cameraBookmarks: [],
    metadata: {},
    license: null,
    ...extra,
  } as unknown as Site;
}
