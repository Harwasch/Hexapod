/**
 * The offline catalog: what the console shows when the API is unreachable but the bucket
 * is not.
 *
 * The reason this file exists at all is the A9 change in what "offline" means. Capture
 * tiles used to be a `StaticFiles` mount on the API, so an unreachable API meant
 * unreachable tiles and a capture listed here would have been a dead link. Now the tiles
 * are in object storage, `app/seed/publish.py` publishes `catalog.json` beside them, and
 * the console reads that instead of carrying a table in its bundle.
 */

import { describe, expect, it } from "vitest";

import { readOfflineCatalog } from "@/api/fallback";
import { readEnv } from "@/app/env";

const SITE = {
  id: "8d1f6b2e-0000-4000-8000-000000000001",
  slug: "sheffield-park",
  name: "Sheffield Park, Florida",
  description: null,
  boundary: { type: "Polygon", coordinates: [[]] },
  centroid: { longitude: -82.6966556, latitude: 28.0390667, height: 10.8 },
  areaM2: 1,
  thumbnailUrl: null,
  metadata: { origin: "capture-pipeline" },
  attribution: [],
  license: null,
  assets: [],
  cameraBookmarks: [],
  createdAt: "2026-09-15T00:00:00Z",
  updatedAt: "2026-09-15T00:00:00Z",
};

describe("readOfflineCatalog", () => {
  it("reads the sites a publish put in the bucket", () => {
    const sites = readOfflineCatalog({ version: 1, sites: [SITE] });
    expect(sites.map((site) => site.slug)).toEqual(["sheffield-park"]);
  });

  it("survives a stale or wrong document rather than taking the globe down with it", () => {
    // Every one of these is a real possibility: a bucket that 404s to an HTML error page,
    // a half-written file, a document from a future schema. None of them is worth a blank
    // screen, so each degrades to "no offline captures" and the built-in demo stands alone.
    expect(readOfflineCatalog(null)).toEqual([]);
    expect(readOfflineCatalog("<!doctype html>")).toEqual([]);
    expect(readOfflineCatalog({})).toEqual([]);
    expect(readOfflineCatalog({ sites: "soon" })).toEqual([]);
    expect(readOfflineCatalog({ sites: [SITE, { slug: "half-written" }, 7, null] })).toHaveLength(
      1,
    );
  });
});

describe("the offline catalog's URL", () => {
  it("is a public Vite variable, because it is the bucket's public read URL", () => {
    const env = readEnv({
      VITE_OFFLINE_CATALOG_URL: " https://tiles.example.com/catalog.json ",
      DEV: false,
    } as unknown as ImportMetaEnv);
    expect(env.offlineCatalogUrl).toBe("https://tiles.example.com/catalog.json");
  });

  it("is absent in a checkout with no bucket, and nothing is then fetched", () => {
    expect(readEnv({ DEV: true } as unknown as ImportMetaEnv).offlineCatalogUrl).toBeUndefined();
  });
});
