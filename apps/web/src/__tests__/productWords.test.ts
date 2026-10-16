import { describe, expect, it } from "vitest";

import { SCAN_SHOWCASE_SLUG } from "@/api/fallback";
import { siteDisplayName } from "@/features/sites/siteNames";
import {
  artifactKindLabel,
  captureKindLabel,
  captureStatusLabel,
  runStatusLabel,
  uploadStatusLabel,
} from "@/lib/labels";
import { DEMO_PROJECT } from "@/missions/demo";
import { DEMO_RANCH_SLUG } from "@/missions/demoRanchData";
import { PRODUCT_LINKS, productHeader } from "@/shared/product";

describe("one product, one set of words", () => {
  it("names a site by its project when one is attached, else by its record", () => {
    expect(siteDisplayName({ id: "x", slug: DEMO_RANCH_SLUG, name: "Ranch record" })).toBe(
      DEMO_PROJECT.name,
    );
    // The scan showcase has no project: its record names it.
    expect(siteDisplayName({ id: "z", slug: SCAN_SHOWCASE_SLUG, name: "Cesium demo" })).toBe(
      "Cesium demo",
    );
    expect(siteDisplayName({ id: "y", slug: "north-orchard", name: "North orchard" })).toBe(
      "North orchard",
    );
  });

  it("labels the API's values the way people say them", () => {
    expect(runStatusLabel("not-started")).toBe("Queued");
    expect(runStatusLabel("in-progress")).toBe("Running");
    expect(captureStatusLabel("not-started")).toBe("Ready");
    expect(captureKindLabel("gaussian-splat")).toBe("Splat");
    expect(artifactKindLabel("3d-tiles")).toBe("3D Tiles");
    expect(artifactKindLabel("point-cloud")).toBe("Points");
    expect(uploadStatusLabel("complete")).toBe("Uploaded");
    // Anything new from the API still shows, as itself.
    expect(runStatusLabel("paused")).toBe("paused");
  });

  it("builds the shared header with the current page marked", () => {
    const header = productHeader("scans");
    const links = Array.from(header.querySelectorAll("a.product-bar__link"));
    expect(links.map((a) => a.textContent)).toEqual(PRODUCT_LINKS.map((l) => l.label));
    expect(links.map((a) => a.getAttribute("href"))).toEqual(["/", "/view.html", "/admin.html"]);
    expect(header.querySelector('[aria-current="page"]')?.textContent).toBe("Scans");
    expect(header.querySelector(".product-bar__brand")?.getAttribute("href")).toBe("/");
  });
});
