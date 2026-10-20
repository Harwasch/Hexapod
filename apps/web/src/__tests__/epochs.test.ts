/**
 * The one rule for which of a site's models is shown, epochs first (lib/epochs.ts), and the
 * words an epoch is named by.
 */
import { describe, expect, it } from "vitest";

import type { SiteAsset } from "@twin/contracts";

import {
  alignmentLabel,
  currentAsset,
  epochOptions,
  formatEpochDate,
  formatEpochDateTime,
  frameAnchor,
  observedAtFromUpload,
  pickShownAsset,
  platformLabel,
  shownOption,
  siteEpochs,
} from "@/lib/epochs";

import { A1, A2, E1, E2, epoch, splat, twoEpochSite } from "./epochFixtures";

const ids = (options: { epoch: { id: string } }[]) => options.map((o) => o.epoch.id);

describe("which asset of a site is shown", () => {
  it("lists published epochs with a current asset of the representation, oldest first", () => {
    // Listed newest first, and a mesh of E1 only: each representation has its own list.
    const mesh = splat("m1", { epochId: "e1", representation: "mesh" });
    expect(ids(epochOptions([E2, E1], [A1, A2, mesh], "gaussian-splat"))).toEqual(["e1", "e2"]);
    expect(ids(epochOptions([E2, E1], [A1, A2, mesh], "mesh"))).toEqual(["e1"]);
    expect(epochOptions([E1, E2], [A1, A2], "point-cloud")).toEqual([]);
  });

  it("an epoch's current asset is its newest one not superseded", () => {
    const e = epoch("e1");
    const old = splat("old", { epochId: "e1", createdAt: "2026-03-02T00:00:00Z" });
    const superseded = splat("newer", {
      epochId: "e1",
      createdAt: "2026-03-05T00:00:00Z",
      validTo: "2026-03-06T00:00:00Z",
    });
    expect(currentAsset(e, [old, superseded], "gaussian-splat")?.id).toBe("old");
    const rerun = splat("rerun", { epochId: "e1", createdAt: "2026-03-07T00:00:00Z" });
    expect(currentAsset(e, [old, rerun], "gaussian-splat")?.id).toBe("rerun");
    // The server's answer wins when it names an asset of this representation.
    expect(currentAsset({ ...e, currentAssetId: "old" }, [old, rerun], "gaussian-splat")?.id).toBe(
      "old",
    );
    // ... and is recomputed for a representation it does not name.
    const mesh = splat("mesh", { epochId: "e1", representation: "mesh" });
    expect(currentAsset({ ...e, currentAssetId: "old" }, [old, mesh], "mesh")?.id).toBe("mesh");
  });

  it("rejected and superseded epochs are not offered", () => {
    const rejected = { ...E2, status: "rejected" as const };
    expect(ids(epochOptions([E1, rejected], [A1, A2], "gaussian-splat"))).toEqual(["e1"]);
    const superseded = { ...E2, status: "superseded" as const };
    expect(ids(epochOptions([E1, superseded], [A1, A2], "gaussian-splat"))).toEqual(["e1"]);
    // And their asset is never shown, even as the newest.
    expect(pickShownAsset([A1, A2], [E1, rejected], "gaussian-splat")?.id).toBe("a1");
  });

  it("an inferred asset is never an epoch's current asset", () => {
    const filled = splat("filled", {
      epochId: "e2",
      sourceKind: "inferred",
      createdAt: "2026-07-01T00:00:00Z",
    });
    expect(currentAsset({ ...E2, currentAssetId: null }, [A2, filled], "gaussian-splat")?.id).toBe(
      "a2",
    );
    expect(pickShownAsset([A1, A2, filled], [E1, E2], "gaussian-splat")?.id).toBe("a2");
  });

  it("shows the latest epoch unless another is chosen", () => {
    // Both are default-visible and A1 is listed first: the old rule would have shown A1.
    expect(pickShownAsset([A1, A2], [E1, E2], "gaussian-splat")?.id).toBe("a2");
    expect(pickShownAsset([A1, A2], [E1, E2], "gaussian-splat", { epochId: "e1" })?.id).toBe("a1");
    expect(shownOption(epochOptions([E1, E2], [A1, A2], "gaussian-splat"), null)?.epoch.id).toBe(
      "e2",
    );
  });

  it("a chosen epoch with nothing of this representation falls back to the latest", () => {
    const mesh1 = splat("m1", { epochId: "e1", representation: "mesh" });
    const mesh2 = splat("m2", { epochId: "e2", representation: "mesh" });
    // E2 has no mesh: its own splat is shown under Splat, E1's mesh under Mesh.
    expect(pickShownAsset([A1, A2, mesh1], [E1, E2], "mesh", { epochId: "e2" })?.id).toBe("m1");
    // An id that is no epoch of this site: the latest.
    expect(pickShownAsset([A1, A2, mesh1, mesh2], [E1, E2], "mesh", { epochId: "gone" })?.id).toBe(
      "m2",
    );
  });

  it("a pinned asset wins (the load pill's Retry)", () => {
    expect(pickShownAsset([A1, A2], [E1, E2], "gaussian-splat", { assetId: "a1" })?.id).toBe("a1");
    // Only among the candidates: a superseded revision cannot be pinned back on screen.
    const superseded = splat("old", { epochId: "e2", validTo: "2026-06-03T00:00:00Z" });
    expect(
      pickShownAsset([A1, A2, superseded], [E1, E2], "gaussian-splat", { assetId: "old" })?.id,
    ).toBe("a2");
  });

  it("a site without epochs keeps the old rule", () => {
    const asset = (id: string, representation: SiteAsset["representation"], extra = {}) =>
      ({ id, representation, defaultVisible: false, observedAt: null, ...extra }) as SiteAsset;
    const assets = [
      asset("old-mesh", "mesh", { observedAt: "2024-01-01" }),
      asset("new-mesh", "mesh", { observedAt: "2025-01-01" }),
      asset("splat", "gaussian-splat", { defaultVisible: true }),
    ];
    expect(pickShownAsset(assets, [], "mesh")?.id).toBe("new-mesh");
    expect(pickShownAsset(assets, [], "mesh", { assetId: "old-mesh" })?.id).toBe("old-mesh");
    expect(pickShownAsset(assets, [], "gaussian-splat")?.id).toBe("splat");
    // It does not reorder the caller's list.
    expect(assets.map((a) => a.id)).toEqual(["old-mesh", "new-mesh", "splat"]);
  });

  it("epoch assets with no epoch list show the newest current one, never nothing", () => {
    const superseded = splat("gone", {
      epochId: "e2",
      createdAt: "2026-09-01T00:00:00Z",
      validTo: "2026-09-02T00:00:00Z",
    });
    expect(pickShownAsset([A1, A2, superseded], [], "gaussian-splat")?.id).toBe("a2");
    // Nor ever an inferred one, however new.
    const filled = splat("filled", {
      epochId: "e2",
      sourceKind: "inferred",
      createdAt: "2026-09-03T00:00:00Z",
    });
    expect(pickShownAsset([A1, A2, filled], [], "gaussian-splat")?.id).toBe("a2");
    // A record that does not carry `epochs` at all reads as none.
    expect(siteEpochs({ ...twoEpochSite(), epochs: undefined } as never)).toEqual([]);
    expect(siteEpochs(twoEpochSite())).toEqual([E1, E2]);
  });

  it("the frame-defining epoch is the lowest ordinal, else the oldest offered", () => {
    expect(frameAnchor([E2, E1], [A1, A2], "gaussian-splat")?.epoch.id).toBe("e1");
    // An explicit join to a frameless site makes the frame too (it is ordinal 1).
    const explicit = { ...E1, joinMethod: "explicit" as const };
    expect(frameAnchor([E2, explicit], [A1, A2], "gaussian-splat")?.epoch.id).toBe("e1");
    // Ordinal 1 rejected: the lowest ordinal still offered.
    const e3 = epoch("e3", { ordinal: 3, observedAt: "2026-01-01T00:00:00Z" });
    const a3 = splat("a3", { epochId: "e3" });
    const rejected = { ...E1, status: "rejected" as const };
    expect(frameAnchor([rejected, E2, e3], [A1, A2, a3], "gaussian-splat")?.epoch.id).toBe("e2");
    expect(frameAnchor([], [A1, A2], "gaussian-splat")).toBeNull();
  });

  it('labels a phone epoch "aligned visually", never surveyed; null alignment is "Not aligned yet"; the anchor "Defines the site frame"', () => {
    const all = [E1, E2];
    expect(alignmentLabel(E1, E1, all)).toEqual({
      short: "site frame",
      long: "Defines the site frame",
    });
    expect(alignmentLabel(E2, E1, all)).toEqual({
      short: "not aligned",
      long: "Not aligned yet: placed by its own GPS",
    });
    const coarse = { ...E2, alignment: { verdict: "coarse", uncertaintyM: 0.0654 } };
    expect(alignmentLabel(coarse, E1, all).long).toBe("Phone video, aligned visually, ±0.065 m");
    expect(alignmentLabel(coarse, E1, all).long).not.toMatch(/survey/i);
    const survey = {
      ...E2,
      platform: "rtk-drone" as const,
      alignment: { verdict: "survey", uncertaintyM: 0.0123 },
    };
    expect(alignmentLabel(survey, E1, all).long).toBe(
      "Aligned to Mar 1, 2026 (survey tier, ±0.012 m)",
    );
    const unaligned = { ...E2, alignment: { verdict: "unaligned" } };
    expect(alignmentLabel(unaligned, E1, all).long).toBe("Not aligned: placed by its own GPS");
    // A verdict this build does not know claims nothing; neither does a missing X.
    expect(alignmentLabel({ ...E2, alignment: { verdict: "maybe" } }, E1, all)).toEqual(
      alignmentLabel(E2, E1, all),
    );
    expect(alignmentLabel({ ...E2, alignment: { verdict: "coarse" } }, E1, all).long).toBe(
      "Phone video, aligned visually",
    );
    // An anchor that is not the lowest ordinal (ordinal 1 rejected) did not make the frame.
    expect(alignmentLabel(E2, E2, all).long).toBe("Not aligned yet: placed by its own GPS");
  });

  it("never calls an epoch survey-grade unless an RTK drone captured it", () => {
    const all = [E1, E2];
    // A phone epoch whose record says survey (a bug upstream, or a tolerance too loose) is
    // shown at the coarse tier, and a label claiming more is not repeated.
    const phone = {
      ...E2,
      alignment: { verdict: "survey", uncertaintyM: 0.012, label: "Survey tier, ±0.012 m" },
    };
    expect(alignmentLabel(phone, E1, all)).toEqual({
      short: "aligned visually",
      long: "Phone video, aligned visually, ±0.012 m",
    });
    const drone = { ...E2, platform: "drone" as const, alignment: { verdict: "survey" } };
    expect(alignmentLabel(drone, E1, all).short).toBe("aligned visually");
    expect(alignmentLabel(drone, E1, all).long).not.toMatch(/survey/i);
    const rtk = { ...E2, platform: "rtk-drone" as const, alignment: { verdict: "survey" } };
    expect(alignmentLabel(rtk, E1, all).short).toBe("survey tier");
  });

  it("prefers the ready-made label M3 writes (verdict, uncertaintyM, label)", () => {
    const m3 = {
      ...E2,
      alignment: {
        verdict: "coarse",
        uncertaintyM: 0.065,
        label: "Phone video, aligned visually, ±0.065 m",
      },
    };
    expect(alignmentLabel(m3, E1, [E1, m3])).toEqual({
      short: "aligned visually",
      long: "Phone video, aligned visually, ±0.065 m",
    });
  });

  it("names platforms, and shows one it does not know as written", () => {
    expect(platformLabel("rtk-drone")).toBe("RTK drone");
    expect(platformLabel("drone")).toBe("Drone");
    expect(platformLabel("phone-video")).toBe("Phone video");
    expect(platformLabel("phone-photo")).toBe("Phone photos");
    expect(platformLabel("unknown")).toBe("Unknown platform");
    expect(platformLabel(null)).toBe("Unknown platform");
    expect(platformLabel("balloon")).toBe("balloon");
  });

  it("dates are UTC", () => {
    expect(formatEpochDate("2026-03-01T02:00:00Z")).toMatch(/Mar 1, 2026/);
    expect(formatEpochDate("2026-03-01T23:30:00-05:00")).toMatch(/Mar 2, 2026/);
    expect(formatEpochDateTime("2026-03-01T15:04:00Z")).toMatch(/Mar 1, 2026.*03:04\sPM UTC$/);
    expect(observedAtFromUpload({ ...E1, observedAtSource: "created" })).toBe(true);
    expect(observedAtFromUpload(E1)).toBe(false);
  });
});
