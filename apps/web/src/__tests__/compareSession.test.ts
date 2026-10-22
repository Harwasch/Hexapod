/**
 * Two dates compared under PlayCanvas (SiteManager `setCompareDates`, ADR 0017): the session
 * (compareSession.ts, loaded on demand) hands the renderer drawing the right-hand date the left
 * one, placed in its frame, and the divider (scanView/scanCompare.ts). The renderer's side is
 * playcanvasCompare.test.ts's; that the cut is drawn is e2e/compareScene.spec.ts's.
 */
import { Cartesian3, Event, Matrix4, Transforms } from "cesium";
import { afterEach, describe, expect, it } from "vitest";

import { startCompare, type CompareSite } from "@/cesium/compareSession";
import { RULE_KIT } from "@/cesium/scanView/ruleKit";
import {
  publishScanCompare,
  watchScanCompare,
  type ScanCompareView,
} from "@/cesium/scanView/scanCompare";
import { lentRuleKit } from "@/cesium/scanView/scanRule";
import { useSplatRoom } from "@/state/splatRoom";

import { A1, A2, E1, E2 } from "./epochFixtures";

const ROOT = Transforms.eastNorthUpToFixedFrame(Cartesian3.fromDegrees(-123.88, 46.1335, 10));

function tileset(url: string, computed: Matrix4 = ROOT) {
  return {
    root: { transform: Matrix4.IDENTITY, computedTransform: computed, extras: { url } },
    resource: { url },
    isDestroyed: () => false,
  };
}

function compared() {
  const march = tileset("https://tiles.example/a1/tileset.json");
  const june = tileset("https://tiles.example/a2/tileset.json");
  const scene = { preUpdate: new Event(), splitPosition: 0.4 };
  const entry = {
    compare: { left: E1.id, right: E2.id },
    representation: "gaussian-splat",
    epochs: [E1, E2],
    site: { id: "site-epochs", assets: [A1, A2] },
    handles: new Map([
      [A1.id, { tileset: march }],
      [A2.id, { tileset: june }],
    ]),
  } as unknown as CompareSite;
  const seen: (ScanCompareView | null)[] = [];
  const stop = watchScanCompare(A2.id, (view) => seen.push(view));
  return { scene, entry, march, june, seen, stop };
}

describe("two dates compared, as PlayCanvas is handed them", () => {
  afterEach(() => {
    publishScanCompare(A1.id, null);
    publishScanCompare(A2.id, null);
  });

  it("hands the right-hand date's renderer the left one and the divider, and lends the kit", () => {
    const { scene, entry, seen, stop } = compared();
    startCompare(scene, entry, RULE_KIT);
    expect(lentRuleKit()).not.toBeNull();
    const view = seen.at(-1);
    expect(view?.left).toEqual({
      url: "https://tiles.example/a1/tileset.json",
      assetId: A1.id,
      extras: { url: "https://tiles.example/a1/tileset.json" },
      // Both dates share the site's one placement: the left one's tiles go where they are.
      placement: null,
      root: Matrix4.toArray(Matrix4.IDENTITY),
      owns: [],
    });
    expect(view?.split).toBe(0.4);
    expect(Matrix4.equals(view?.toWorld(), ROOT)).toBe(true);
    stop();
  });

  it("hands over both roots, and ends the comparison, saying why, when the renderer has no room", () => {
    const { scene, entry, march, june, seen, stop } = compared();
    (march.root.extras as Record<string, unknown>).gaussians = 300_000;
    (june.root.extras as Record<string, unknown>).gaussians = 200_000;
    let ended = 0;
    startCompare(scene, entry, RULE_KIT, () => {
      ended += 1;
    });
    const view = seen.at(-1);
    expect(view?.scanRoot).toBe(200_000);
    expect(view?.left.rootGaussians).toBe(300_000);
    view?.refuse?.(2);
    expect(ended).toBe(1);
    expect(useSplatRoom.getState().compare).toEqual({
      key: `site-epochs|${E1.id}|${E2.id}`,
      dates: 2,
    });
    useSplatRoom.setState({ compare: null });
    stop();
  });

  it("follows the divider, once a frame, and says nothing new while it holds still", () => {
    const { scene, entry, seen, stop } = compared();
    startCompare(scene, entry, RULE_KIT);
    const before = seen.length;
    scene.preUpdate.raiseEvent();
    expect(seen.length).toBe(before);
    scene.splitPosition = 0.7;
    scene.preUpdate.raiseEvent();
    expect(seen.at(-1)?.split).toBe(0.7);
    expect(seen.at(-1)?.left).toEqual(seen.at(-2)?.left);
    stop();
  });

  it("places a date in its own frame where it is in the scan's", () => {
    const { scene, entry, march, seen, stop } = compared();
    const east = Matrix4.multiply(
      ROOT,
      Matrix4.fromTranslation(new Cartesian3(2, 0, 0)),
      new Matrix4(),
    );
    march.root.computedTransform = east;
    startCompare(scene, entry, RULE_KIT);
    const placement = seen.at(-1)?.left.placement ?? [];
    expect(placement[12]).toBeCloseTo(2, 6);
    expect(placement[13]).toBeCloseTo(0, 6);
    stop();
  });

  it("lets go once the comparison ends or changes, and a new pair is its own", () => {
    const { scene, entry, seen, stop } = compared();
    startCompare(scene, entry, RULE_KIT);
    // Asked for twice (the same pair): one session.
    startCompare(scene, entry, RULE_KIT);
    expect(seen.filter(Boolean)).toHaveLength(1);
    // Swapped: the left date is the scan now; the right-hand renderer is let go.
    const flipped = { left: E2.id, right: E1.id };
    (entry as { compare: unknown }).compare = flipped;
    const other: (ScanCompareView | null)[] = [];
    const off = watchScanCompare(A1.id, (view) => other.push(view));
    startCompare(scene, entry, RULE_KIT);
    scene.preUpdate.raiseEvent();
    expect(seen.at(-1)).toBeNull();
    expect(other.at(-1)?.left.assetId).toBe(A2.id);
    (entry as { compare: unknown }).compare = null;
    scene.preUpdate.raiseEvent();
    expect(other.at(-1)).toBeNull();
    // Nothing left listening.
    expect(scene.preUpdate.numberOfListeners).toBe(0);
    off();
    stop();
  });

  it("draws nothing beside the scan until both dates' tilesets are in", () => {
    const { scene, entry, seen, stop } = compared();
    (entry.handles as Map<string, unknown>).set(A1.id, { tileset: null });
    startCompare(scene, entry, RULE_KIT);
    expect(seen.filter(Boolean)).toHaveLength(0);
    stop();
  });

  it("is let go at once by its disposer, its view taken back, with no frame after", () => {
    const { scene, entry, seen, stop } = compared();
    const dispose = startCompare(scene, entry, RULE_KIT);
    expect(seen.at(-1)?.left.assetId).toBe(A1.id);
    dispose();
    expect(seen.at(-1)).toBeNull();
    expect(scene.preUpdate.numberOfListeners).toBe(0);
    // A second start for the same pair is a session of its own again.
    startCompare(scene, entry, RULE_KIT);
    expect(seen.at(-1)?.left.assetId).toBe(A1.id);
    stop();
  });
});
