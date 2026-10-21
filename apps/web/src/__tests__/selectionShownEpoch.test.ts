/**
 * A site's selection card names the model the scene shows: on a site with two dated
 * captures, the shown epoch's asset, its name, date and attribution -- not the site's first
 * asset of the representation. Through the real SelectionManager on a stub scene.
 */
import { Ellipsoid, EntityCollection, Event as CesiumEvent, JulianDate, Ray } from "cesium";
import { afterEach, expect, it, vi } from "vitest";

import { SelectionManager } from "@/cesium/SelectionManager";
import type { SceneEvents } from "@/cesium/types";
import { Emitter } from "@/lib/emitter";

import { A1, A2, E1, SITE_ID, twoEpochSite } from "./epochFixtures";

const managers: SelectionManager[] = [];
afterEach(() => {
  for (const manager of managers.splice(0)) manager.destroy();
});

function rig(first: typeof A1) {
  let shown = first;
  const site = twoEpochSite({
    boundary: {
      type: "Polygon",
      coordinates: [
        [
          [-123.881, 46.133],
          [-123.879, 46.133],
          [-123.879, 46.134],
          [-123.881, 46.134],
          [-123.881, 46.133],
        ],
      ],
    },
    assets: [
      { ...A1, name: "March drone flight", observedAt: "2026-03-01T15:00:00Z" },
      { ...A2, name: "June phone video", observedAt: "2026-06-01T15:00:00Z" },
    ],
  });
  const scene = {
    pickAsync: vi.fn(() => Promise.resolve(undefined)),
    drillPick: vi.fn(() => []),
    pickPositionSupported: true,
    pickPosition: vi.fn(() => undefined),
    globe: { pick: vi.fn(() => undefined), ellipsoid: Ellipsoid.WGS84 },
    postRender: new CesiumEvent(),
    requestRender: vi.fn(),
    isDestroyed: () => false,
  };
  const viewer = {
    canvas: document.createElement("canvas"),
    scene,
    camera: { getPickRay: () => new Ray(), pickEllipsoid: () => undefined },
    clock: { currentTime: JulianDate.now() },
    entities: new EntityCollection(),
  };
  const sites = {
    activeSite: site,
    activeRepresentation: "gaussian-splat",
    shownAsset: () => site.assets.find((asset) => asset.id === shown.id) ?? null,
  };
  const layers = { layerIdForPrimitive: () => undefined, get: () => undefined };
  const events = new Emitter<SceneEvents>();
  const manager = new SelectionManager(
    viewer as never,
    events,
    { pose: () => ({ altitude: 100 }), isMoving: false } as never,
    layers as never,
    sites as never,
  );
  managers.push(manager);
  const select = (
    manager as unknown as {
      selectTileset: (tileset: unknown, base: Record<string, unknown>) => Record<string, unknown>;
    }
  ).selectTileset.bind(manager);
  const card = () =>
    select({}, { longitude: -123.88, latitude: 46.1335, height: 10, terrainHeight: 4 });
  const show = (next: typeof A1) => (shown = next);
  return { card, show, events, manager };
}

it("a site selection names the shown epoch's asset", () => {
  const june = rig(A2).card();
  expect(june.kind).toBe("site");
  expect(String(june.sourceLabel)).toMatch(/^June phone video/);
  expect(june.observedAt).toBe("2026-06-01T15:00:00Z");
  const march = rig(A1).card();
  expect(String(march.sourceLabel)).toMatch(/^March drone flight/);
  expect(march.observedAt).toBe("2026-03-01T15:00:00Z");
});

it("an open site card follows another date being chosen", () => {
  const { card, show, events, manager } = rig(A2);
  // The card is open on the June model (as `select` leaves it).
  (manager as unknown as { current: unknown }).current = card();
  const cards: { sourceLabel?: string; observedAt?: string | null }[] = [];
  events.on("selection", (selection) => {
    if (selection) cards.push(selection);
  });
  show(A1);
  events.emit("epoch", { siteId: SITE_ID, epochId: E1.id });
  expect(cards.at(-1)?.sourceLabel).toMatch(/^March drone flight/);
  expect(cards.at(-1)?.observedAt).toBe("2026-03-01T15:00:00Z");
  // Another site's date says nothing about this card.
  events.emit("epoch", { siteId: "elsewhere", epochId: null });
  expect(cards).toHaveLength(1);
});
