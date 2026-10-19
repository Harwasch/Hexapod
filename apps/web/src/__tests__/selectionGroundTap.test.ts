/**
 * A click on the ground while a layer takes taps (the fog, Show once), through the real
 * SelectionManager on a stub scene: where the tap lands when the depth pick is not ready yet,
 * which of a Land's mapped features a tap answers for, and a later click superseding a tap
 * still waiting for its frame.
 */

import {
  Cartesian2,
  Cartesian3,
  Ellipsoid,
  Entity,
  EntityCollection,
  Event as CesiumEvent,
  JulianDate,
  PolygonHierarchy,
  Ray,
  ScreenSpaceEventType,
  type ScreenSpaceEventHandler,
} from "cesium";
import { afterEach, expect, it, vi } from "vitest";

import { SelectionManager } from "@/cesium/SelectionManager";
import type { SceneEvents } from "@/cesium/types";
import { Emitter } from "@/lib/emitter";

/** Where the ground really is, and where the bare ellipsoid behind it says (90 m off). */
const GROUND = { longitude: -120.04, latitude: 34.698 };
const ELLIPSOID = { longitude: -120.041, latitude: 34.6975 };
const ring = Cartesian3.fromDegreesArray([-120.05, 34.69, -120.03, 34.69, -120.03, 34.7]);

function rig(picked?: unknown) {
  const state = { picked, drawn: false };
  const postRender = new CesiumEvent();
  const scene = {
    pickAsync: vi.fn(() => Promise.resolve(state.picked)),
    drillPick: vi.fn(() => []),
    pickPositionSupported: true,
    // The depth answers only once a frame has drawn the ground there.
    pickPosition: vi.fn(() =>
      state.drawn ? Cartesian3.fromDegrees(GROUND.longitude, GROUND.latitude, 250) : undefined,
    ),
    globe: { pick: vi.fn(() => undefined), ellipsoid: Ellipsoid.WGS84 },
    postRender,
    requestRender: vi.fn(),
    isDestroyed: () => false,
  };
  const viewer = {
    canvas: document.createElement("canvas"),
    scene,
    camera: {
      getPickRay: () => new Ray(),
      pickEllipsoid: () => Cartesian3.fromDegrees(ELLIPSOID.longitude, ELLIPSOID.latitude, 0),
    },
    clock: { currentTime: JulianDate.now() },
    entities: new EntityCollection(),
  };
  const events = new Emitter<SceneEvents>();
  const camera = { pose: () => ({ altitude: 4_400 }), isMoving: false };
  const manager = new SelectionManager(
    viewer as never,
    events,
    camera as never,
    {} as never,
    {} as never,
  );
  const taps: SceneEvents["ground-tap"][] = [];
  const features: string[] = [];
  events.on("ground-tap", (tap) => taps.push(tap));
  events.on("land-feature-select", ({ id }) => features.push(id));
  const handler = (manager as unknown as { handler: ScreenSpaceEventHandler }).handler;
  const onClick = handler.getInputAction(ScreenSpaceEventType.LEFT_CLICK) as (event: {
    position: Cartesian2;
  }) => void;
  const click = (x = 10, y = 20) => onClick({ position: new Cartesian2(x, y) });
  managers.push(manager);
  return { state, scene, manager, taps, features, click };
}

const managers: SelectionManager[] = [];

afterEach(() => {
  for (const manager of managers.splice(0)) manager.destroy();
  vi.useRealTimers();
});

function expectAt(tap: SceneEvents["ground-tap"] | undefined, at: typeof GROUND): void {
  expect(tap?.longitude).toBeCloseTo(at.longitude, 6);
  expect(tap?.latitude).toBeCloseTo(at.latitude, 6);
}

const landArea = () =>
  new Entity({
    id: "land-context:inventory/oak#0",
    polygon: { hierarchy: new PolygonHierarchy(ring) },
  });
const landLine = () =>
  new Entity({ id: "land-context:inventory/fence", polyline: { positions: ring } });

it("asks again after a fresh frame when only the bare ellipsoid answered, and taps there", async () => {
  const { state, scene, manager, taps, click } = rig();
  manager.wantGroundTaps();
  click();
  await vi.waitFor(() => expect(scene.requestRender).toHaveBeenCalled());
  // Waiting for the frame: the ellipsoid's guess, 90 m off, is not sent.
  expect(taps).toHaveLength(0);
  state.drawn = true;
  scene.postRender.raiseEvent(scene);
  await vi.waitFor(() => expect(taps).toHaveLength(1));
  expectAt(taps[0], GROUND);
  expect(taps[0]).toMatchObject({ x: 10, y: 20 });
});

it("falls back to the ellipsoid when no frame comes within half a second", async () => {
  vi.useFakeTimers();
  const { scene, manager, taps, click } = rig();
  manager.wantGroundTaps();
  click();
  // The pick resolves (a microtask), then the tap waits for a frame.
  await vi.advanceTimersByTimeAsync(0);
  expect(scene.requestRender).toHaveBeenCalled();
  await vi.advanceTimersByTimeAsync(499);
  expect(taps).toHaveLength(0);
  await vi.advanceTimersByTimeAsync(1);
  expect(taps).toHaveLength(1);
  expectAt(taps[0], ELLIPSOID);
});

it("takes a tap at once when the depth already answers", async () => {
  const { state, scene, manager, taps, click } = rig();
  state.drawn = true;
  manager.wantGroundTaps();
  click();
  await vi.waitFor(() => expect(taps).toHaveLength(1));
  expectAt(taps[0], GROUND);
  expect(scene.requestRender).not.toHaveBeenCalled();
});

it("makes a Land's mapped area a tap on the claimed ground, and selects it elsewhere", async () => {
  const { state, manager, taps, features, click } = rig({ id: landArea() });
  state.drawn = true;
  const release = manager.wantGroundTaps(() => true);
  click();
  await vi.waitFor(() => expect(taps).toHaveLength(1));
  expect(features).toEqual([]);
  release();
  // The layer claims none of this ground: the oak woodland opens as a feature.
  manager.wantGroundTaps(() => false);
  click();
  await vi.waitFor(() => expect(features).toEqual(["oak"]));
  expect(taps).toHaveLength(1);
});

it("taps over a Land's lines only for a tool in hand", async () => {
  const { state, manager, taps, features, click } = rig({ id: landLine() });
  state.drawn = true;
  const release = manager.wantGroundTaps(() => true);
  click();
  await vi.waitFor(() => expect(features).toEqual(["fence"]));
  expect(taps).toHaveLength(0);
  release();
  manager.wantGroundTaps(() => true, { overFeatures: true });
  click();
  await vi.waitFor(() => expect(taps).toHaveLength(1));
  expect(features).toEqual(["fence"]);
});

it("drops a tap still waiting for its frame when a later click comes", async () => {
  const { state, scene, manager, taps, click } = rig();
  manager.wantGroundTaps();
  click(10, 20);
  await vi.waitFor(() => expect(scene.requestRender).toHaveBeenCalled());
  // The ground draws; the second click is answered at once.
  state.drawn = true;
  click(30, 40);
  await vi.waitFor(() => expect(taps).toHaveLength(1));
  expect(taps[0]).toMatchObject({ x: 30, y: 40 });
  // The first click's frame comes: it is not the latest any more, and says nothing.
  scene.postRender.raiseEvent(scene);
  await new Promise((resolve) => setTimeout(resolve, 10));
  expect(taps).toHaveLength(1);
});
