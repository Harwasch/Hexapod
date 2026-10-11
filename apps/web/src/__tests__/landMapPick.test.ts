/**
 * Drawing land places each point where the person clicked. Photorealistic tiles hide the
 * globe, so a globe pick lands on the terrain beneath them and drifts from the clicked spot
 * as the camera tilts; the depth buffer is the surface the person actually sees.
 */
import {
  Cartesian2,
  Cartesian3,
  Cartographic,
  Ellipsoid,
  HeightReference,
  Math as CesiumMath,
  ScreenSpaceEventType,
  type Entity,
} from "cesium";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { LandMapController } from "@/cesium/LandMapController";
import type { LandPoint } from "@/state/land";

type Click = (event: { position: Cartesian2 }) => void;
const clicks: Click[] = [];

vi.mock("cesium", async (original) => {
  const cesium: Record<string, unknown> = await original();
  class FakeHandler {
    setInputAction(action: Click, type: ScreenSpaceEventType) {
      if (type === ScreenSpaceEventType.LEFT_CLICK) clicks.push(action);
    }
    destroy() {
      clicks.length = 0;
    }
  }
  return { ...cesium, ScreenSpaceEventHandler: FakeHandler };
});

/** The tiles' surface where the person clicked, and the hidden globe some way off. */
const VISIBLE = Cartesian3.fromDegrees(-119.9, 36.6, 350);
const HIDDEN = Cartesian3.fromDegrees(-119.905, 36.59, 0);
const CAMERA = Cartesian3.fromDegrees(-119.9, 36.59, 1_850);

function host(altitude: number, depth: () => Cartesian3 = () => VISIBLE.clone()) {
  const added: Entity.ConstructorOptions[] = [];
  const manager = {
    isDestroyed: false,
    camera: { pose: () => ({ altitude }) },
    scene: {
      canvas: document.createElement("canvas"),
      pickPositionSupported: true,
      sampleHeightSupported: false,
      pickPosition: depth,
      globe: { pick: () => HIDDEN.clone(), ellipsoid: Ellipsoid.WGS84 },
      drillPick: () => [],
      requestRender: () => undefined,
      cartesianToCanvasCoordinates: () => undefined,
      screenSpaceCameraController: { enableInputs: true },
    },
    viewer: {
      camera: {
        frustum: { near: 0.1 },
        positionWC: CAMERA,
        getPickRay: () => ({}),
        pickEllipsoid: (_: Cartesian2, ellipsoid: Ellipsoid) =>
          ellipsoid === Ellipsoid.WGS84 ? HIDDEN.clone() : VISIBLE.clone(),
      },
      entities: {
        add: (entity: Entity.ConstructorOptions) => (added.push(entity), entity),
        remove: () => true,
      },
    },
  } as unknown as CesiumSceneManager;
  return { manager, added };
}

function degrees(world: Cartesian3): LandPoint {
  const at = Cartographic.fromCartesian(world);
  return [CesiumMath.toDegrees(at.longitude), CesiumMath.toDegrees(at.latitude)];
}

function drawOnce(manager: CesiumSceneManager): LandPoint[] {
  const placed: LandPoint[] = [];
  const controller = new LandMapController(
    manager,
    (point) => placed.push(point),
    () => undefined,
  );
  controller.setMode("draw");
  clicks.at(-1)?.({ position: new Cartesian2(640, 360) });
  controller.show(null, placed, false);
  controller.destroy();
  return placed;
}

afterEach(() => {
  clicks.length = 0;
});

describe("LandMapController picking", () => {
  it("places a drawn point on the visible surface, not the globe hidden beneath it", () => {
    const [point] = drawOnce(host(1_500).manager);
    const [lon, lat] = degrees(VISIBLE);
    expect(point?.[0]).toBeCloseTo(lon, 7);
    expect(point?.[1]).toBeCloseTo(lat, 7);
  });

  it("uses the analytic globe pick from high altitude, where depth is imprecise", () => {
    const [point] = drawOnce(host(50_000).manager);
    const [lon, lat] = degrees(HIDDEN);
    expect(point?.[0]).toBeCloseTo(lon, 7);
    expect(point?.[1]).toBeCloseTo(lat, 7);
  });

  it("ignores depth left over from before the camera moved, which lands underground", () => {
    const stale = Cartesian3.fromDegrees(-119.91, 36.62, -3_500);
    const [point] = drawOnce(host(1_500, () => stale.clone()).manager);
    const [lon, lat] = degrees(HIDDEN);
    expect(point?.[0]).toBeCloseTo(lon, 7);
    expect(point?.[1]).toBeCloseTo(lat, 7);
  });

  it("draws the point's dot at the picked surface height, not clamped to the hidden globe", () => {
    const { manager, added } = host(1_500);
    drawOnce(manager);
    const dot = added.find((entity) => entity.id === "land-boundary:drawing:0");
    expect(dot?.point?.heightReference).toBe(HeightReference.NONE);
    const height = Cartographic.fromCartesian(dot?.position as Cartesian3).height;
    expect(height).toBeCloseTo(350, 3);
  });

  it("ignores the near-plane depth a dot writes, using the surface last seen instead", () => {
    let depth = VISIBLE;
    const { manager } = host(1_500, () => depth.clone());
    const placed: LandPoint[] = [];
    const controller = new LandMapController(
      manager,
      (point) => placed.push(point),
      () => undefined,
    );
    controller.setMode("draw");
    clicks.at(-1)?.({ position: new Cartesian2(640, 360) });
    // The second click lands on the first point's dot: its depth is at the camera.
    depth = Cartesian3.add(CAMERA, new Cartesian3(0.05, 0, 0), new Cartesian3());
    clicks.at(-1)?.({ position: new Cartesian2(640, 360) });
    controller.destroy();
    const [lon, lat] = degrees(VISIBLE);
    expect(placed[1]?.[0]).toBeCloseTo(lon, 7);
    expect(placed[1]?.[1]).toBeCloseTo(lat, 7);
  });
});
