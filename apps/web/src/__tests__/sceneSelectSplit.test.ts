/**
 * Scene selection with two dates compared (SiteManager `setCompareDates`, ADR 0017): each scan
 * is drawn on its own side of the divider only, so a click and a brush stroke answer from the
 * date drawn where they are, never from the one cut away there -- as `SplatCollider.raycast`
 * does for the other picks.
 */
import { Cartesian3, Matrix4, Ray, SplitDirection } from "cesium";
import { afterEach, describe, expect, it, vi } from "vitest";

import { registerPickSource } from "@/cesium/sceneSelect/pickSources";
import { SceneSelectController } from "@/cesium/sceneSelect/SceneSelectController";
import type * as SplatInstances from "@/cesium/splatInstances";
import { parseInstances, type InstancesDoc } from "@/lib/instances";
import type { PickTile } from "@/lib/splatPick";
import { useInstances } from "@/state/instances";
import { useSceneSelect } from "@/state/sceneSelect";

/** A scan of one object: a 9 x 9 grid of splats 5 m in front of the camera, the screen wide. */
function scan(checksum: string, id: number): { doc: InstancesDoc; tile: PickTile } {
  const positions: number[] = [];
  for (let i = 0; i < 9; i++)
    for (let j = 0; j < 9; j++) positions.push((i - 4) * 0.1, (j - 4) * 0.1, 5);
  const count = positions.length / 3;
  const doc = parseInstances({
    format: "hexapod.instances",
    version: 1,
    instances: [
      { id, parent: null, level: 0, splats: count, bounds: { min: [-1, -1, 4], max: [1, 1, 6] } },
    ],
    tiles: { [checksum]: [id, count] },
  });
  if (!doc) throw new Error("fixture did not parse");
  return {
    doc,
    tile: {
      checksum,
      count,
      positions: Float32Array.from(positions),
      radii: new Float32Array(count).fill(0.08),
      opacity: new Float32Array(count).fill(0.9),
    },
  };
}

const MARCH = scan("fnv1a32:81:0000aaaa", 1);
const JUNE = scan("fnv1a32:81:0000bbbb", 2);
const DOCS: Record<string, InstancesDoc> = { e1: MARCH.doc, e2: JUNE.doc };

vi.mock("@/cesium/splatInstances", async (original) => ({
  ...(await original<typeof SplatInstances>()),
  instancesDocOf: (assetId: string) => DOCS[assetId],
}));

/** x, y in [-1, 1] straight to the screen, w = 10 - z (as the painting tests' view). */
const VIEW_PROJ = [10, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, -1, 0, 0, 0, 10];
const SIZE = 200;

/** A 200 px square view split down the middle, its pick ray meeting z = 5 where VIEW_PROJ does. */
function viewer() {
  const canvas = document.body.appendChild(document.createElement("canvas"));
  Object.defineProperty(canvas, "clientWidth", { value: SIZE });
  Object.defineProperty(canvas, "clientHeight", { value: SIZE });
  return {
    scene: {
      screenSpaceCameraController: { enableInputs: true },
      requestRender: () => undefined,
      splitPosition: 0.5,
    },
    camera: {
      positionWC: { x: 0, y: 0, z: 0 },
      directionWC: { x: 0, y: 0, z: 1 },
      viewMatrix: Matrix4.IDENTITY,
      frustum: { fovy: Math.PI / 3, projectionMatrix: Matrix4.fromArray(VIEW_PROJ) },
      getPickRay: (p: { x: number; y: number }) => {
        const cx = (p.x / SIZE) * 2 - 1;
        const cy = 1 - (p.y / SIZE) * 2;
        const direction = Cartesian3.normalize(new Cartesian3(cx / 2, cy / 2, 5), new Cartesian3());
        return new Ray(new Cartesian3(0, 0, 0), direction);
      },
    },
    canvas,
  };
}

/** Two dates compared: March drawn left of the divider, June right of it, over one another. */
function compared() {
  const offs = (
    [
      ["e1", MARCH.tile, SplitDirection.LEFT],
      ["e2", JUNE.tile, SplitDirection.RIGHT],
    ] as const
  ).map(([assetId, tile, side]) => {
    useInstances.getState().setTable(assetId, DOCS[assetId] ?? null);
    return registerPickSource(
      assetId,
      {
        renderer: "test",
        tiles: () => [tile],
        toWorld: () => Matrix4.IDENTITY,
        split: () => side,
      },
      0,
    );
  });
  return () => offs.forEach((off) => off());
}

describe("scene selection with two dates compared", () => {
  let off: (() => void) | null = null;
  afterEach(() => {
    off?.();
    useSceneSelect.getState().clear();
    useSceneSelect.getState().setMode("pick");
    useInstances.setState({ assets: {}, gaps: {} });
  });

  it("a click selects from the date drawn on the side clicked", () => {
    off = compared();
    const controller = new SceneSelectController(viewer() as never);
    expect(controller.pickAt(60, 100)).toBe(true);
    expect(useSceneSelect.getState().assetId).toBe("e1");
    expect(controller.pickAt(140, 100)).toBe(true);
    expect(useSceneSelect.getState().assetId).toBe("e2");
    controller.destroy();
  });

  it("the brush paints the date drawn where the stroke is", () => {
    off = compared();
    const controller = new SceneSelectController(viewer() as never);
    controller.setPainting(true);
    controller.paintStroke(
      [
        { x: 130, y: 90 },
        { x: 150, y: 110 },
      ],
      "replace",
    );
    expect(useSceneSelect.getState().assetId).toBe("e2");
    controller.paintStroke(
      [
        { x: 50, y: 90 },
        { x: 70, y: 110 },
      ],
      "replace",
    );
    expect(useSceneSelect.getState().assetId).toBe("e1");
    controller.destroy();
  });
});
