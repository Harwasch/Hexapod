/**
 * Scene selection when the scan shown changes under it: another date of the site chosen
 * (SiteManager's `epoch`) hides one scan and shows another, with the camera standing still.
 * An object of the hidden scan is not selected any more, and the brush paints what is drawn
 * now, never the projection it kept of the scan that was.
 */
import { Matrix4 } from "cesium";
import { afterEach, describe, expect, it, vi } from "vitest";

import { registerPickSource } from "@/cesium/sceneSelect/pickSources";
import { SceneSelectController } from "@/cesium/sceneSelect/SceneSelectController";
import type * as SplatInstances from "@/cesium/splatInstances";
import { parseInstances, type InstancesDoc } from "@/lib/instances";
import type { PickTile } from "@/lib/splatPick";
import { useInstances } from "@/state/instances";
import { useSceneSelect } from "@/state/sceneSelect";

/** A scan of one object: a 9 x 9 grid of splats 5 m in front of the camera, in one tile. */
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

const OLD = scan("fnv1a32:81:0000aaaa", 1);
const NEW = scan("fnv1a32:81:0000bbbb", 2);
const DOCS: Record<string, InstancesDoc> = { e1: OLD.doc, e2: NEW.doc };

vi.mock("@/cesium/splatInstances", async (original) => ({
  ...(await original<typeof SplatInstances>()),
  instancesDocOf: (assetId: string) => DOCS[assetId],
}));

/** x, y in [-1, 1] straight to the screen, w = 10 - z (as the painting tests' view). */
const VIEW_PROJ = [10, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, -1, 0, 0, 0, 10];

function viewer() {
  const canvas = document.body.appendChild(document.createElement("canvas"));
  Object.defineProperty(canvas, "clientWidth", { value: 200 });
  Object.defineProperty(canvas, "clientHeight", { value: 200 });
  return {
    scene: {
      screenSpaceCameraController: { enableInputs: true },
      requestRender: () => undefined,
    },
    camera: {
      positionWC: { x: 0, y: 0, z: 0 },
      directionWC: { x: 0, y: 0, z: 1 },
      viewMatrix: Matrix4.IDENTITY,
      frustum: { fovy: Math.PI / 3, projectionMatrix: Matrix4.fromArray(VIEW_PROJ) },
    },
    canvas,
  };
}

/** Two dates of one site: `shown` says which is drawn (the other's source has no tiles). */
function twoDates() {
  let shown = "e1";
  const offs = (["e1", "e2"] as const).map((assetId) => {
    useInstances.getState().setTable(assetId, DOCS[assetId] ?? null);
    const tile = assetId === "e1" ? OLD.tile : NEW.tile;
    return registerPickSource(
      assetId,
      {
        renderer: "test",
        tiles: () => (shown === assetId ? [tile] : []),
        toWorld: () => Matrix4.IDENTITY,
      },
      0,
    );
  });
  return {
    show: (assetId: string) => (shown = assetId),
    off: () => offs.forEach((off) => off()),
  };
}

const STROKE = [
  { x: 90, y: 90 },
  { x: 110, y: 110 },
];

describe("scene selection across a site's dates", () => {
  let dates: ReturnType<typeof twoDates> | null = null;
  afterEach(() => {
    dates?.off();
    useSceneSelect.getState().clear();
    useSceneSelect.getState().setMode("pick");
    useInstances.setState({ assets: {}, gaps: {} });
  });

  it("the brush paints the date drawn now, with the camera where it was", () => {
    dates = twoDates();
    const controller = new SceneSelectController(viewer() as never);
    controller.setPainting(true);
    controller.paintStroke(STROKE, "replace");
    expect(useSceneSelect.getState().assetId).toBe("e1");
    dates.show("e2");
    controller.paintStroke(STROKE, "replace");
    expect(useSceneSelect.getState().assetId).toBe("e2");
    controller.destroy();
  });
});
