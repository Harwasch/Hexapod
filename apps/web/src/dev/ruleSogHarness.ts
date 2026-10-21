/**
 * A per-cell view rule on a scan PlayCanvas streams natively, as SOG with SH bands -- the
 * driver for the colour-only refresh test in e2e/compositorScene.spec.ts.
 *
 * PlayCanvas re-copies only the colours of SH splats into its work buffer as the camera moves
 * (gsplat-world `applyWorkBufferUpdates`, `renderColor`), and for a SOG resource that copy reads
 * the geometry back from the work buffer (`GSPLAT_WORKBUFFER_GEOMETRY`) and skips
 * `modifySplatRotationScale`. The rule's weight needs each splat's spread: read where only the
 * full copy hands it over, a colour-only refresh weighed every splat as a metre-wide one and
 * faded the scan for metres around a seam.
 *
 * The tree with SH bands (data/tiles/synthetic-tree-sh, its streamed package
 * synthetic-tree-sh-sog, splat-transform 3.8.0 by the backfill workflow's own commands) is the
 * scan; a rule is handed straight to the renderer (scanView/scanRule.ts): the scan owns the
 * cells west of x = 0, another date (with nothing drawn) those east of it, a seam down the
 * tree's middle. `cover` counts the overlay's covered pixels; `move` takes the camera away and
 * back, far enough for PlayCanvas's colour refresh and with its level of detail held, so the
 * pixels after are the same view as before.
 *
 * Loaded dynamically by the spec; nothing imports it, so it never reaches the production
 * bundle. Headless GL is SwiftShader: pixels are counted, not eyeballed.
 */

import {
  Cartesian3,
  Cesium3DTileset,
  CesiumWidget,
  Color,
  HeadingPitchRange,
  Math as CesiumMath,
  Matrix4,
  type Scene,
} from "cesium";
import * as pc from "playcanvas";

import { ScanRendererHost } from "@/cesium/scanView/ScanRendererHost";
import { publishScanRule, ruleOwnerRows, type ScanRuleView } from "@/cesium/scanView/scanRule";
import { NO_OWNER, ownerPlane } from "@/lib/compositor";

const ASSET = "tree-sh";

export interface RuleSogHarness {
  /** Whether the renderer streams the scan natively, and splats it draws. */
  status(): { native: boolean; gaussians: number };
  /** Covered pixels of the overlay (alpha over a quarter). */
  cover(): number;
  /** Takes the camera `metres` east and back, drawing on the way; resolves at rest. */
  move(metres: number): Promise<void>;
}

function nextFrame(scene: Scene): Promise<void> {
  return new Promise((resolve) => {
    const remove = scene.postRender.addEventListener(() => {
      remove();
      resolve();
    });
    scene.requestRender();
  });
}

export async function startRuleSogHarness(options: {
  container: HTMLElement;
  url: string;
}): Promise<RuleSogHarness> {
  const widget = new CesiumWidget(options.container, {
    baseLayer: false,
    requestRenderMode: true,
    msaaSamples: 1,
    contextOptions: { webgl: { preserveDrawingBuffer: true } },
  });
  const { scene } = widget;
  scene.globe.show = false;
  if (scene.skyBox) scene.skyBox.show = false;
  if (scene.skyAtmosphere) scene.skyAtmosphere.show = false;
  if (scene.sun) scene.sun.show = false;
  scene.backgroundColor = Color.fromCssColorString("#10141a");
  const tileset = await Cesium3DTileset.fromUrl(options.url);
  scene.primitives.add(tileset);
  tileset.show = false;
  const host = new ScanRendererHost(widget, { preserveDrawingBuffer: true });
  host.setRenderer("playcanvas");
  host.setTarget({ key: ASSET, tileset, assetId: ASSET });

  // The scan owns the cells west of x = 0 (source 0); another date, with nothing drawn, those
  // east (source 1): a seam down the tree's middle, 1 m cells.
  const plane = ownerPlane(
    "latest",
    [
      {
        source: 0,
        epochId: "scan",
        observedAt: "2026-03-01T00:00:00Z",
        inferred: false,
        boxes: [{ minX: -4, minY: -4, maxX: 0, maxY: 4 }],
      },
      {
        source: 1,
        epochId: "other",
        observedAt: "2026-06-01T00:00:00Z",
        inferred: false,
        boxes: [{ minX: 0, minY: -4, maxX: 4, maxY: 4 }],
      },
    ],
    null,
  );
  const view: ScanRuleView = {
    plane: [plane.i0, plane.j0, plane.ni, plane.nj],
    cellM: plane.cellM,
    owners: ruleOwnerRows(plane),
    sources: new Map(),
    native: 0,
    tinted: 0,
    reach: 2,
    enabled: true,
    dates: [],
    age: null,
  };
  if (plane.owners.includes(NO_OWNER)) throw new Error("the plane has a cell nobody owns");
  publishScanRule(ASSET, view);

  const centre = tileset.boundingSphere.center;
  const look = (eastM: number) => {
    const root = (tileset.root as { computedTransform?: Matrix4 }).computedTransform;
    const at = Matrix4.multiplyByPoint(
      root ?? Matrix4.IDENTITY,
      new Cartesian3(eastM, 0, 3),
      new Cartesian3(),
    );
    scene.camera.lookAt(
      eastM === 0 ? centre : at,
      new HeadingPitchRange(0, CesiumMath.toRadians(-35), 14),
    );
  };
  const settle = async (frames: number) => {
    for (let frame = 0; frame < frames; frame += 1) {
      await nextFrame(scene);
      await new Promise((resolve) => setTimeout(resolve, 15));
    }
  };
  look(0);
  for (let frame = 0; frame < 2000; frame += 1) {
    await nextFrame(scene);
    const status = host.status();
    if (status.active && status.native && status.gaussians > 0) break;
  }
  const app = pc.AppBase.getApplication();
  if (!app) throw new Error("no PlayCanvas app");
  // The level of detail is held, so the move re-copies colours alone, never a whole node.
  app.scene.gsplat.lodUpdateDistance = 1e6;
  await settle(80);

  return {
    status: () => {
      const status = host.status();
      return { native: status.native, gaussians: status.gaussians };
    },
    cover: () => {
      const canvas = document.querySelector<HTMLCanvasElement>("canvas[data-scan-renderer]");
      if (!canvas) return 0;
      const copy = document.createElement("canvas");
      copy.width = canvas.width;
      copy.height = canvas.height;
      const context = copy.getContext("2d");
      if (!context) return 0;
      context.drawImage(canvas, 0, 0);
      const data = context.getImageData(0, 0, copy.width, copy.height).data;
      let covered = 0;
      for (let p = 3; p < data.length; p += 4) if ((data[p] ?? 0) > 64) covered += 1;
      return covered;
    },
    move: async (metres) => {
      for (let step = 1; step <= 6; step += 1) {
        look((metres * step) / 6);
        await settle(3);
      }
      for (let step = 5; step >= 0; step -= 1) {
        look((metres * step) / 6);
        await settle(3);
      }
      await settle(60);
    },
  };
}
