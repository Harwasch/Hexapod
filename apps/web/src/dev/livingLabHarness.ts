/**
 * The living lab's recording page (scripts/living-lab/harness.html): one scan drawn by
 * PlayCanvas over a bare CesiumWidget -- no globe, imagery or HUD, as dev/scanRendererHarness.ts
 * -- so a software renderer draws a frame in seconds rather than a minute. The lab's store
 * (`window.__livingLab`, state/livingLab.ts) sets the arms; `view` frames the camera in the
 * scan's own east/north/up metres.
 *
 * Nothing imports it, so it never reaches the production bundle.
 */

import {
  Cartesian3,
  CesiumWidget,
  Color,
  HeadingPitchRange,
  Math as CesiumMath,
  Matrix4,
} from "cesium";
import type { SiteAsset } from "@twin/contracts";

import { createSiteTileset } from "@/cesium/providers/tiles";
import { ScanRendererHost } from "@/cesium/scanView/ScanRendererHost";
import { installSplatDecoder } from "@/cesium/splatDecoder";
import { installSplatSorter } from "@/cesium/splatSorter";
import { useLivingLab } from "@/state/livingLab";

export async function startLivingLabHarness(container: HTMLElement, tilesetUrl: string) {
  const widget = new CesiumWidget(container, {
    baseLayer: false,
    requestRenderMode: false,
    msaaSamples: 1,
  });
  const { scene, camera } = widget;
  scene.globe.show = false;
  if (scene.skyBox) scene.skyBox.show = false;
  if (scene.skyAtmosphere) scene.skyAtmosphere.show = false;
  if (scene.sun) scene.sun.show = false;
  scene.backgroundColor = Color.fromCssColorString("#9fb3c8");
  installSplatSorter();
  installSplatDecoder();
  const asset = {
    representation: "gaussian-splat",
    source: { type: "3d-tiles-url", url: tilesetUrl },
    renderConfig: {},
  } as unknown as SiteAsset;
  const tileset = await createSiteTileset(asset, { maximumScreenSpaceError: 1 });
  tileset.show = false;
  tileset.preloadWhenHidden = false;
  scene.primitives.add(tileset);
  const host = new ScanRendererHost(widget, { preserveDrawingBuffer: true });
  host.setRenderer("playcanvas");
  host.setTarget({ key: "lab", tileset });
  const view = (
    target: [number, number, number],
    headingDeg: number,
    pitchDeg: number,
    rangeM: number,
  ) => {
    const at = Matrix4.multiplyByPoint(
      tileset.root.computedTransform,
      Cartesian3.fromArray(target),
      new Cartesian3(),
    );
    camera.lookAt(
      at,
      new HeadingPitchRange(
        CesiumMath.toRadians(headingDeg),
        CesiumMath.toRadians(pitchDeg),
        rangeM,
      ),
    );
    camera.lookAtTransform(Matrix4.IDENTITY);
  };
  view([0, 0, 0], 20, -20, tileset.boundingSphere.radius * 2);
  /** The Living Survey harness's framing: its bounding sphere's centre, at a heading and pitch. */
  const viewBounds = (headingDeg: number, pitchDeg: number, rangeM: number) => {
    camera.lookAt(
      tileset.boundingSphere.center,
      new HeadingPitchRange(
        CesiumMath.toRadians(headingDeg),
        CesiumMath.toRadians(pitchDeg),
        rangeM,
      ),
    );
    camera.lookAtTransform(Matrix4.IDENTITY);
  };
  return {
    widget,
    tileset,
    host,
    view,
    viewBounds,
    lab: useLivingLab,
    status: () => host.status(),
  };
}
