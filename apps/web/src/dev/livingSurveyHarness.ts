/**
 * A bare CesiumJS page that loads one splat tileset and deforms it — the driver for the Living
 * Survey's end-to-end check.
 *
 * It exists because `SplatDeformer` is the one piece of this feature that cannot be proved
 * headlessly: the texel arithmetic, the frame round trip and the refusal logic are all unit
 * tested against a fake primitive, but whether the engine's real packed buffer, the real
 * addressing parameters and a real `Texture.copyFrom` compose into a tree that moves can only be
 * answered by a browser. This is the smallest page that asks that question — no API, no app
 * shell, no store.
 *
 * Loaded dynamically by `e2e/livingSurvey.spec.ts`; nothing imports it, so it never reaches the
 * production bundle. It drives `apply()` by hand rather than from `scene.preUpdate` — the
 * animation hook belongs to `LivingSurveyManager`, and a test that installs its own is testing
 * the deformer rather than the manager.
 */

import {
  Cartesian3,
  Cesium3DTileset,
  CesiumWidget,
  Color,
  HeadingPitchRange,
  Math as CesiumMath,
  type Scene,
} from "cesium";

import { deform, flutterField, parseRig, type MotionRig, type WindSettings } from "@twin/world";

import { SplatDeformer, type DeformerStatus } from "@/cesium/SplatDeformer";
import { installSplatTextureInterception } from "@/cesium/splatCapture";
import { splatCaptureCount } from "@/cesium/splatCaptureRegistry";
import { splatTilesetOf } from "@/cesium/splatInternals";

export interface LivingSurveyHarnessOptions {
  readonly container: HTMLElement;
  /** A `tileset.json` for a single-tile Gaussian splat capture. */
  readonly tilesetUrl: string;
  /** The matching `rig.json`. */
  readonly rigUrl: string;
}

/** What the Playwright spec drives. Everything returns plain JSON so it crosses the bridge. */
export interface LivingSurveyHarness {
  /** Applies the rig at time `t` under `wind`, then renders one frame. */
  step(t: number, wind: WindSettings): Promise<DeformerStatus>;
  /** Renders frames until the deformer attaches or `timeoutMs` elapses. */
  waitUntilReady(timeoutMs: number): Promise<DeformerStatus>;
  status(): DeformerStatus & { captures: number; numSplatsLoaded: number };
  /** Largest distance any splat has moved from its measured position, metres. */
  displacementM(t: number, wind: WindSettings): number;
  /** One `apply()` with no render, for timing the CPU and upload-submit cost alone. */
  applyOnly(t: number, wind: WindSettings): DeformerStatus;
}

/** Resolves on the next completed frame. */
function nextFrame(scene: Scene): Promise<void> {
  return new Promise((resolve) => {
    const remove = scene.postRender.addEventListener(() => {
      remove();
      resolve();
    });
    scene.requestRender();
  });
}

export async function startLivingSurveyHarness(
  options: LivingSurveyHarnessOptions,
): Promise<LivingSurveyHarness> {
  // Before anything can load: the interception only sees calls made after it is installed.
  installSplatTextureInterception();

  const widget = new CesiumWidget(options.container, {
    baseLayer: false,
    // Continuous rendering. SwiftShader frames here run 740–1730 ms, and a request-render scene
    // that stops asking never fires `postRender` for the listener to resolve on.
    requestRenderMode: false,
    msaaSamples: 1,
  });
  const { scene } = widget;
  scene.globe.show = false;
  if (scene.skyBox) scene.skyBox.show = false;
  if (scene.skyAtmosphere) scene.skyAtmosphere.show = false;
  if (scene.sun) scene.sun.show = false;
  scene.backgroundColor = Color.fromCssColorString("#10141a");

  const rigResponse = await fetch(options.rigUrl);
  const rig: MotionRig = parseRig(await rigResponse.text());

  const tileset = await Cesium3DTileset.fromUrl(options.tilesetUrl, {
    maximumScreenSpaceError: 1,
  });
  scene.primitives.add(tileset);

  const bounds = tileset.boundingSphere;
  scene.camera.lookAt(
    bounds.center,
    new HeadingPitchRange(CesiumMath.toRadians(35), CesiumMath.toRadians(-12), bounds.radius * 3.2),
  );

  const internals = splatTilesetOf(tileset);
  const deformer = new SplatDeformer({ tileset: internals, rig });

  async function step(t: number, wind: WindSettings): Promise<DeformerStatus> {
    const status = deformer.apply(deform(rig, t, wind), flutterField(rig, t, wind));
    await nextFrame(scene);
    return status;
  }

  return {
    step,
    applyOnly(t: number, wind: WindSettings): DeformerStatus {
      return deformer.apply(deform(rig, t, wind), flutterField(rig, t, wind));
    },
    async waitUntilReady(timeoutMs: number): Promise<DeformerStatus> {
      const deadline = Date.now() + timeoutMs;
      let status = deformer.status;
      while (Date.now() < deadline && status.phase !== "ready" && status.phase !== "refused") {
        status = await step(0, { strength: 0, bearingDeg: 0 });
      }
      return status;
    },
    status() {
      return {
        ...deformer.status,
        captures: splatCaptureCount(),
        numSplatsLoaded: internals.gaussianSplatPrimitive?._numSplats ?? 0,
      };
    },
    displacementM(t: number, wind: WindSettings): number {
      const canonical = deformer.canonicalPositions;
      const assignment = deformer.assignment;
      if (canonical === undefined || assignment === undefined) return 0;
      const transforms = deform(rig, t, wind);
      let worst = 0;
      const point = new Cartesian3();
      for (let i = 0; i < assignment.length; i += 1) {
        const transform = transforms[assignment[i] ?? 0];
        if (transform === undefined) continue;
        const x = canonical[i * 3] ?? 0;
        const y = canonical[i * 3 + 1] ?? 0;
        const z = canonical[i * 3 + 2] ?? 0;
        const [qx, qy, qz, qw] = transform.rotation;
        // q ⊗ v ⊗ q⁻¹, written out so this stays independent of @twin/world's own helpers.
        const ux = qy * z - qz * y;
        const uy = qz * x - qx * z;
        const uz = qx * y - qy * x;
        const tx = ux + qw * x;
        const ty = uy + qw * y;
        const tz = uz + qw * z;
        point.x = x + 2 * (qy * tz - qz * ty) + transform.translation[0];
        point.y = y + 2 * (qz * tx - qx * tz) + transform.translation[1];
        point.z = z + 2 * (qx * ty - qy * tx) + transform.translation[2];
        const moved = Math.hypot(point.x - x, point.y - y, point.z - z);
        if (moved > worst) worst = moved;
      }
      return worst;
    },
  };
}
