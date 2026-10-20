/**
 * A bare CesiumJS page that draws the sunlight slab (`sunlightSlab.ts`) through the app's own
 * relight path -- the slot `attachSunlight` on the tileset, the stores it reads -- the driver
 * for e2e/sunlightScene.spec.ts.
 *
 * What it answers, in a real engine with the patch: does `splatSunlightColor` compile into the
 * splat vertex shader (with `czm_sunDirectionEC` declared by the engine because the text uses
 * it), find the discs' normals in the covariance texel, and light the east-facing side by the
 * sun of the scene clock -- brighter in the morning than in the evening -- while the round
 * splats beside it stay as they are by day? And with the sun off, or As captured on, is the
 * picture the photograph's?
 *
 * The camera stands east of the wall looking west, so the discs (south) are the left of the
 * screen and the round splats (north) the right. Loaded dynamically by the spec; nothing
 * imports it, so it never reaches the production bundle. Headless GL is SwiftShader: pixels
 * are measured, not eyeballed.
 */

import {
  Cartesian3,
  Cesium3DTileset,
  CesiumWidget,
  Color,
  Ellipsoid,
  HeadingPitchRange,
  JulianDate,
  Math as CesiumMath,
  Matrix3,
  Simon1994PlanetaryPositions,
  Transforms,
  type Scene,
} from "cesium";

import { colorChainOf, type ColorPrimitive } from "@/cesium/splatColor";
import { incrementalSplats, splatTilesetOf } from "@/cesium/splatInternals";
import { sunSine } from "@/cesium/splatSunlight";
import { attachSunlight } from "@/cesium/sunlightSlot";
import { SUN_AMBIENT, sunUpFactor } from "@/lib/splatNormals";
import { useAsCaptured } from "@/state/asCaptured";
import { useLighting, type SunMode } from "@/state/lighting";
import { useSunReport } from "@/state/sunReport";

export interface SlabView {
  /** Share of the canvas the wall covers. */
  coverage: number;
  /** Mean brightness (0-255, the mean of r, g, b) of covered pixels: the discs, the round splats. */
  flat: number;
  round: number;
  /** What lib/splatNormals.ts says the discs are multiplied by, at this sun. */
  predictedFlat: number;
  /** Sine of the sun's elevation at the wall, as the relight works it out. */
  sine: number;
  /** The colour chain of the slab's primitive. */
  parts: string[];
  /** What the relight reported drawing, if anything. */
  reported: { at: string; sine: number } | null;
}

export interface SunlightHarness {
  view(options: { sun: SunMode; at: string; asCaptured?: boolean }): Promise<SlabView>;
}

const BACKGROUND = "#10141a";

function nextFrame(scene: Scene): Promise<void> {
  return new Promise((resolve) => {
    const remove = scene.postRender.addEventListener(() => {
      remove();
      resolve();
    });
    scene.requestRender();
  });
}

/** The sun's direction, Earth-fixed, by the engine's arithmetic (as `sunSine`). */
function sunDirection(time: JulianDate): Cartesian3 {
  const m = new Matrix3();
  Transforms.computeIcrfToCentralBodyFixedMatrix(time, m);
  const p = Simon1994PlanetaryPositions.computeSunPositionInEarthInertialFrame(time);
  return Cartesian3.normalize(Matrix3.multiplyByVector(m, p, p), p);
}

export async function startSunlightHarness(options: {
  container: HTMLElement;
  url: string;
}): Promise<SunlightHarness> {
  const widget = new CesiumWidget(options.container, {
    baseLayer: false,
    requestRenderMode: true,
    msaaSamples: 1,
    contextOptions: { webgl: { preserveDrawingBuffer: true } },
  });
  const { scene, clock } = widget;
  scene.globe.show = false;
  if (scene.skyBox) scene.skyBox.show = false;
  if (scene.skyAtmosphere) scene.skyAtmosphere.show = false;
  if (scene.sun) scene.sun.show = false;
  scene.backgroundColor = Color.fromCssColorString(BACKGROUND);

  const tileset = await Cesium3DTileset.fromUrl(options.url, {
    maximumScreenSpaceError: 4,
    skipLevelOfDetail: false,
  });
  incrementalSplats(tileset, 0);
  scene.primitives.add(tileset);
  // The app's path: the slot, which loads the relight when the sun is on and As captured off.
  attachSunlight(tileset, scene, "slab");
  const primitive = (): ColorPrimitive | undefined =>
    splatTilesetOf(tileset).gaussianSplatPrimitive;

  function measure(): Pick<SlabView, "coverage" | "flat" | "round"> {
    const canvas = scene.canvas;
    const copy = document.createElement("canvas");
    copy.width = canvas.width;
    copy.height = canvas.height;
    const context = copy.getContext("2d");
    if (!context) return { coverage: 0, flat: 0, round: 0 };
    context.drawImage(canvas, 0, 0);
    const data = context.getImageData(0, 0, copy.width, copy.height).data;
    const bg = Color.fromCssColorString(BACKGROUND);
    const [br, bgG, bb] = [bg.red * 255, bg.green * 255, bg.blue * 255];
    const half = copy.width / 2;
    let covered = 0;
    const sum = { left: 0, right: 0 };
    const count = { left: 0, right: 0 };
    for (let p = 0; p < data.length; p += 4) {
      const r = data[p] ?? 0;
      const g = data[p + 1] ?? 0;
      const b = data[p + 2] ?? 0;
      if (Math.abs(r - br) + Math.abs(g - bgG) + Math.abs(b - bb) <= 12) continue;
      covered += 1;
      const side = (p / 4) % copy.width < half ? "left" : "right";
      sum[side] += (r + g + b) / 3;
      count[side] += 1;
    }
    return {
      coverage: covered / (data.length / 4),
      flat: count.left > 0 ? sum.left / count.left : 0,
      round: count.right > 0 ? sum.right / count.right : 0,
    };
  }

  return {
    async view({ sun, at, asCaptured = false }) {
      useAsCaptured.getState().set(asCaptured);
      useLighting.getState().set(sun);
      const time = JulianDate.fromIso8601(at);
      clock.shouldAnimate = false;
      clock.currentTime = time;
      // East of the wall, looking west: the discs (south) left, the round splats right.
      scene.camera.lookAt(
        tileset.boundingSphere.center,
        new HeadingPitchRange(CesiumMath.toRadians(270), CesiumMath.toRadians(-4), 9),
      );
      for (let frame = 0; frame < 600 && !(tileset.tilesLoaded && primitive()); frame += 1) {
        await nextFrame(scene);
      }
      // The relight loads on first use; the draw command is rebuilt on the next render.
      for (let frame = 0; frame < 30; frame += 1) await nextFrame(scene);
      const centre = tileset.boundingSphere.center;
      const sine = sunSine(time, centre);
      const east = Transforms.eastNorthUpToFixedFrame(centre, Ellipsoid.WGS84);
      const eastAxis = new Cartesian3(east[0], east[1], east[2]);
      const lambert = Math.max(Cartesian3.dot(eastAxis, sunDirection(time)), 0);
      const relit = sun !== "off" && !asCaptured;
      const target = primitive();
      return {
        ...measure(),
        predictedFlat: relit ? SUN_AMBIENT + (1 - SUN_AMBIENT) * lambert * sunUpFactor(sine) : 1,
        sine,
        parts: (target ? (colorChainOf(target)?.parts ?? []) : []).map((p) => p.colorFunction),
        reported: useSunReport.getState().reports.slab ?? null,
      };
    },
  };
}
