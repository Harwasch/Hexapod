/**
 * A real `CesiumSceneManager`, one rigged site, and a hand on the clock.
 *
 * `livingSurveyHarness.ts` drives `SplatDeformer` directly, which answers "do the texels land in
 * the right places". This one answers the questions that belong to `LivingSurveyManager` and can
 * only be asked of the whole scene: does the interception installed at scene construction see
 * the tileset that loads afterwards, does `scene.preUpdate` drive the tick in
 * `requestRenderMode`, does an idle scene actually stop rendering, and does `snapshot()`
 * photograph the measurement rather than the simulation.
 *
 * The scene clock is frozen and stepped by hand, which is the whole point of taking time from
 * `viewer.clock.currentTime`: a given scene time always produces the same frame, so a test can
 * compare two of them.
 *
 * Loaded dynamically by `e2e/livingSurvey.spec.ts`; nothing imports it, so it never reaches the
 * production bundle.
 */

import { Cartesian3, HeadingPitchRange, JulianDate, Math as CesiumMath } from "cesium";

import type { Site, SiteSummary } from "@twin/contracts";
import type { WindSettings } from "@twin/world";

import { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { LIVING_EPOCH_ISO, type LivingMotionModel } from "@/cesium/LivingSurveyManager";
import {
  digestSplatPositions,
  findSplatCapture,
  splatCaptureCount,
} from "@/cesium/splatCaptureRegistry";
import { splatTilesetOf, type SplatPrimitive } from "@/cesium/splatInternals";
import { positionWordOffset, splatTextureLayout } from "@/cesium/splatTexels";
import type { LivingSurveyStatus } from "@/state/living";

export interface LivingSceneHarnessOptions {
  readonly container: HTMLElement;
  /** A single-tile Gaussian splat `tileset.json` for a capture the catalog gives a rig. */
  readonly tilesetUrl: string;
  /** The capture slug. Identity only — since A9 the rig is claimed by the asset, not the slug. */
  readonly slug: string;
  /**
   * The catalog's rig claim for that asset, relative to `tilesetUrl`, as
   * `renderConfig.rigUrl` carries it. Defaults to where `tools/captures` puts one.
   */
  readonly rigUrl?: string | null;
  readonly longitude: number;
  readonly latitude: number;
  /** The build's `VITE_SPLAT_GPU_MOTION`, as `CesiumSceneManager` takes it. Default true. */
  readonly splatGpuMotion?: boolean;
}

/**
 * The attribute texture read back from the GPU, against the packed buffer the engine uploaded
 * into it. `mismatches` counts words that differ from the engine's own bytes (zero padding
 * included), so 0 is "exactly the measured pose" and anything else is displaced positions.
 */
export interface TextureReadback {
  readonly words: number;
  readonly mismatches: number;
  /** Mismatching words that are splat positions, as against other attribute words. */
  readonly positionMismatches: number;
  /** A GL error or a missing capture, when the comparison could not be made at all. */
  readonly error: string | null;
}

/**
 * Whether a manager's per-tick `requestRender()` is honoured in the same animation frame.
 *
 * `Scene.render` raises `preUpdate` *before* it decides whether to render, from
 * `_renderRequested`, so a request made in the tick is this frame's — unless something
 * between the two consumed or ignored it. `requestedAfterTick` counts ticks that ended with the
 * flag set; `renderedAfterRequest` those of them that went on to render. `meanTickMs` is the
 * mean interval between ticks, which is the animation frame interval.
 */
export interface CadenceProbe {
  readonly ticks: number;
  readonly requestedAfterTick: number;
  readonly renderedAfterRequest: number;
  readonly meanTickMs: number;
}

/**
 * Three counters, because only one of them is the feature's.
 *
 * `ticks` runs at the animation frame rate whatever else happens, so it is what a test waits on.
 * `livingRequests` counts `scene.requestRender()` calls made *inside the Living Survey's own
 * tick* — bracketed between two `preUpdate` listeners, one registered before the manager's and
 * one after — and that is the number that must stop at zero wind.
 *
 * `renders` and `renderRequests` are reported but are not measures of this feature. A scene has
 * many other reasons to redraw: `SiteManager` re-applies its screen-space error on a timer, and
 * in this harness the globe is hidden, so its quadtree never updates, `globe.tilesLoaded` never
 * turns true, and `Scene.render` sets `_renderRequested` directly on every frame for as long as
 * that is so. In the real app the globe settles; here it cannot, and counting its frames as the
 * deformer's would be measuring the harness.
 */
export interface FrameCounts {
  readonly ticks: number;
  readonly livingRequests: number;
  readonly renderRequests: number;
  readonly renders: number;
}

export interface LivingSceneHarness {
  /** Moves the scene clock to `t` seconds after the Living Survey epoch. */
  setTime(t: number): void;
  setWind(wind: WindSettings): void;
  /** The viewer's "Motion on GPU" setting. */
  setGpuMotion(enabled: boolean): void;
  /** Whether the patched primitive currently carries a motion hook. */
  hookInstalled(): boolean;
  /** The attribute texture, read back and compared with the engine's own packed bytes. */
  readAttributeTexture(): TextureReadback;
  /** Watches `n` ticks and reports whether each tick's render request was a rendered frame. */
  cadence(n: number, timeoutMs?: number): Promise<CadenceProbe>;
  /** Living Mode's sidecar model (`auto`) or the legacy nine-sine model, for comparison clips. */
  setMotionModel(model: LivingMotionModel): void;
  status(): LivingSurveyStatus & { captures: number };
  counts(): FrameCounts;
  /** Waits until `ticks` has advanced by `n`, whether or not anything renders. */
  waitTicks(n: number, timeoutMs?: number): Promise<FrameCounts>;
  /**
   * Renders one frame at the current scene time and returns a digest of its pixels. Does *not*
   * hold the measured pose, so this is what the survey actually looks like right now.
   */
  grabFrameHash(): Promise<string>;
  /** The same frame as {@link grabFrameHash}, as a PNG data URL — for comparison clips. */
  grabFrame(): Promise<string>;
  /**
   * `CesiumSceneManager.snapshot()`, with a witness.
   *
   * The digest of the picture, plus whether any deformer still held displaced positions in the
   * frame the picture was taken from. The flag is the assertable half: pixels cannot settle
   * this, because the splat sorter runs asynchronously off canonical positions the deformer
   * never touches, so two frames standing at *identical* positions can still be drawn in
   * different orders. The flag is read from inside that frame and answers exactly what
   * `holdMeasuredPose` promises.
   */
  snapshotProbe(): Promise<{ hash: string; displacedDuringCapture: boolean }>;
  /** Live listener counts, so `destroy()` can be shown to have released them. */
  probe(): { preUpdate: number };
  destroyLiving(): void;
  waitUntilReady(timeoutMs: number): Promise<LivingSurveyStatus>;
}

/**
 * FNV-1a over a data URL. Frames are compared for *equality*, so a digest carries exactly the
 * information the test needs and keeps a megabyte of base64 off the Playwright bridge.
 */
function digest(text: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) h = Math.imul(h ^ text.charCodeAt(i), 0x01000193) >>> 0;
  return `${text.length}:${h.toString(16).padStart(8, "0")}`;
}

const SITE_ID = "00000000-0000-4000-8000-000000000001";
const ASSET_ID = "00000000-0000-4000-8000-000000000002";

function siteFixture(options: LivingSceneHarnessOptions): { site: Site; summary: SiteSummary } {
  const { longitude, latitude } = options;
  const d = 0.0001;
  const ring = [
    [longitude - d, latitude - d],
    [longitude + d, latitude - d],
    [longitude + d, latitude + d],
    [longitude - d, latitude + d],
    [longitude - d, latitude - d],
  ];
  const boundary = { type: "Polygon", coordinates: [ring] };
  const timestamps = { createdAt: "2026-01-01T00:00:00Z", updatedAt: "2026-01-01T00:00:00Z" };
  const site = {
    id: SITE_ID,
    slug: options.slug,
    name: "Living Survey fixture",
    description: "",
    boundary,
    centroid: { longitude, latitude, height: 0 },
    areaM2: 400,
    thumbnailUrl: null,
    metadata: {},
    attribution: [],
    license: null,
    assets: [
      {
        id: ASSET_ID,
        siteId: SITE_ID,
        provider: "3d-tiles-url",
        name: "Gaussian splat (procedural)",
        representation: "gaussian-splat",
        source: { type: "3d-tiles-url", url: options.tilesetUrl },
        footprint: null,
        observedAt: null,
        validFrom: null,
        validTo: null,
        resolution: null,
        crs: null,
        license: null,
        attribution: [],
        provenance: null,
        renderConfig: {
          maximumScreenSpaceError: 1,
          pointCloudShading: null,
          // No clipping and no terrain sample: the tileset's own placement is what the rig's
          // checksum was taken against, and nothing here should move it.
          clipsWorld: false,
          clipFootprint: "catalog",
          heightOffsetM: 0,
          // The catalog's claim that this capture can move. It used to be an entry in
          // LIVING_RIGS keyed by slug; since A9 it rides on the asset, so a harness that
          // omitted it would produce a site the Living Survey correctly declines.
          rigUrl: options.rigUrl === undefined ? "../source/rig.json" : options.rigUrl,
        },
        defaultVisible: true,
        ...timestamps,
      },
    ],
    cameraBookmarks: [],
    ...timestamps,
  };
  const summary = {
    id: SITE_ID,
    slug: options.slug,
    name: site.name,
    description: "",
    centroid: site.centroid,
    areaM2: site.areaM2,
    thumbnailUrl: null,
    representations: ["gaussian-splat"],
    latestObservedAt: null,
    quality: { resolutionDescription: null, groundSampleDistanceM: null },
    ...timestamps,
  };
  return { site: site as unknown as Site, summary: summary as unknown as SiteSummary };
}

export async function startLivingSceneHarness(
  options: LivingSceneHarnessOptions,
): Promise<LivingSceneHarness> {
  // The real thing, including the texture interception it installs before anything can load.
  const scene = new CesiumSceneManager(options.container, {
    ionToken: undefined,
    home: { longitude: options.longitude, latitude: options.latitude, height: 40 },
    splatGpuMotion: options.splatGpuMotion,
  });
  const viewer = scene.viewer;
  const gl = viewer.scene;

  // Nothing but the tree in the frame: the globe, the sky and the atmosphere all change between
  // frames on their own, and this harness compares frames for equality.
  gl.globe.show = false;
  if (gl.skyAtmosphere) gl.skyAtmosphere.show = false;
  if (gl.skyBox) gl.skyBox.show = false;
  if (gl.sun) gl.sun.show = false;
  gl.fog.enabled = false;

  // Pin the render settings. The adaptive ladder is the only thing that changes resolution or
  // MSAA, and `sharpenAtRest` deliberately does not fire while the scene animates (S3b) — so
  // with it on, a frame at rest and a frame under wind would differ in how they were rendered
  // as well as in where the splats are, and this harness compares frames for equality.
  scene.performance.configure({
    preset: "balanced",
    manualScreenSpaceError: null,
    adaptive: false,
  });

  // The clock is ours. `shouldAnimate` off means `currentTime` only moves when we move it, so
  // every frame below is a pure function of a number this harness chose.
  viewer.clock.shouldAnimate = false;
  viewer.clock.currentTime = JulianDate.fromIso8601(LIVING_EPOCH_ISO);

  let ticks = 0;
  let renders = 0;
  let renderRequests = 0;
  let livingRequests = 0;
  let atTickStart = 0;
  // The cadence probe's state (see `CadenceProbe`).
  const internals = gl as unknown as { _renderRequested?: boolean };
  let probing = false;
  let requestedAfterTick = 0;
  let renderedAfterRequest = 0;
  let awaitingRender = false;
  let lastTickAt: number | null = null;
  const tickIntervals: number[] = [];
  // First listener on preUpdate, so it runs before the manager's tick (Cesium raises listeners
  // in registration order, and the manager registers its own only once a site attaches).
  gl.preUpdate.addEventListener(() => {
    ticks += 1;
    atTickStart = renderRequests;
    awaitingRender = false;
    const now = performance.now();
    if (probing && lastTickAt !== null) tickIntervals.push(now - lastTickAt);
    lastTickAt = now;
  });
  gl.postRender.addEventListener(() => {
    renders += 1;
    if (awaitingRender) renderedAfterRequest += 1;
    awaitingRender = false;
  });
  const requestRender = gl.requestRender.bind(gl);
  gl.requestRender = () => {
    renderRequests += 1;
    requestRender();
  };
  /** Closes the bracket around the manager's tick. Registered once the manager's is in place. */
  let closeBracket: (() => void) | null = null;
  function armLivingRequestCounter(): void {
    closeBracket?.();
    closeBracket = gl.preUpdate.addEventListener(() => {
      // A request from inside the manager's tick, still pending when the tick ends.
      if (probing && renderRequests > atTickStart && internals._renderRequested === true) {
        requestedAfterTick += 1;
        awaitingRender = true;
      }
      livingRequests += renderRequests - atTickStart;
    });
  }

  const { site, summary } = siteFixture(options);
  scene.sites.setCatalog([summary], () => Promise.resolve(site));
  await scene.sites.activate(SITE_ID);
  const sphere = await scene.sites.boundingSphere(site);
  if (sphere) {
    viewer.camera.lookAt(
      sphere.center,
      new HeadingPitchRange(
        CesiumMath.toRadians(35),
        CesiumMath.toRadians(-10),
        Math.max(sphere.radius * 3.2, 12),
      ),
    );
  } else {
    viewer.camera.lookAt(
      Cartesian3.fromDegrees(options.longitude, options.latitude, 3),
      new HeadingPitchRange(0, CesiumMath.toRadians(-10), 25),
    );
  }

  /** Resolves on the next completed frame, asking for one from inside the listener.
   *
   * SwiftShader frames here run 740–1730 ms and the scene renders on request, so a listener
   * that does not ask for a frame may wait forever. */
  function nextRenderedFrame(): Promise<void> {
    return new Promise((resolve) => {
      const remove = gl.postRender.addEventListener(() => {
        remove();
        resolve();
      });
      gl.requestRender();
    });
  }

  function waitTicks(n: number, timeoutMs = 60_000): Promise<FrameCounts> {
    const target = ticks + n;
    const deadline = Date.now() + timeoutMs;
    return new Promise((resolve, reject) => {
      const poll = () => {
        if (ticks >= target) {
          resolve({ ticks, livingRequests, renderRequests, renders });
          return;
        }
        if (Date.now() > deadline) {
          reject(new Error(`only ${ticks} ticks, wanted ${target}`));
          return;
        }
        requestAnimationFrame(poll);
      };
      poll();
    });
  }

  /**
   * Renders one frame at the current scene time and returns it as a PNG data URL. Read inside a
   * postRender, the same way `CesiumSceneManager.snapshot()` does: the drawing buffer is not
   * preserved between frames. Asking for the frame from inside the listener is what keeps this
   * from hanging in request-render mode.
   */
  function grabPng(): Promise<string> {
    return new Promise((resolve) => {
      const remove = gl.postRender.addEventListener(() => {
        remove();
        const source = viewer.canvas;
        const target = document.createElement("canvas");
        target.width = source.width;
        target.height = source.height;
        target.getContext("2d")?.drawImage(source, 0, 0);
        resolve(target.toDataURL("image/png"));
      });
      gl.requestRender();
    });
  }

  function splatPrimitive(): SplatPrimitive | undefined {
    const tileset = scene.sites.tilesetFor(SITE_ID, "gaussian-splat");
    return tileset === null ? undefined : splatTilesetOf(tileset).gaussianSplatPrimitive;
  }

  /**
   * Reads the RGBA32UI attribute texture back through a framebuffer of our own, and compares
   * it word for word with the packed buffer the engine uploaded into it (the interception's
   * capture, zero-padded to the texture as the engine pads it). Restores the framebuffer
   * binding Cesium had, so its cached GL state stays true.
   */
  function readAttributeTexture(): TextureReadback {
    const fail = (error: string): TextureReadback => ({
      words: 0,
      mismatches: -1,
      positionMismatches: -1,
      error,
    });
    const primitive = splatPrimitive();
    const positions = primitive?._positions;
    const numSplats = primitive?._numSplats ?? 0;
    const texture = primitive?.gaussianSplatTexture as
      { _texture?: WebGLTexture; _width?: number; _height?: number } | undefined;
    if (positions === undefined || texture?._texture === undefined) return fail("no texture");
    const capture = findSplatCapture(numSplats, digestSplatPositions(positions, numSplats));
    if (capture === undefined) return fail("no capture");
    const context = (gl as unknown as { context: { _gl: WebGL2RenderingContext } }).context;
    const webgl = context._gl;
    const width = texture._width ?? 0;
    const height = texture._height ?? 0;
    const previous = webgl.getParameter(webgl.FRAMEBUFFER_BINDING) as WebGLFramebuffer | null;
    const framebuffer = webgl.createFramebuffer();
    const words = new Uint32Array(width * height * 4);
    try {
      webgl.bindFramebuffer(webgl.FRAMEBUFFER, framebuffer);
      webgl.framebufferTexture2D(
        webgl.FRAMEBUFFER,
        webgl.COLOR_ATTACHMENT0,
        webgl.TEXTURE_2D,
        texture._texture,
        0,
      );
      const status = webgl.checkFramebufferStatus(webgl.FRAMEBUFFER);
      if (status !== webgl.FRAMEBUFFER_COMPLETE) return fail(`framebuffer ${String(status)}`);
      webgl.readPixels(0, 0, width, height, webgl.RGBA_INTEGER, webgl.UNSIGNED_INT, words);
      const error = webgl.getError();
      if (error !== webgl.NO_ERROR) return fail(`readPixels ${String(error)}`);
    } finally {
      webgl.bindFramebuffer(webgl.FRAMEBUFFER, previous);
      webgl.deleteFramebuffer(framebuffer);
    }
    const layout = splatTextureLayout(
      numSplats,
      primitive?._splatRowMask ?? 0,
      primitive?._splatRowShift ?? 0,
    );
    const isPosition = new Uint8Array(words.length);
    for (let splat = 0; splat < numSplats; splat += 1) {
      const offset = positionWordOffset(splat, layout);
      isPosition[offset] = 1;
      isPosition[offset + 1] = 1;
      isPosition[offset + 2] = 1;
    }
    let mismatches = 0;
    let positionMismatches = 0;
    for (let i = 0; i < words.length; i += 1) {
      if (words[i] === (capture.data[i] ?? 0)) continue;
      mismatches += 1;
      if (isPosition[i] === 1) positionMismatches += 1;
    }
    return { words: words.length, mismatches, positionMismatches, error: null };
  }

  function cadence(n: number, timeoutMs = 120_000): Promise<CadenceProbe> {
    requestedAfterTick = 0;
    renderedAfterRequest = 0;
    tickIntervals.length = 0;
    probing = true;
    return waitTicks(n, timeoutMs)
      .then(() => ({
        ticks: n,
        requestedAfterTick,
        renderedAfterRequest,
        meanTickMs:
          tickIntervals.length === 0
            ? 0
            : tickIntervals.reduce((sum, ms) => sum + ms, 0) / tickIntervals.length,
      }))
      .finally(() => {
        probing = false;
      });
  }

  return {
    setGpuMotion(enabled: boolean): void {
      scene.living.setGpuMotion(enabled);
    },
    hookInstalled(): boolean {
      return splatPrimitive()?.vertexMotion !== undefined;
    },
    readAttributeTexture,
    cadence,
    setTime(t: number): void {
      viewer.clock.currentTime = JulianDate.addSeconds(
        JulianDate.fromIso8601(LIVING_EPOCH_ISO),
        t,
        new JulianDate(),
      );
    },
    setWind(wind: WindSettings): void {
      scene.living.setWind(wind);
    },
    status() {
      return { ...scene.living.status, captures: splatCaptureCount() };
    },
    counts(): FrameCounts {
      return { ticks, livingRequests, renderRequests, renders };
    },
    waitTicks,
    setMotionModel(model: LivingMotionModel): void {
      scene.living.setMotionModel(model);
    },
    async grabFrameHash(): Promise<string> {
      return digest(await grabPng());
    },
    grabFrame: grabPng,
    async snapshotProbe(): Promise<{ hash: string; displacedDuringCapture: boolean }> {
      // Registered before `snapshot()` registers its own, so it runs first — while the hold is
      // still in force and before `release()`. `displaced` reports what the deformer wrote in
      // this frame's preUpdate, which is the frame the canvas is about to be read from.
      let displacedDuringCapture = true;
      const remove = gl.postRender.addEventListener(() => {
        remove();
        displacedDuringCapture = scene.living.status.sites.some((site) => site.displaced);
      });
      try {
        const result = await scene.snapshot(640);
        return { hash: digest(result?.image ?? ""), displacedDuringCapture };
      } finally {
        remove();
      }
    },
    probe() {
      return { preUpdate: gl.preUpdate.numberOfListeners };
    },
    destroyLiving(): void {
      scene.living.destroy();
    },
    async waitUntilReady(timeoutMs: number): Promise<LivingSurveyStatus> {
      const deadline = Date.now() + timeoutMs;
      while (Date.now() < deadline) {
        const status = scene.living.status;
        if (status.sites.some((s) => s.phase === "ready")) {
          armLivingRequestCounter();
          return status;
        }
        await nextRenderedFrame();
      }
      armLivingRequestCounter();
      return scene.living.status;
    },
  };
}
