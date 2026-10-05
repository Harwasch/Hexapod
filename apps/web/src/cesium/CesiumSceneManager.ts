import {
  Cartesian2,
  type Cartesian3,
  Cartographic,
  CesiumWidget,
  Color,
  Math as CesiumMath,
  Moon,
  RequestScheduler,
  type Scene,
  SkyBox,
  Sun,
} from "cesium";

import { Emitter } from "@/lib/emitter";
import { createLogger, describeError } from "@/lib/log";

import { AreaEditor } from "./AreaEditor";
import { CameraController } from "./CameraController";
import { ClippingManager } from "./ClippingManager";
import { DebugManager } from "./DebugManager";
import { ExploreController } from "./ExploreController";
import { KeyboardNavigator } from "./KeyboardNavigator";
import { configureIonToken } from "./ion";
import { LayerManager } from "./LayerManager";
import { LivingSurveyManager } from "./LivingSurveyManager";
import { MeasurementManager } from "./MeasurementManager";
import { MissionManager } from "./MissionManager";
import { PerformanceManager } from "./PerformanceManager";
import { FallbackGeocoder, IonGeocoder, NominatimGeocoder } from "./providers/geocoder";
import { ScaleMeasure } from "./ScaleMeasure";
import { SelectionManager } from "./SelectionManager";
import { SiteManager } from "./SiteManager";
import { installSplatTextureInterception } from "./splatCapture";
import { SplatCollider } from "./SplatCollider";
import { SplatMotionGate } from "./splatMotionGate";
import { installCameraPickHook } from "./cameraPickHook";
import { installSplatDecoder } from "./splatDecoder";
import { installSplatSorter } from "./splatSorter";
import { UiActivity } from "./uiActivity";
import {
  prefetchScanDestination,
  ScanRendererHost,
  type ScanRendererStatus,
} from "./scanView/ScanRendererHost";
import { SceneSelectController } from "./sceneSelect/SceneSelectController";
import type { SplatRendererKind } from "./scanView/types";
import type { Geocoder, SceneEvents } from "./types";
import type { TokenState } from "@/state/viewer";

const log = createLogger("scene");

export interface SceneManagerOptions {
  ionToken: string | undefined;
  /** Initial camera in degrees / metres. */
  home?: { longitude: number; latitude: number; height: number };
  /**
   * Whether this build allows Living Survey motion on the GPU (`VITE_SPLAT_GPU_MOTION`, default
   * true). Where allowed, `living.setGpuMotion` — the viewer's setting — decides; the CPU path
   * is the fallback either way when the engine or a snapshot cannot take the shader path.
   */
  splatGpuMotion?: boolean;
  /**
   * The stars and the moon, added once the globe has its first tiles (deferNightSky). Default
   * true; false for a harness that compares frames and must not have the sky arrive mid-run.
   */
  nightSky?: boolean;
}

/** Below this altitude over a splat site the view is the scan, shown ungraded. */
const SCAN_GRADE_ALTITUDE_M = 400;
/** Inside a scan and this close to the ground, the scan is the ground: no terrain under it. */
const SCAN_FLOORLESS_ALTITUDE_M = 30;
/** The stars wait for the globe's first tiles at most this long (deferNightSky). */
const NIGHT_SKY_DEADLINE_MS = 5_000;

/**
 * The single owner of the CesiumJS viewer. React talks to this object through
 * a context; it never touches Cesium primitives directly. Construct once,
 * destroy once.
 */
export class CesiumSceneManager {
  /**
   * The engine's own widget: canvas, scene, clock, entities and data sources. Named `viewer`
   * because it used to be one (see the constructor) and because e2e reads
   * `__twin.viewer.entities` and `.clock`, which the widget has too.
   */
  readonly viewer: CesiumWidget;
  /** `.cesium-viewer`: the widget and its credit bar, removed as one on destroy. */
  private readonly host: HTMLElement;
  readonly scene: Scene;
  readonly events = new Emitter<SceneEvents>();
  readonly camera: CameraController;
  readonly clipping: ClippingManager;
  readonly layers: LayerManager;
  readonly performance: PerformanceManager;
  readonly splatGate: SplatMotionGate;
  readonly collider: SplatCollider;
  private readonly uninstallSplatSorter: () => void;
  private readonly uninstallSplatDecoder: () => void;
  private readonly uninstallPickHook: () => void;
  private readonly uiActivity: UiActivity;
  private readonly scanRenderer: ScanRendererHost;
  readonly sites: SiteManager;
  readonly living: LivingSurveyManager;
  readonly selection: SelectionManager;
  /** Selecting a scan's objects in the scene: click, cycle, paint (cesium/sceneSelect). */
  readonly sceneSelect: SceneSelectController;
  readonly measurement: MeasurementManager;
  /** Two points on a scan, for the Set real size tool (features/inspector/RealSize.tsx). */
  readonly scaleMeasure: ScaleMeasure;
  readonly mission: MissionManager;
  readonly areas: AreaEditor;
  readonly explore: ExploreController;
  readonly keyboard: KeyboardNavigator;
  readonly debug: DebugManager;
  readonly tokenState: TokenState;
  private geocoderInstance: Geocoder;
  private destroyed = false;
  private interactionMode: "select" | "measure" | "explore" = "select";
  private exploring = false;
  private insideScan = false;
  private scanAltitude = Number.POSITIVE_INFINITY;
  private cameraMoving = false;
  private pickingGround = false;
  private renderRecoveries = 0;
  private readonly unsubscribe: (() => void)[] = [];

  constructor(container: HTMLElement, options: SceneManagerOptions) {
    const tokenState = configureIonToken(options.ionToken);
    this.tokenState = tokenState;
    // A `CesiumWidget`, not a `Viewer`. Every one of Viewer's widgets was switched off here, and
    // since 1.145 the widget itself owns what the app used of Viewer -- entities, data sources,
    // the clock and its ticking, resizing. Constructing a Viewer is what put `@cesium/widgets`
    // in the bundle (the widgets, knockout, their view models: 226 kB of the engine chunk, and
    // widgets.css): code that ran only to build DOM nobody saw. The DOM of Viewer's that the
    // app does rely on is rebuilt here: a `.cesium-viewer` host and, after the widget, the
    // `.cesium-viewer-bottom` credit bar that CreditSlot moves into the HUD. The host is
    // removed on destroy, so CreditSlot's cleanup finds the bar's home gone and drops it too.
    const host = document.createElement("div");
    host.className = "cesium-viewer";
    container.appendChild(host);
    this.host = host;
    const credits = document.createElement("div");
    credits.className = "cesium-viewer-bottom";
    this.viewer = new CesiumWidget(host, {
      scene3DOnly: true,
      shouldAnimate: true,
      baseLayer: false,
      // No sky box (and so, from the widget, no sun or moon) yet: deferNightSky adds them once
      // the globe has its first tiles.
      skyBox: false,
      // Render only when something changed (camera, tiles, entities, explicit requests).
      // Every manager calls scene.requestRender() after it mutates the scene.
      requestRenderMode: true,
      // Render errors are logged, toasted and recovered from here; Cesium's own panel would
      // stop the app dead on an engine hiccup.
      showRenderLoopErrors: false,
      maximumRenderTimeChange: Number.POSITIVE_INFINITY,
      // Motion never uses MSAA; the PerformanceManager raises it for the still frame.
      msaaSamples: 1,
      contextOptions: { webgl: { powerPreference: "high-performance", antialias: true } },
      // The "Data attribution" dialog is modal and belongs over the whole app, but Cesium
      // would hang it inside the widget — which lives in `.viewport`, a stacking context
      // pinned at `--z-map`, so the HUD would paint over the dialog and swallow its clicks.
      // Hosting it on <body> takes it out of that context; `app.css` then puts it on the
      // sheet layer and dresses it in the glass material.
      creditViewport: document.body,
      creditContainer: credits,
    });
    host.appendChild(credits);
    this.scene = this.viewer.scene;
    const scene = this.scene;
    // Before any tileset can load. The interception only sees `generateFromAttributes` calls
    // made after it is in place, and a splat whose texture was packed first leaves the deformer
    // reporting `no-capture` until that tileset next rebuilds. Idempotent; returns its own
    // uninstaller, which `destroy()` runs with the rest.
    this.unsubscribe.push(installSplatTextureInterception());
    // The sun is drawn procedurally (no download), so it is there from the first frame.
    scene.sun = new Sun();
    if (options.nightSky !== false) this.unsubscribe.push(this.deferNightSky());
    scene.globe.depthTestAgainstTerrain = true;
    scene.globe.enableLighting = false;
    scene.globe.showGroundAtmosphere = true;
    if (scene.skyAtmosphere) scene.skyAtmosphere.show = true;
    scene.fog.enabled = true;
    scene.fog.density = 0.0006;
    scene.globe.baseColor = Color.fromCssColorString("#0b1a3a");
    scene.highDynamicRange = false;
    // MSAA is on by default; FXAA only when the PerformanceManager turns MSAA off.
    scene.postProcessStages.fxaa.enabled = false;

    // Terrain, imagery and ion-hosted site meshes share one host; with a deep mesh streaming,
    // its requests would otherwise crowd out the terrain under it for as long as it loads.
    const perServer = RequestScheduler.requestsByServer as Record<string, number>;
    perServer["assets.ion.cesium.com:443"] = 36;
    perServer["tile.googleapis.com:443"] = 24;

    this.camera = new CameraController(this.viewer, this.events);
    this.clipping = new ClippingManager(scene);
    this.layers = new LayerManager(this.viewer, this.events, this.clipping);
    this.performance = new PerformanceManager(this.viewer, this.events);
    // The interface first, then the camera, then streaming (uiActivity.ts).
    this.uiActivity = new UiActivity(this.viewer.canvas);
    const interfaceBusy = (): boolean => this.uiActivity.active;
    this.splatGate = new SplatMotionGate(this.viewer.scene, this.events, interfaceBusy);
    this.uninstallSplatSorter = installSplatSorter();
    // Installs the hook only: the decode workers start with the first splat tile, not here
    // before the first frame (splatDecoder.ts).
    this.uninstallSplatDecoder = installSplatDecoder(interfaceBusy);
    this.collider = new SplatCollider(this.viewer.scene, () => this.splatGate.holding);
    this.scanRenderer = new ScanRendererHost(this.viewer);
    // A dedicated splat renderer that throws while drawing is retired on the spot, and the
    // globe carries on (overlayFrames.ts); the scan it drew is gone from the view, so say why.
    this.scanRenderer.onFailure = (message) =>
      this.events.emit("toast", {
        tone: "error",
        title: "The scan stopped drawing",
        body: `${message.replace(/\.$/, "")}. The map carries on; Settings › Advanced › Splat renderer can draw the scan with another renderer.`,
        id: "scan-renderer-failed",
      });
    this.camera.setCollider(this.collider);
    // Cesium's own camera control asks the splats' solids before reading depth back from
    // the GPU (engine patch, ScreenSpaceCameraController.pickHook).
    this.uninstallPickHook = installCameraPickHook(this.collider);
    this.performance.addScreenSpaceErrorSink("world", (sse, pixelRatio) =>
      this.layers.applyWorldScreenSpaceError(sse, pixelRatio),
    );
    this.layers.bindPerformance(this.performance);
    this.sites = new SiteManager(
      this.viewer,
      this.events,
      this.camera,
      this.clipping,
      this.performance,
    );
    this.living = new LivingSurveyManager(this.viewer, this.events, this.sites, this.performance, {
      gpuMotion: options.splatGpuMotion !== false,
    });
    this.selection = new SelectionManager(
      this.viewer,
      this.events,
      this.camera,
      this.layers,
      this.sites,
    );
    this.selection.setCollider(this.collider);
    // Fly to goes through the camera controller (its pace, range and the dedicated renderer's
    // destination prefetch), and B and Escape are the app's keys (AppShell's GlobalHotkeys).
    this.camera.setDestinationPrefetch(prefetchScanDestination);
    this.sceneSelect = new SceneSelectController(this.viewer, {
      enabled: () => this.selectionWanted(),
      ownClicks: false,
      ownKeys: false,
      fly: (sphere) => this.camera.flyToObject(sphere),
    });
    // One click, one answer: an object of a scan under the cursor first, else the cards.
    this.selection.setClickClaim((position) => this.sceneSelect.click(position.x, position.y));
    this.measurement = new MeasurementManager(this.viewer, this.events);
    this.mission = new MissionManager(this.viewer, this.events, this.camera);
    this.areas = new AreaEditor(this.viewer, this.events);
    this.unsubscribe.push(
      // A photographic scan close up is shown in its own colours (PerformanceManager
      // setGradeSuppressed): within a few hundred metres of a splat site that is on screen.
      this.events.on("camera", (pose) => {
        const splat = this.sites.activeRepresentation === "gaussian-splat";
        this.performance.setGradeSuppressed(splat && pose.altitude < SCAN_GRADE_ALTITUDE_M);
        this.selection.setHoverEnabled(!(splat && pose.altitude < SCAN_GRADE_ALTITUDE_M));
        this.insideScan = splat && this.sites.insideSplatScan();
        this.updateScanRenderer();
        this.scanAltitude = pose.altitude;
        this.updateScanView();
      }),
    );
    // While the map waits for "the ground you mean", selection keeps its hands off the click;
    // while exploring (walk or fly, from the toolbar or F) it does no hover picks either: each
    // is a render pass and a GPU read-back, and the pointer is the look.
    this.unsubscribe.push(
      this.events.on("explore", (on) => {
        this.exploring = on;
        this.selection.setEnabled(this.selectionWanted());
        this.updateScanView();
      }),
      this.events.on("motion", (moving) => {
        this.cameraMoving = moving;
        this.updateScanView();
      }),
      // A site engaging or changing representation changes what a splat renderer draws.
      this.events.on("tilesets", () => this.updateScanRenderer()),
      this.events.on("ground-pick-mode", (on) => {
        this.pickingGround = on;
        this.selection.setEnabled(this.selectionWanted());
        if (on) this.viewer.canvas.style.cursor = "crosshair";
      }),
    );
    this.explore = new ExploreController(this.viewer, this.events);
    this.explore.setCollider(this.collider);
    this.explore.setCameraController(this.camera);
    this.scaleMeasure = new ScaleMeasure(this.viewer, this.events, this.collider, (siteId) =>
      this.sites.scalableTileset(siteId),
    );
    this.keyboard = new KeyboardNavigator(this.viewer, this.camera);
    this.debug = new DebugManager(this.viewer, this.sites, (enabled) =>
      this.clipping.setEnabled(enabled),
    );
    this.geocoderInstance = new FallbackGeocoder(new IonGeocoder(scene), new NominatimGeocoder());

    const home = options.home ?? { longitude: -110, latitude: 35, height: 18_000_000 };
    this.camera.setView(home.longitude, home.latitude, home.height);

    this.unsubscribe.push(
      scene.renderError.addEventListener((_scene: Scene, error: unknown) => {
        log.error("render error", { error: describeError(error) });
        this.recoverFromRenderError(describeError(error));
      }),
    );
    const canvas = this.viewer.canvas;
    const onLost = (event: Event) => {
      event.preventDefault();
      log.warn("WebGL context lost");
      this.events.emit("status", {
        status: "context-lost",
        message: "The graphics context was lost. Reload to continue.",
      });
    };
    const onRestored = () => {
      log.info("WebGL context restored");
      this.events.emit("status", { status: "ready" });
      scene.requestRender();
    };
    canvas.addEventListener("webglcontextlost", onLost);
    canvas.addEventListener("webglcontextrestored", onRestored);
    this.unsubscribe.push(() => {
      canvas.removeEventListener("webglcontextlost", onLost);
      canvas.removeEventListener("webglcontextrestored", onRestored);
    });

    this.events.emit("token", tokenState);
    this.events.emit("status", { status: "ready" });
    log.info("viewer ready", { tokenState });
  }

  get geocoder(): Geocoder {
    return this.geocoderInstance;
  }

  /**
   * The stars and the moon, once the globe has its first tiles rather than before it.
   *
   * The engine's default sky box is six 1024² star maps, 868 kB of JPEG from
   * `CESIUM_BASE_URL` (plus 18 kB of moon), requested by the widget's constructor -- the
   * biggest static download of the boot, queued beside the first terrain and imagery tiles
   * that the opening view is actually waiting for. Deferred, it loads in the first idle moment
   * after the globe's initial tiles are in (or after `NIGHT_SKY_DEADLINE_MS` regardless), and
   * the stars appear a moment later on an otherwise identical view.
   *
   * Deferring rather than replacing it with a lighter sky, because there is no flash to hide:
   * until a sky box's textures arrive it draws nothing, so space shows the scene's black
   * background either way -- exactly what the default showed while those 868 kB downloaded.
   * Only the ordering changes. Near the ground the atmosphere hides the stars in any case.
   */
  private deferNightSky(): () => void {
    const scene = this.scene;
    let sawLoading = false;
    let cancelIdle: (() => void) | undefined;
    let removeFrame: (() => void) | undefined;
    const install = (): void => {
      cancelIdle = undefined;
      if (this.destroyed || scene.isDestroyed()) return;
      scene.skyBox = SkyBox.createEarthSkyBox();
      scene.moon = new Moon();
      scene.requestRender();
    };
    const schedule = (): void => {
      if (!removeFrame) return;
      removeFrame();
      removeFrame = undefined;
      clearTimeout(deadline);
      if (typeof requestIdleCallback === "function") {
        const handle = requestIdleCallback(install, { timeout: 1_000 });
        cancelIdle = () => cancelIdleCallback(handle);
      } else {
        const handle = window.setTimeout(install, 0);
        cancelIdle = () => clearTimeout(handle);
      }
    };
    removeFrame = scene.postRender.addEventListener(() => {
      // The first frame reports an empty queue before the globe has asked for anything.
      if (!scene.globe.tilesLoaded) sawLoading = true;
      else if (sawLoading) schedule();
    });
    const deadline = window.setTimeout(schedule, NIGHT_SKY_DEADLINE_MS);
    return () => {
      removeFrame?.();
      removeFrame = undefined;
      clearTimeout(deadline);
      cancelIdle?.();
    };
  }

  /**
   * Cesium stops its render loop after an exception. A single bad tile or
   * decode failure must not take the globe down, so we restart the loop a
   * few times before declaring the viewer broken.
   */
  private recoverFromRenderError(message: string): void {
    if (this.destroyed) return;
    this.renderRecoveries += 1;
    if (this.renderRecoveries > 5) {
      this.events.emit("status", { status: "error", message });
      return;
    }
    this.events.emit("toast", {
      tone: "warning",
      title: "Recovered from a rendering error",
      body: message,
      id: "render-error",
    });
    setTimeout(() => {
      if (this.destroyed || this.viewer.isDestroyed()) return;
      this.viewer.useDefaultRenderLoop = false;
      this.viewer.useDefaultRenderLoop = true;
      this.scene.requestRender();
    }, 250);
  }

  /** Google Photorealistic tiles must be paired with the Google geocoder (Google ToS). */
  useGoogleGeocoder(enabled: boolean): void {
    this.geocoderInstance = new FallbackGeocoder(
      new IonGeocoder(this.scene, enabled ? "google" : "default"),
      new NominatimGeocoder(),
    );
  }

  /**
   * A picture of the view, at most `maxWidth` wide, as a JPEG data URL. Read inside the
   * next frame's postRender so the drawing buffer is still intact.
   *
   * A Living Survey site is pinned at its measured pose for the captured frame, so the picture
   * shows what was surveyed rather than what was simulated — a photograph of a swaying tree
   * carries no label saying the sway was modelled. The deformer needs no special path for it:
   * applying the rig at zero wind restores the exact measured bytes in one frame.
   */
  snapshot(maxWidth = 1024): Promise<{ image: string; width: number; height: number } | null> {
    return new Promise((resolve) => {
      const scene = this.scene;
      const release = this.living.holdMeasuredPose();
      const remove = scene.postRender.addEventListener(() => {
        remove();
        release();
        try {
          const source = this.viewer.canvas;
          const scale = Math.min(1, maxWidth / source.width);
          const width = Math.max(1, Math.round(source.width * scale));
          const height = Math.max(1, Math.round(source.height * scale));
          const target = document.createElement("canvas");
          target.width = width;
          target.height = height;
          const context = target.getContext("2d");
          if (!context) {
            resolve(null);
            return;
          }
          context.drawImage(source, 0, 0, width, height);
          resolve({ image: target.toDataURL("image/jpeg", 0.85), width, height });
        } catch {
          resolve(null);
        }
      });
      scene.requestRender();
    });
  }

  /** The ground under a canvas pixel, in degrees, or null for sky. */
  groundAt(x: number, y: number): { longitude: number; latitude: number } | null {
    const window = new Cartesian2(x, y);
    let position: Cartesian3 | undefined;
    if (this.scene.pickPositionSupported) position = this.scene.pickPosition(window);
    if (!position) {
      const ray = this.viewer.camera.getPickRay(window);
      position = ray ? this.scene.globe.pick(ray, this.scene) : undefined;
    }
    position ??= this.viewer.camera.pickEllipsoid(window, this.scene.globe.ellipsoid);
    if (!position) return null;
    const carto = Cartographic.fromCartesian(position);
    return {
      longitude: CesiumMath.toDegrees(carto.longitude),
      latitude: CesiumMath.toDegrees(carto.latitude),
    };
  }

  /**
   * Inside a splat scan: the world around it holds still while the camera moves
   * (LayerManager.holdWorld) and refines once it stops; and near the ground -- walking,
   * flying low -- the scan is its own ground, with no terrain floor under it
   * (ClippingManager.setFloorless).
   */
  /**
   * Who draws splat scans: CesiumJS, or a dedicated renderer over the globe (Spark, PlayCanvas;
   * scanView/ScanRendererHost.ts), for comparison. Everything else stays CesiumJS's.
   */
  setSplatRenderer(kind: SplatRendererKind): void {
    this.sites.setSplatRenderer(kind);
    this.scanRenderer.setRenderer(kind);
    this.updateScanRenderer();
  }

  /** The dedicated splat renderer's state, for the debug panel and tests. */
  get scanRendererStatus(): ScanRendererStatus {
    return this.scanRenderer.status();
  }

  private updateScanRenderer(): void {
    const target = this.sites.scanTarget();
    this.scanRenderer.setTarget(target);
    this.collider.setSolidWhileHidden(target?.tileset ?? null);
  }

  private updateScanView(): void {
    const inside = this.insideScan;
    this.layers.holdWorld(inside, this.cameraMoving);
    const low = this.exploring || this.scanAltitude < SCAN_FLOORLESS_ALTITUDE_M;
    this.clipping.setFloorless(inside && low ? (this.sites.activeSite?.id ?? null) : null);
  }

  private selectionWanted(): boolean {
    return this.interactionMode === "select" && !this.pickingGround && !this.exploring;
  }

  /** Measuring and exploring take over the pointer; selection yields. */
  setInteractionMode(mode: "select" | "measure" | "explore"): void {
    this.interactionMode = mode;
    this.selection.setEnabled(this.selectionWanted());
    if (mode !== "measure") this.measurement.stop();
    if (mode !== "explore") this.explore.exit();
  }

  get isDestroyed(): boolean {
    return this.destroyed;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const off of this.unsubscribe) off();
    this.debug.destroy();
    this.keyboard.destroy();
    this.explore.destroy();
    this.scaleMeasure.destroy();
    this.measurement.destroy();
    this.mission.destroy();
    this.areas.destroy();
    this.selection.destroy();
    this.sceneSelect.destroy();
    this.living.destroy();
    this.sites.destroy();
    this.collider.destroy();
    this.uninstallSplatSorter();
    this.uninstallSplatDecoder();
    this.uninstallPickHook();
    this.scanRenderer.destroy();
    this.uiActivity.destroy();
    this.splatGate.destroy();
    this.performance.destroy();
    this.layers.destroy();
    this.clipping.destroy();
    this.camera.destroy();
    this.events.clear();
    if (!this.viewer.isDestroyed()) this.viewer.destroy();
    this.host.remove();
    log.info("viewer destroyed");
  }
}
