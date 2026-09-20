import {
  Cartesian2,
  type Cartesian3,
  Cartographic,
  Color,
  Math as CesiumMath,
  RequestScheduler,
  type Scene,
  Viewer,
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
import { SelectionManager } from "./SelectionManager";
import { SiteManager } from "./SiteManager";
import { installSplatTextureInterception } from "./splatCapture";
import type { Geocoder, SceneEvents } from "./types";
import type { TokenState } from "@/state/viewer";

const log = createLogger("scene");

export interface SceneManagerOptions {
  ionToken: string | undefined;
  /** Initial camera in degrees / metres. */
  home?: { longitude: number; latitude: number; height: number };
}

/**
 * The single owner of the CesiumJS viewer. React talks to this object through
 * a context; it never touches Cesium primitives directly. Construct once,
 * destroy once.
 */
export class CesiumSceneManager {
  readonly viewer: Viewer;
  readonly scene: Scene;
  readonly events = new Emitter<SceneEvents>();
  readonly camera: CameraController;
  readonly clipping: ClippingManager;
  readonly layers: LayerManager;
  readonly performance: PerformanceManager;
  readonly sites: SiteManager;
  readonly living: LivingSurveyManager;
  readonly selection: SelectionManager;
  readonly measurement: MeasurementManager;
  readonly mission: MissionManager;
  readonly areas: AreaEditor;
  readonly explore: ExploreController;
  readonly keyboard: KeyboardNavigator;
  readonly debug: DebugManager;
  readonly tokenState: TokenState;
  private geocoderInstance: Geocoder;
  private destroyed = false;
  private interactionMode: "select" | "measure" | "explore" = "select";
  private pickingGround = false;
  private renderRecoveries = 0;
  private readonly unsubscribe: (() => void)[] = [];

  constructor(container: HTMLElement, options: SceneManagerOptions) {
    const tokenState = configureIonToken(options.ionToken);
    this.tokenState = tokenState;
    this.viewer = new Viewer(container, {
      animation: false,
      timeline: false,
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      fullscreenButton: false,
      infoBox: false,
      selectionIndicator: false,
      vrButton: false,
      scene3DOnly: true,
      shouldAnimate: true,
      baseLayer: false,
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
    });
    this.scene = this.viewer.scene;
    const scene = this.scene;
    // Before any tileset can load. The interception only sees `generateFromAttributes` calls
    // made after it is in place, and a splat whose texture was packed first leaves the deformer
    // reporting `no-capture` until that tileset next rebuilds. Idempotent; returns its own
    // uninstaller, which `destroy()` runs with the rest.
    this.unsubscribe.push(installSplatTextureInterception());
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
    this.living = new LivingSurveyManager(this.viewer, this.events, this.sites, this.performance);
    this.selection = new SelectionManager(
      this.viewer,
      this.events,
      this.camera,
      this.layers,
      this.sites,
    );
    this.measurement = new MeasurementManager(this.viewer, this.events);
    this.mission = new MissionManager(this.viewer, this.events, this.camera);
    this.areas = new AreaEditor(this.viewer, this.events);
    // While the map waits for "the ground you mean", selection keeps its hands off the click.
    this.unsubscribe.push(
      this.events.on("ground-pick-mode", (on) => {
        this.pickingGround = on;
        this.selection.setEnabled(this.interactionMode === "select" && !on);
        if (on) this.viewer.canvas.style.cursor = "crosshair";
      }),
    );
    this.explore = new ExploreController(this.viewer, this.events);
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

  /** Measuring and exploring take over the pointer; selection yields. */
  setInteractionMode(mode: "select" | "measure" | "explore"): void {
    this.interactionMode = mode;
    this.selection.setEnabled(mode === "select" && !this.pickingGround);
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
    this.measurement.destroy();
    this.mission.destroy();
    this.areas.destroy();
    this.selection.destroy();
    this.living.destroy();
    this.sites.destroy();
    this.performance.destroy();
    this.layers.destroy();
    this.clipping.destroy();
    this.camera.destroy();
    this.events.clear();
    if (!this.viewer.isDestroyed()) this.viewer.destroy();
    log.info("viewer destroyed");
  }
}
