/**
 * The Time Lantern: a lens under the cursor through which dated public imagery shows instead of
 * today's photorealistic world.
 *
 * Three engine pieces make the hole and fill it, and nothing leaves the scene:
 *
 * - **The cut.** The world tileset's one custom shader (worldShader.ts) discards the Google mesh
 *   inside a screen circle at the cursor, with a stippled feather. The tileset is never
 *   re-clipped: a clipping polygon on it rebakes every tile model on each change.
 * - **The ground.** In the photorealistic world the globe is drawn only inside site footprints
 *   (ClippingManager); a globe-only window under the lens (`setLanternWindow`), swapped as the
 *   lens moves, keeps it drawn there too, so terrain shows where the mesh was cut away. The reach
 *   (`setLanternReach`) keeps the globe loading the whole area the lens may go while hidden.
 * - **The imagery.** Dated sources are imagery layers on the globe, limited to the reach: the
 *   current one at full alpha, its neighbours in time loading at alpha 0 so a step is instant,
 *   and under them a dark grid that reads as "no record" where a source has none.
 *
 * Input stays in the engine: Alt held over the map peeks; scrolling while the lens shows steps
 * through dates (a capture listener on window, attached only while it shows, that runs before
 * the camera's own wheel handling, so zoom comes back the moment it closes); Esc closes it.
 * React drives the rest through LanternBridge and hears back through `events.lantern`.
 *
 * The lens is offered only where it can be honest about what it shows: the photorealistic world
 * below 20 km, WebGL 2 clipping on, not while exploring or painting with the object brush.
 */
import {
  Cartesian2,
  Cartesian3,
  Cartographic,
  Color,
  Credit,
  Ellipsoid,
  GeographicTilingScheme,
  GridImageryProvider,
  ImageryLayer,
  Math as CesiumMath,
  Matrix4,
  Rectangle,
  Resource,
  SceneTransforms,
  UrlTemplateImageryProvider,
  WebMercatorTilingScheme,
  type CesiumWidget,
  type Scene,
} from "cesium";

import { isTyping } from "@/lib/hotkeys";
import type { Emitter } from "@/lib/emitter";
import { createLogger } from "@/lib/log";
import {
  LANTERN_MAX_ALTITUDE_M,
  type LanternAvailability,
  type LanternMode,
  type LanternUnavailable,
} from "@/state/lantern";
import { useSceneSelect } from "@/state/sceneSelect";

import type { CameraController } from "./CameraController";
import type { ClippingManager } from "./ClippingManager";
import type { LayerManager } from "./LayerManager";
import type { LanternArea, LanternFrame, LanternSource } from "./lanternTypes";
import {
  coverageAt,
  groundWindow,
  lensRadii,
  toDrawingBuffer,
  WHEEL_IDLE,
  wheelSteps,
  type WheelState,
} from "./timeLanternMath";
import type { SceneEvents } from "./types";
import type { WorldShader } from "./worldShader";

const log = createLogger("lantern");

/** The reach is the area asked about, grown by this share, and at least this wide (m). */
const REACH_GROWTH = 0.5;
const REACH_MIN_M = 3_000;
/**
 * The ground window around the lens, in lens radii. Its edge is found on the terrain the globe
 * draws (groundWindow's height refinement), so it lands on the screen within a few pixels of
 * where it is aimed, and a small margin is enough. Wherever the window is wider than the cut the
 * terrain is drawn beside the mesh and shows wherever it is nearer the camera than the mesh: at
 * 1.35 radii that read as a halo around the lens over the Oak Bench Ranch hills. The rim's glow
 * covers this margin.
 */
const WINDOW_MARGIN = 1.06;
const WINDOW_RAYS = 32;
/** A pointer at rest this long has its ground read from the depth buffer (once). */
const REST_MS = 200;
/** Layers of a closed lantern are kept this long, for a quick return, then destroyed. */
const IDLE_DISPOSE_MS = 60_000;
/** Planetary Computer mosaics answer nothing (204) below about this level. */
const WEB_MERCATOR_MIN_LEVEL = 12;
const METRES_PER_DEGREE = 111_320;
const NO_RECORD = Color.fromCssColorString("#0d1215");
const NO_RECORD_LINES = Color.fromCssColorString("rgba(160, 180, 190, 0.18)");

const scratchCartographic = new Cartographic();
const scratchSample = new Cartographic();
const scratchWindow = new Cartesian2();

function same(a: LanternAvailability, b: LanternAvailability): boolean {
  return a.ok === b.ok && (a.ok || (!b.ok && a.reason === b.reason));
}

/** The area asked about, grown into the reach the lens may roam (degrees). */
export function reachOf(area: LanternArea): LanternArea {
  const latitude = (area.south + area.north) / 2;
  const metresPerLon = METRES_PER_DEGREE * Math.max(Math.cos(CesiumMath.toRadians(latitude)), 0.01);
  const halfW = Math.max(
    ((area.east - area.west) * (1 + REACH_GROWTH)) / 2,
    REACH_MIN_M / 2 / metresPerLon,
  );
  const halfH = Math.max(
    ((area.north - area.south) * (1 + REACH_GROWTH)) / 2,
    REACH_MIN_M / 2 / METRES_PER_DEGREE,
  );
  const lon = (area.west + area.east) / 2;
  return {
    west: Math.max(-180, lon - halfW),
    south: Math.max(-90, latitude - halfH),
    east: Math.min(180, lon + halfW),
    north: Math.min(90, latitude + halfH),
  };
}

/** The coarsest level at which `rect` spans at most two tiles each way, as Cesium asks. */
function minimumLevelFor(rect: Rectangle, geographic: boolean): number {
  const width = Math.max(CesiumMath.toDegrees(rect.width), CesiumMath.toDegrees(rect.height), 1e-6);
  return Math.max(0, Math.floor(Math.log2((geographic ? 360 : 720) / width)));
}

export class TimeLantern {
  private readonly scene: Scene;
  private mode: LanternMode = "off";
  private sources: LanternSource[] = [];
  private index = 0;
  private reach: LanternArea | null = null;
  private reachKey = "";
  /** Ellipsoid height of the ground at the reach's centre (m), once sampled. */
  private planeHeight = 0;
  private availability: LanternAvailability = { ok: false, reason: "open-world" };
  private exploring = false;
  private altitude = Number.POSITIVE_INFINITY;
  /** The pointer over the canvas (CSS px from its top left), or null when it is elsewhere. */
  private pointer: { x: number; y: number } | null = null;
  private touch = false;
  /** The pointer is over the HUD, not the map: Alt there is not a peek at the map. */
  private pointerOutside = true;
  private dirty = true;
  private readonly lastView = new Matrix4();
  private lastBuffer = { width: 0, height: 0 };
  /** The ground at the resting pointer, read from the depth buffer (`REST_MS`). */
  private rested: { x: number; y: number; point: Cartesian3 } | null = null;
  private restDue = false;
  private restTimer: ReturnType<typeof setTimeout> | null = null;
  private wheel: WheelState = WHEEL_IDLE;
  private wheelOff: (() => void) | null = null;
  private readonly imagery = new Map<string, ImageryLayer>();
  private backdrop: ImageryLayer | null = null;
  private basemap: Promise<void> | null = null;
  private basemapReady = false;
  private disposeTimer: ReturnType<typeof setTimeout> | null = null;
  private lensShown = false;
  /** The window last given to the clipping, so an unchanged one is not swapped again. */
  private ring: Cartesian3[] | null = null;
  /** The lens is waiting for terrain under the cursor (`cursorGround`). */
  private waitingForGround = false;
  private lastFrame: LanternFrame | null = null;
  private readonly off: (() => void)[] = [];
  private destroyed = false;

  constructor(
    private readonly viewer: CesiumWidget,
    private readonly events: Emitter<SceneEvents>,
    private readonly clipping: ClippingManager,
    private readonly layers: LayerManager,
    private readonly worldShader: WorldShader,
    private readonly camera: CameraController,
  ) {
    this.scene = viewer.scene;
    const canvas = viewer.canvas;
    canvas.addEventListener("pointermove", this.onPointerMove, { passive: true });
    canvas.addEventListener("pointerdown", this.onPointerMove, { passive: true });
    canvas.addEventListener("pointerleave", this.onPointerLeave, { passive: true });
    window.addEventListener("keydown", this.onKeyDown, { capture: true });
    window.addEventListener("keyup", this.onKeyUp, { capture: true });
    window.addEventListener("blur", this.onBlur);
    this.off.push(
      () => {
        canvas.removeEventListener("pointermove", this.onPointerMove);
        canvas.removeEventListener("pointerdown", this.onPointerMove);
        canvas.removeEventListener("pointerleave", this.onPointerLeave);
        window.removeEventListener("keydown", this.onKeyDown, { capture: true });
        window.removeEventListener("keyup", this.onKeyUp, { capture: true });
        window.removeEventListener("blur", this.onBlur);
      },
      this.scene.preRender.addEventListener(this.frame),
      // Terrain arriving under the lens moves its ground: the window is placed again.
      this.scene.globe.tileLoadProgressEvent.addEventListener(this.onTilesLoading),
      events.on("world", () => this.updateAvailability()),
      events.on("explore", (on) => {
        this.exploring = on;
        this.updateAvailability();
      }),
      events.on("camera", (pose) => {
        this.altitude = pose.altitude;
        this.updateAvailability();
      }),
      useSceneSelect.subscribe((state, previous) => {
        if (state.mode !== previous.mode) this.updateAvailability();
      }),
    );
    this.updateAvailability();
  }

  /** Whether the lens can show here and now, and why not. */
  get available(): LanternAvailability {
    return this.availability;
  }

  get currentMode(): LanternMode {
    return this.mode;
  }

  get currentIndex(): number {
    return this.index;
  }

  /** Whether a wheel listener is attached (tests): only while the lens shows. */
  get listensToWheel(): boolean {
    return this.wheelOff !== null;
  }

  /**
   * The dated sources to step through, newest first, and the area they were listed for. The
   * lens may roam the area grown by half (at least 3 km across), where the globe preloads.
   */
  setSources(sources: LanternSource[], area: LanternArea | null): void {
    const previousId = this.sources[this.index]?.id;
    this.sources = sources;
    const kept = previousId ? sources.findIndex((s) => s.id === previousId) : -1;
    this.index = kept >= 0 ? kept : Math.min(this.index, Math.max(0, sources.length - 1));
    const key = area ? `${area.west},${area.south},${area.east},${area.north}` : "";
    if (key !== this.reachKey) {
      this.reachKey = key;
      this.reach = area ? reachOf(area) : null;
      this.planeHeight = 0;
      this.disposeImagery();
      if (this.reach) {
        const reach = this.reach;
        void this.camera
          .groundHeightAt((reach.west + reach.east) / 2, (reach.south + reach.north) / 2)
          .then((height) => {
            if (height === undefined || this.reach !== reach || this.destroyed) return;
            this.planeHeight = height;
            this.dirty = true;
            this.scene.requestRender();
          });
      }
      if (this.mode !== "off") this.clipping.setLanternReach(this.reach);
    }
    this.syncImagery();
    this.dirty = true;
    this.publish();
  }

  setMode(mode: LanternMode): void {
    const next = this.availability.ok ? mode : "off";
    if (next === this.mode) {
      if (next !== mode) this.publish();
      return;
    }
    this.mode = next;
    if (next === "off") {
      this.clearLens();
      this.clipping.setLanternReach(null);
      this.wheelOff?.();
      this.wheelOff = null;
      this.wheel = WHEEL_IDLE;
    } else {
      this.clipping.setLanternReach(this.reach);
      this.attachWheel();
      this.dirty = true;
    }
    this.syncImagery();
    this.scene.requestRender();
    this.publish();
  }

  /** Shows the source at `index` (the rail's order). */
  select(index: number): void {
    const next = Math.max(0, Math.min(index, this.sources.length - 1));
    if (next === this.index || this.sources.length === 0) return;
    this.index = next;
    this.syncImagery();
    this.dirty = true;
    this.scene.requestRender();
    this.publish();
  }

  /** Older (+) or newer (-) by `delta` sources. */
  step(delta: number): void {
    this.select(this.index + delta);
  }

  // ---- Input ---------------------------------------------------------------------------

  private readonly onPointerMove = (event: PointerEvent): void => {
    const canvas = this.viewer.canvas;
    this.touch = event.pointerType === "touch";
    if (this.touch) this.pointer = { x: canvas.clientWidth / 2, y: canvas.clientHeight / 2 };
    else {
      const rect = canvas.getBoundingClientRect();
      this.pointer = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    }
    this.moved();
  };

  private readonly onPointerLeave = (event: PointerEvent): void => {
    if (event.pointerType === "touch") return;
    // The lens stays where the pointer left the map; Alt over a panel does not peek.
    this.pointerOutside = true;
  };

  private moved(): void {
    this.pointerOutside = false;
    this.dirty = true;
    this.rested = null;
    this.restDue = false;
    if (this.restTimer !== null) clearTimeout(this.restTimer);
    this.restTimer = null;
    if (this.mode === "off") return;
    this.restTimer = setTimeout(() => {
      this.restTimer = null;
      this.restDue = true;
      this.scene.requestRender();
    }, REST_MS);
    this.scene.requestRender();
  }

  private readonly onTilesLoading = (): void => {
    if (this.mode === "off") return;
    this.dirty = true;
    if (this.waitingForGround) this.scene.requestRender();
  };

  private readonly onKeyDown = (event: KeyboardEvent): void => {
    if (event.key === "Alt") {
      if (event.repeat || event.ctrlKey || event.metaKey || event.shiftKey) return;
      if (isTyping(event.target) || this.pointerOutside) return;
      // Firefox would open its menu bar on Alt; the key is the lantern's over the map.
      event.preventDefault();
      if (this.mode === "off") this.setMode("peek");
      return;
    }
    if (
      (event.key === "[" || event.key === "]") &&
      this.mode !== "off" &&
      !event.ctrlKey &&
      !event.metaKey &&
      !event.altKey &&
      this.mapHasKeys(event.target) &&
      useSceneSelect.getState().index < 0
    ) {
      // Newer, older: the map's own keys while the lens shows. A scan object's parts take them
      // while one is selected (SceneSelectController), and the strip's rail handles its own.
      event.preventDefault();
      this.step(event.key === "]" ? 1 : -1);
      return;
    }
    if (event.key === "Escape" && this.mode !== "off" && !isTyping(event.target)) {
      // Before the app's own Escape (GlobalHotkeys, which skips a handled key): one step back.
      event.preventDefault();
      this.setMode("off");
    }
  };

  /** Whether a key went to the map rather than to a control: nothing focused, or the canvas. */
  private mapHasKeys(target: EventTarget | null): boolean {
    return (
      !(target instanceof Element) || target === document.body || target === this.viewer.canvas
    );
  }

  private readonly onKeyUp = (event: KeyboardEvent): void => {
    if (event.key !== "Alt") return;
    if (this.mode === "peek") {
      event.preventDefault();
      this.setMode("off");
    }
  };

  private readonly onBlur = (): void => {
    if (this.mode === "peek") this.setMode("off");
  };

  /**
   * While the lens shows, the wheel over the map travels in time instead of zooming. On window,
   * in the capture phase, so it runs before the camera's wheel handling (its container's capture
   * listeners) and Cesium's zoom (the canvas's); attached only while needed, since a cancelable
   * wheel listener makes the browser wait for script before every scroll (SceneSelectController
   * #syncWheel). The object brush's Alt+wheel, which comes first, is left alone.
   */
  private attachWheel(): void {
    if (this.wheelOff) return;
    const listener = (event: WheelEvent): void => {
      if (event.target !== this.viewer.canvas || event.defaultPrevented) return;
      const select = useSceneSelect.getState();
      if (event.altKey && select.index >= 0 && select.candidates.length > 1) return;
      event.preventDefault();
      event.stopPropagation();
      const { steps, state } = wheelSteps(
        this.wheel,
        event.deltaY,
        event.deltaMode,
        performance.now(),
      );
      this.wheel = state;
      if (steps !== 0) this.step(steps);
    };
    window.addEventListener("wheel", listener, { capture: true, passive: false });
    this.wheelOff = () => window.removeEventListener("wheel", listener, { capture: true });
  }

  // ---- Availability ----------------------------------------------------------------------

  private updateAvailability(): void {
    const reason = this.unavailableReason();
    const next: LanternAvailability = reason ? { ok: false, reason } : { ok: true };
    if (same(next, this.availability)) return;
    this.availability = next;
    if (!next.ok && this.mode !== "off") this.setMode("off");
    else this.publish();
  }

  private unavailableReason(): LanternUnavailable | null {
    if (!this.clipping.supported) return "webgl1";
    if (!this.clipping.photorealisticWorld) return "open-world";
    if (!this.clipping.isEnabled) return "clipping-off";
    if (this.clipping.orbitView) return "orbit";
    if (this.altitude > LANTERN_MAX_ALTITUDE_M) return "altitude";
    if (this.exploring) return "explore";
    if (useSceneSelect.getState().mode === "paint") return "paint";
    return null;
  }

  // ---- The lens, each frame ----------------------------------------------------------------

  private readonly frame = (): void => {
    if (this.mode === "off" || this.destroyed) return;
    const camera = this.viewer.camera;
    const canvas = this.viewer.canvas;
    const buffer = { width: this.scene.drawingBufferWidth, height: this.scene.drawingBufferHeight };
    const viewChanged = !Matrix4.equalsEpsilon(camera.viewMatrix, this.lastView, 1e-9);
    const bufferChanged =
      buffer.width !== this.lastBuffer.width || buffer.height !== this.lastBuffer.height;
    if (!this.dirty && !viewChanged && !bufferChanged && !this.restDue) return;
    Matrix4.clone(camera.viewMatrix, this.lastView);
    this.lastBuffer = buffer;
    this.dirty = false;
    if (viewChanged) this.rested = null;
    this.placeLens(canvas);
  };

  private placeLens(canvas: HTMLCanvasElement): void {
    const reach = this.reach;
    const source = this.sources[this.index];
    if (!reach || !source || !this.clipping.lanternSupported) {
      this.hideLens(null);
      return;
    }
    const clientWidth = canvas.clientWidth;
    const clientHeight = canvas.clientHeight;
    const cursor =
      this.pointer && !this.touch ? this.pointer : { x: clientWidth / 2, y: clientHeight / 2 };
    const radii = lensRadii(clientWidth, clientHeight);
    const ground = this.cursorGround(cursor);
    if (!ground) {
      this.waitingForGround = true;
      this.hideLens(null);
      return;
    }
    this.waitingForGround = false;
    const carto = Cartographic.fromCartesian(ground, Ellipsoid.WGS84, scratchCartographic);
    const longitude = CesiumMath.toDegrees(carto.longitude);
    const latitude = CesiumMath.toDegrees(carto.latitude);
    const at = { longitude, latitude };
    if (
      longitude < reach.west ||
      longitude > reach.east ||
      latitude < reach.south ||
      latitude > reach.north
    ) {
      this.hideLens(at);
      return;
    }
    const normal = Ellipsoid.WGS84.geodeticSurfaceNormal(ground, new Cartesian3());
    const base = carto.height;
    const globe = this.scene.globe;
    const ring = groundWindow(
      this.viewer.camera,
      cursor,
      radii.outer,
      { point: ground, normal },
      WINDOW_RAYS,
      WINDOW_MARGIN,
      (point) => {
        const at = Cartographic.fromCartesian(point, Ellipsoid.WGS84, scratchSample);
        at.height = 0;
        const height = globe.getHeight(at);
        return height !== undefined && height > -500 && height < 9000 ? height - base : undefined;
      },
    );
    if (!ring) {
      this.hideLens(at);
      return;
    }
    const size = {
      clientWidth,
      clientHeight,
      drawingBufferWidth: this.scene.drawingBufferWidth,
      drawingBufferHeight: this.scene.drawingBufferHeight,
    };
    const centre = toDrawingBuffer(cursor.x, cursor.y, size);
    const projected = SceneTransforms.worldToWindowCoordinates(this.scene, ground, scratchWindow);
    const limitCentre = projected ? toDrawingBuffer(projected.x, projected.y, size) : centre;
    this.worldShader.setLantern({
      x: centre.x,
      y: centre.y,
      inner: radii.inner * centre.scale,
      outer: radii.outer * centre.scale,
      limit: { x: limitCentre.x, y: limitCentre.y, r: WINDOW_MARGIN * radii.outer * centre.scale },
    });
    if (!this.lensShown || !sameRing(ring, this.ring)) {
      this.clipping.setLanternWindow(ring);
      this.ring = ring;
    }
    this.lensShown = true;
    const part = coverageAt(source.parts, longitude, latitude);
    this.publish({
      lens: { x: cursor.x, y: cursor.y, r: radii.outer },
      ground: at,
      part: part ? { dateLabel: part.dateLabel, accuracyLabel: part.accuracyLabel } : null,
      covered: part !== null,
    });
  }

  /**
   * The ground under the lens's centre: the cursor's ray met with the terrain the globe has
   * loaded (a few CPU height lookups from a plane at the area's height), or the depth buffer's
   * answer once the pointer has rested -- a GPU read-back, so never per move.
   */
  private cursorGround(cursor: { x: number; y: number }): Cartesian3 | null {
    if (this.restDue) {
      this.restDue = false;
      const picked = this.scene.pickPositionSupported
        ? this.scene.pickPosition(new Cartesian2(cursor.x, cursor.y))
        : undefined;
      if (picked) {
        const height = Cartographic.fromCartesian(picked).height;
        if (height > -500 && height < 9000)
          this.rested = { x: cursor.x, y: cursor.y, point: picked };
      }
    }
    const rested = this.rested;
    if (rested?.x === cursor.x && rested.y === cursor.y) return rested.point;
    const ray = this.viewer.camera.getPickRay(new Cartesian2(cursor.x, cursor.y));
    if (!ray) return null;
    let height = this.planeHeight;
    let hit: Cartesian3 | null = null;
    let known = false;
    for (let i = 0; i < 4; i++) {
      hit = intersectHeight(ray.origin, ray.direction, height);
      if (!hit) return null;
      const carto = Cartographic.fromCartesian(hit, Ellipsoid.WGS84, scratchCartographic);
      carto.height = 0;
      const loaded = this.scene.globe.getHeight(carto);
      if (loaded === undefined || !(loaded > -500 && loaded < 9000)) {
        known = false;
        break;
      }
      known = true;
      if (Math.abs(loaded - height) < 0.5) break;
      height = loaded;
    }
    // Only ground the globe has: a guess here puts the window where the lens is not, and the
    // cut would open on the sky. The lens waits for the terrain (`onTilesLoading`).
    return known ? hit : null;
  }

  private hideLens(ground: { longitude: number; latitude: number } | null): void {
    if (this.lensShown) this.clearLens();
    this.publish({ lens: null, ground, part: null, covered: false });
  }

  private clearLens(): void {
    this.lensShown = false;
    this.ring = null;
    this.worldShader.setLantern(null);
    this.clipping.setLanternWindow(null);
  }

  // ---- Imagery ---------------------------------------------------------------------------

  /**
   * The current source on top at full alpha, its neighbours in time loading under it at alpha
   * 0, the "no record" grid under them, all limited to the reach. Everything waits for a basemap
   * first: Cesium stretches the bottom imagery layer over the whole globe, and a dated layer
   * must never be that layer (LandRasterBridge has the same rule).
   */
  private syncImagery(): void {
    if (this.destroyed || this.viewer.isDestroyed()) return;
    if (this.mode === "off" || !this.reach) {
      this.hideImagery();
      return;
    }
    if (this.disposeTimer !== null) clearTimeout(this.disposeTimer);
    this.disposeTimer = null;
    if (!this.basemapReady) {
      this.basemap ??= this.layers.ensureFallbackBasemap().then(() => {
        this.basemapReady = true;
        this.syncImagery();
      });
      return;
    }
    const collection = this.viewer.imageryLayers;
    const reach = Rectangle.fromDegrees(
      this.reach.west,
      this.reach.south,
      this.reach.east,
      this.reach.north,
    );
    if (!this.backdrop) {
      this.backdrop = new ImageryLayer(
        new GridImageryProvider({
          cells: 4,
          color: NO_RECORD_LINES,
          glowColor: Color.TRANSPARENT,
          glowWidth: 0,
          backgroundColor: NO_RECORD,
        }),
        { rectangle: reach },
      );
      collection.add(this.backdrop);
    }
    this.backdrop.show = true;
    const wanted = new Set<string>();
    for (const i of [this.index - 1, this.index, this.index + 1]) {
      const source = this.sources[i];
      if (source) wanted.add(source.id);
    }
    for (const [id, layer] of this.imagery) {
      if (wanted.has(id)) continue;
      collection.remove(layer, true);
      this.imagery.delete(id);
    }
    for (const i of [this.index - 1, this.index + 1, this.index]) {
      const source = this.sources[i];
      if (!source) continue;
      let layer = this.imagery.get(source.id);
      if (!layer) {
        const created = this.createLayer(source, reach);
        if (!created) continue;
        layer = created;
        this.imagery.set(source.id, layer);
        collection.add(layer);
      }
      layer.show = true;
      layer.alpha = i === this.index ? 1 : 0;
    }
    collection.raiseToTop(this.backdrop);
    const current = this.sources[this.index];
    const top = current ? this.imagery.get(current.id) : undefined;
    if (top) collection.raiseToTop(top);
    this.scene.requestRender();
  }

  private createLayer(source: LanternSource, reach: Rectangle): ImageryLayer | null {
    const [west, south, east, north] = source.bounds;
    const rectangle = Rectangle.intersection(
      Rectangle.fromDegrees(west, south, east, north),
      reach,
    );
    if (!rectangle) return null;
    const geographic = source.tiles.scheme === "geographic";
    const maximumLevel = source.tiles.maximumLevel;
    const minimumLevel = Math.min(
      maximumLevel,
      Math.max(minimumLevelFor(rectangle, geographic), geographic ? 0 : WEB_MERCATOR_MIN_LEVEL),
    );
    const provider = new UrlTemplateImageryProvider({
      url: new Resource({ url: source.tiles.urlTemplate, headers: source.tiles.headers }),
      tilingScheme: geographic ? new GeographicTilingScheme() : new WebMercatorTilingScheme(),
      rectangle,
      minimumLevel,
      maximumLevel,
      enablePickFeatures: false,
      credit: new Credit(source.attribution),
    });
    // A tile a server has no picture for (a mosaic's edge, a sheet's corner) is "no record":
    // counted, not logged per tile.
    let failures = 0;
    provider.errorEvent.addEventListener(() => {
      failures += 1;
      if (failures === 20) log.warn("dated imagery tiles failing", { source: source.id });
    });
    return new ImageryLayer(provider, { rectangle, show: false, alpha: 0 });
  }

  /** Hidden at once, destroyed after a minute closed. */
  private hideImagery(): void {
    for (const layer of this.imagery.values()) layer.show = false;
    if (this.backdrop) this.backdrop.show = false;
    if (this.disposeTimer === null && (this.imagery.size > 0 || this.backdrop))
      this.disposeTimer = setTimeout(() => {
        this.disposeTimer = null;
        this.disposeImagery();
      }, IDLE_DISPOSE_MS);
    this.scene.requestRender();
  }

  private disposeImagery(): void {
    if (this.viewer.isDestroyed()) return;
    const collection = this.viewer.imageryLayers;
    for (const layer of this.imagery.values()) collection.remove(layer, true);
    this.imagery.clear();
    if (this.backdrop) collection.remove(this.backdrop, true);
    this.backdrop = null;
  }

  // ---- Publishing ----------------------------------------------------------------------------

  /**
   * Tells the HUD what the lantern is doing: on every change of mode, source or availability,
   * and on every frame the lens moves (`lens`). Without `lens`, the last lens is repeated.
   */
  private publish(lens?: Pick<LanternFrame, "lens" | "ground" | "part" | "covered">): void {
    const source = this.sources[this.index];
    const placed = lens ??
      this.lastFrame ?? { lens: null, ground: null, part: null, covered: false };
    const frame: LanternFrame = {
      mode: this.mode,
      index: this.index,
      availability: this.availability,
      lens: this.mode === "off" ? null : placed.lens,
      ground: placed.ground,
      source: source
        ? {
            id: source.id,
            kind: source.kind,
            title: source.title,
            dateLabel: source.dateLabel,
            license: source.license,
            accuracyLabel: source.accuracyLabel,
            sortDate: source.sortDate,
          }
        : null,
      part: placed.part,
      covered: placed.covered,
    };
    this.lastFrame = frame;
    this.events.emit("lantern", frame);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.setMode("off");
    this.destroyed = true;
    for (const off of this.off) off();
    if (this.restTimer !== null) clearTimeout(this.restTimer);
    if (this.disposeTimer !== null) clearTimeout(this.disposeTimer);
    this.wheelOff?.();
    this.wheelOff = null;
    this.disposeImagery();
  }
}

/** Whether two windows are the same within half a metre at every corner. */
function sameRing(a: Cartesian3[], b: Cartesian3[] | null): boolean {
  if (b?.length !== a.length) return false;
  for (let i = 0; i < a.length; i++) {
    const p = a[i];
    const q = b[i];
    if (!p || !q || Cartesian3.distanceSquared(p, q) > 0.25) return false;
  }
  return true;
}

/**
 * Where a ray meets the surface `height` metres above the ellipsoid, approximated by the sphere
 * through the ray's foot at that height: within centimetres over the few kilometres a lens
 * spans. Null when the ray points away from it.
 */
function intersectHeight(
  origin: Cartesian3,
  direction: Cartesian3,
  height: number,
): Cartesian3 | null {
  const foot = Ellipsoid.WGS84.scaleToGeodeticSurface(origin, new Cartesian3());
  if (!foot) return null;
  const radius = Cartesian3.magnitude(foot) + height;
  // |o + t d|² = r²  →  t² + 2 (o·d) t + |o|² − r² = 0, d unit length.
  const b = Cartesian3.dot(origin, direction);
  const c = Cartesian3.magnitudeSquared(origin) - radius * radius;
  const discriminant = b * b - c;
  if (discriminant < 0) return null;
  const root = Math.sqrt(discriminant);
  const t = -b - root > 0 ? -b - root : -b + root;
  if (t <= 0) return null;
  return Cartesian3.add(
    origin,
    Cartesian3.multiplyByScalar(direction, t, new Cartesian3()),
    new Cartesian3(),
  );
}
