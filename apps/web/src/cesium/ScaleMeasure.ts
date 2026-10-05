/**
 * A length on a scan, for the Set real size tool (features/inspector/RealSize.tsx): two points
 * picked on the scan as it is drawn, whose distance the person then says the true length of.
 *
 * A point is where the ray through it meets the scan's solids (SplatCollider), the picking the
 * map's other tools use on a scan: a splat writes no depth, so the depth buffer under a scan is
 * the ground beneath it, and a length "on" the scan read from it would be the ground's. A click
 * that meets no splat of the scan is not a point.
 *
 * Each point is kept in the scan's own frame (the root's transform under the model matrix), so
 * the markers stay on the scan while a preview resizes it, and `lengthM` is always the length
 * as drawn now: measured at one scale, it reads the true length once the preview has the scan
 * at the scale that length says.
 *
 * Mouse and touch: a click or a tap. Keyboard: `markCentre` puts a point where the centre of
 * the view meets the scan (the tool's "Mark the centre" button), the view steered with the keys.
 * While points are being picked, selection keeps its hands off the click (`ground-pick-mode`).
 */

import {
  CallbackPositionProperty,
  CallbackProperty,
  Cartesian2,
  Cartesian3,
  Color,
  type Cesium3DTileset,
  type CesiumWidget,
  type Entity,
  HorizontalOrigin,
  LabelStyle,
  Matrix4,
  ScreenSpaceEventHandler,
  ScreenSpaceEventType,
  VerticalOrigin,
} from "cesium";

import { formatLength, type UnitSystem } from "@twin/geo";

import type { Emitter } from "@/lib/emitter";

import type { SplatCollider } from "./SplatCollider";
import { inverseScaledTransformation } from "./tilesetScale";
import type { SceneEvents } from "./types";

const LINE = Color.fromCssColorString("#5eead4");
const LABEL_BG = Color.fromCssColorString("#0f172a").withAlpha(0.72);

/** What the tool shows: whether it is waiting for a point, how many it has, the length. */
export interface ScaleMeasureState {
  readonly picking: boolean;
  readonly points: number;
  /** Metres between the two points as the scan is drawn now; null until both are placed. */
  readonly lengthM: number | null;
  /** The last click missed the scan (no splat under it). */
  readonly missed: boolean;
}

/** Where the scan is: the tileset whose solids a point must be on. */
export type ScanOf = (siteId: string) => Cesium3DTileset | null;

const IDLE: ScaleMeasureState = { picking: false, points: 0, lengthM: null, missed: false };

export class ScaleMeasure {
  private handler: ScreenSpaceEventHandler | null = null;
  private tileset: Cesium3DTileset | null = null;
  /** The points, in the scan's own frame. */
  private readonly local: Cartesian3[] = [];
  private readonly entities: Entity[] = [];
  private listener: ((state: ScaleMeasureState) => void) | null = null;
  private missed = false;
  private units: UnitSystem = "metric";

  constructor(
    private readonly viewer: CesiumWidget,
    private readonly events: Emitter<SceneEvents>,
    private readonly collider: SplatCollider,
    private readonly scanOf: ScanOf,
  ) {}

  get state(): ScaleMeasureState {
    if (!this.tileset && this.local.length === 0) return IDLE;
    return {
      picking: this.handler !== null,
      points: this.local.length,
      lengthM: this.lengthM(),
      missed: this.missed,
    };
  }

  setUnits(units: UnitSystem): void {
    this.units = units;
  }

  /**
   * Starts picking two points on `siteId`'s scan, dropping any picked before; `onChange` hears
   * every change. False when the site has no scan in the scene to measure.
   */
  start(siteId: string, onChange: (state: ScaleMeasureState) => void): boolean {
    this.clear();
    const tileset = this.scanOf(siteId);
    if (!tileset || tileset.isDestroyed()) return false;
    this.tileset = tileset;
    this.listener = onChange;
    const canvas = this.viewer.canvas;
    const handler = new ScreenSpaceEventHandler(canvas);
    handler.setInputAction((event: ScreenSpaceEventHandler.PositionedEvent) => {
      this.markAt(event.position.x, event.position.y);
    }, ScreenSpaceEventType.LEFT_CLICK);
    this.handler = handler;
    canvas.style.cursor = "crosshair";
    this.events.emit("ground-pick-mode", true);
    this.addEntities();
    this.changed();
    return true;
  }

  /**
   * Puts the next point where the scan is under (`x`, `y`), CSS px from the canvas's top left.
   * The third starts a new length. False when no splat of the scan is there.
   */
  markAt(x: number, y: number): boolean {
    const tileset = this.tileset;
    if (!this.handler || !tileset || tileset.isDestroyed()) return false;
    const ray = this.viewer.camera.getPickRay(new Cartesian2(x, y));
    const hit = ray ? this.collider.raycast(ray) : null;
    const sphere = tileset.boundingSphere;
    // On this scan's solids, not another scan's beside it.
    const onScan = hit && Cartesian3.distance(hit.point, sphere.center) <= sphere.radius * 1.01;
    if (!hit || !onScan) {
      this.missed = true;
      this.changed();
      return false;
    }
    if (this.local.length >= 2) this.local.length = 0;
    const toLocal = inverseScaledTransformation(this.toWorld(tileset), new Matrix4());
    this.local.push(Matrix4.multiplyByPoint(toLocal, hit.point, new Cartesian3()));
    this.missed = false;
    if (this.local.length === 2) this.stopPicking();
    this.changed();
    return true;
  }

  /** Puts the next point where the centre of the view meets the scan: the keyboard's way. */
  markCentre(): boolean {
    const canvas = this.viewer.canvas;
    return this.markAt(canvas.clientWidth / 2, canvas.clientHeight / 2);
  }

  /** Picks the two points again, from none. */
  restart(): boolean {
    const tileset = this.tileset;
    const listener = this.listener;
    if (!tileset || !listener) return false;
    this.local.length = 0;
    if (!this.handler) {
      const handler = new ScreenSpaceEventHandler(this.viewer.canvas);
      handler.setInputAction((event: ScreenSpaceEventHandler.PositionedEvent) => {
        this.markAt(event.position.x, event.position.y);
      }, ScreenSpaceEventType.LEFT_CLICK);
      this.handler = handler;
      this.viewer.canvas.style.cursor = "crosshair";
      this.events.emit("ground-pick-mode", true);
    }
    this.changed();
    return true;
  }

  /** The two points' distance as the scan is drawn now, metres; null until both are placed. */
  lengthM(): number | null {
    const [a, b] = this.worldPoints();
    return a && b ? Cartesian3.distance(a, b) : null;
  }

  /** Stops picking and takes the markers away. */
  clear(): void {
    this.stopPicking();
    for (const entity of this.entities) this.viewer.entities.remove(entity);
    this.entities.length = 0;
    this.local.length = 0;
    this.tileset = null;
    this.missed = false;
    const listener = this.listener;
    this.listener = null;
    listener?.(IDLE);
    this.viewer.scene.requestRender();
  }

  destroy(): void {
    this.clear();
  }

  private stopPicking(): void {
    if (!this.handler) return;
    this.handler.destroy();
    this.handler = null;
    this.viewer.canvas.style.cursor = "";
    this.events.emit("ground-pick-mode", false);
  }

  private changed(): void {
    this.viewer.scene.requestRender();
    this.listener?.(this.state);
  }

  /** The scan's frame to the globe now: its model matrix over its root's own transform. */
  private toWorld(tileset: Cesium3DTileset): Matrix4 {
    const root = (tileset.root as { transform?: Matrix4 } | undefined)?.transform;
    return Matrix4.multiply(tileset.modelMatrix, root ?? Matrix4.IDENTITY, new Matrix4());
  }

  private worldPoints(): Cartesian3[] {
    const tileset = this.tileset;
    if (!tileset || tileset.isDestroyed()) return [];
    const toWorld = this.toWorld(tileset);
    return this.local.map((p) => Matrix4.multiplyByPoint(toWorld, p, new Cartesian3()));
  }

  private addEntities(): void {
    const entities = this.viewer.entities;
    for (const index of [0, 1]) {
      this.entities.push(
        entities.add({
          position: new CallbackPositionProperty(() => this.worldPoints()[index], false),
          point: {
            pixelSize: 9,
            color: LINE,
            outlineColor: Color.BLACK.withAlpha(0.6),
            outlineWidth: 1.5,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
        }),
      );
    }
    this.entities.push(
      entities.add({
        polyline: {
          positions: new CallbackProperty(() => this.worldPoints(), false),
          width: 3,
          material: LINE,
          depthFailMaterial: LINE.withAlpha(0.45),
        },
      }),
      entities.add({
        position: new CallbackPositionProperty(() => {
          const [a, b] = this.worldPoints();
          return a && b ? Cartesian3.midpoint(a, b, new Cartesian3()) : undefined;
        }, false),
        label: {
          text: new CallbackProperty(() => {
            const length = this.lengthM();
            return length === null ? "" : formatLength(length, this.units);
          }, false),
          font: "600 15px 'Azeret Mono', ui-monospace, SFMono-Regular, Menlo, monospace",
          style: LabelStyle.FILL,
          fillColor: Color.WHITE,
          showBackground: true,
          backgroundColor: LABEL_BG,
          backgroundPadding: new Cartesian2(10, 6),
          horizontalOrigin: HorizontalOrigin.CENTER,
          verticalOrigin: VerticalOrigin.BOTTOM,
          pixelOffset: new Cartesian2(0, -12),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      }),
    );
  }
}
