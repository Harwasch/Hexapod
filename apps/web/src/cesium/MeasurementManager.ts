import {
  CallbackProperty,
  Cartesian2,
  Cartesian3,
  Cartographic,
  Color,
  EllipsoidGeodesic,
  type Entity,
  HeightReference,
  HorizontalOrigin,
  LabelStyle,
  Math as CesiumMath,
  PolygonHierarchy,
  ScreenSpaceEventHandler,
  ScreenSpaceEventType,
  VerticalOrigin,
  sampleTerrainMostDetailed,
  type Scene,
  type Viewer,
} from "cesium";

import { formatArea, formatLength, ringAreaM2, type UnitSystem } from "@twin/geo";

import type { Emitter } from "@/lib/emitter";
import type { Measurement, MeasurementPoint } from "@/state/measurements";
import type { MeasureMode } from "@/state/ui";

import type { SceneEvents } from "./types";

const LINE = Color.fromCssColorString("#ffd60a");
const FILL = LINE.withAlpha(0.22);
const LABEL_BG = Color.fromCssColorString("#0f172a").withAlpha(0.72);

interface Draft {
  id: string;
  mode: MeasureMode;
  points: Cartesian3[];
  floating: Cartesian3 | null;
  entities: Entity[];
}

/**
 * Point, distance, area, height and elevation measurements drawn in the scene
 * as entities. Results are emitted to React; entities are owned here.
 */
export class MeasurementManager {
  private readonly scene: Scene;
  private handler: ScreenSpaceEventHandler | null = null;
  private mode: MeasureMode | null = null;
  private draft: Draft | null = null;
  private readonly finished = new Map<string, Entity[]>();
  private units: UnitSystem = "metric";
  private counter = 0;
  private readonly onKey = (event: KeyboardEvent) => {
    if (event.key === "Escape" && this.mode) {
      event.preventDefault();
      this.stop();
    }
    if (event.key === "Enter" && this.draft?.mode === "area") this.finishArea();
  };

  constructor(
    private readonly viewer: Viewer,
    private readonly events: Emitter<SceneEvents>,
  ) {
    this.scene = viewer.scene;
  }

  get activeMode(): MeasureMode | null {
    return this.mode;
  }

  setUnits(units: UnitSystem): void {
    this.units = units;
    this.scene.requestRender();
  }

  start(mode: MeasureMode): void {
    this.stop();
    this.mode = mode;
    this.viewer.canvas.style.cursor = "crosshair";
    this.handler = new ScreenSpaceEventHandler(this.viewer.canvas);
    this.handler.setInputAction(
      (e: ScreenSpaceEventHandler.PositionedEvent) => this.onClick(e.position),
      ScreenSpaceEventType.LEFT_CLICK,
    );
    this.handler.setInputAction(
      (e: ScreenSpaceEventHandler.MotionEvent) => this.onMove(e.endPosition),
      ScreenSpaceEventType.MOUSE_MOVE,
    );
    this.handler.setInputAction(() => this.finishArea(), ScreenSpaceEventType.RIGHT_CLICK);
    this.handler.setInputAction(() => this.finishArea(), ScreenSpaceEventType.LEFT_DOUBLE_CLICK);
    window.addEventListener("keydown", this.onKey);
  }

  stop(): void {
    if (this.draft) this.discardDraft();
    this.handler?.destroy();
    this.handler = null;
    window.removeEventListener("keydown", this.onKey);
    this.viewer.canvas.style.cursor = "";
    if (this.mode) {
      this.mode = null;
      this.events.emit("measurement-mode", null);
    }
  }

  remove(id: string): void {
    const entities = this.finished.get(id);
    if (entities) {
      for (const entity of entities) this.viewer.entities.remove(entity);
      this.finished.delete(id);
      this.scene.requestRender();
    }
  }

  clearAll(): void {
    for (const id of Array.from(this.finished.keys())) this.remove(id);
  }

  private pick(window: Cartesian2): Cartesian3 | null {
    let position: Cartesian3 | undefined;
    if (this.scene.pickPositionSupported) position = this.scene.pickPosition(window);
    if (!position) {
      const ray = this.viewer.camera.getPickRay(window);
      position = ray ? this.scene.globe.pick(ray, this.scene) : undefined;
    }
    return position ?? null;
  }

  private onClick(window: Cartesian2): void {
    const position = this.pick(window);
    if (!position || !this.mode) return;
    if (this.mode === "point" || this.mode === "elevation") {
      const draft = this.newDraft(this.mode);
      draft.points.push(position);
      this.addPoint(draft, position);
      this.complete(draft);
      return;
    }
    if (this.mode === "distance" || this.mode === "height") {
      if (!this.draft) {
        const draft = this.newDraft(this.mode);
        draft.points.push(position);
        this.addPoint(draft, position);
        this.addLine(draft);
        return;
      }
      this.draft.points.push(position);
      this.addPoint(this.draft, position);
      this.draft.floating = null;
      this.complete(this.draft);
      return;
    }
    if (this.mode === "area") {
      if (!this.draft) {
        const draft = this.newDraft("area");
        this.addPolygon(draft);
      }
      this.draft?.points.push(position);
      if (this.draft) this.addPoint(this.draft, position);
    }
  }

  private onMove(window: Cartesian2): void {
    if (!this.draft || this.draft.points.length === 0) return;
    const position = this.pick(window);
    if (position) {
      this.draft.floating = position;
      this.scene.requestRender();
    }
  }

  private finishArea(): void {
    if (this.draft?.mode !== "area") return;
    if (this.draft.points.length < 3) {
      this.discardDraft();
      return;
    }
    this.draft.floating = null;
    this.complete(this.draft);
  }

  private newDraft(mode: MeasureMode): Draft {
    this.draft = {
      id: `m-${Date.now()}-${++this.counter}`,
      mode,
      points: [],
      floating: null,
      entities: [],
    };
    return this.draft;
  }

  private discardDraft(): void {
    if (!this.draft) return;
    for (const entity of this.draft.entities) this.viewer.entities.remove(entity);
    this.draft = null;
    this.scene.requestRender();
  }

  private positions(draft: Draft): Cartesian3[] {
    return draft.floating ? [...draft.points, draft.floating] : draft.points;
  }

  private addPoint(draft: Draft, position: Cartesian3): void {
    draft.entities.push(
      this.viewer.entities.add({
        position,
        point: {
          pixelSize: 8,
          color: LINE,
          outlineColor: Color.BLACK.withAlpha(0.6),
          outlineWidth: 1.5,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
          heightReference: HeightReference.NONE,
        },
      }),
    );
  }

  private addLine(draft: Draft): void {
    draft.entities.push(
      this.viewer.entities.add({
        polyline: {
          positions: new CallbackProperty(() => this.positions(draft), false),
          width: 3,
          material: LINE,
          depthFailMaterial: LINE.withAlpha(0.35),
        },
      }),
      this.viewer.entities.add({
        position: new CallbackProperty(() => this.midpoint(this.positions(draft)), false) as never,
        label: this.label(() => this.describe(draft)),
      }),
    );
  }

  private addPolygon(draft: Draft): void {
    draft.entities.push(
      this.viewer.entities.add({
        polygon: {
          hierarchy: new CallbackProperty(() => new PolygonHierarchy(this.positions(draft)), false),
          material: FILL,
          perPositionHeight: false,
          outline: false,
        },
      }),
      this.viewer.entities.add({
        polyline: {
          positions: new CallbackProperty(() => {
            const p = this.positions(draft);
            const first = p[0];
            return p.length > 1 && first ? [...p, first] : p;
          }, false),
          width: 2.5,
          material: LINE,
          clampToGround: true,
        },
      }),
      this.viewer.entities.add({
        position: new CallbackProperty(() => this.midpoint(this.positions(draft)), false) as never,
        label: this.label(() => this.describe(draft)),
      }),
    );
  }

  private label(text: () => string) {
    return {
      text: new CallbackProperty(text, false),
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
    };
  }

  private midpoint(positions: Cartesian3[]): Cartesian3 | undefined {
    if (positions.length === 0) return undefined;
    const sum = positions.reduce((acc, p) => Cartesian3.add(acc, p, acc), new Cartesian3());
    return Cartesian3.divideByScalar(sum, positions.length, sum);
  }

  private toPoints(positions: Cartesian3[]): MeasurementPoint[] {
    return positions.map((p) => {
      const c = Cartographic.fromCartesian(p);
      return {
        longitude: CesiumMath.toDegrees(c.longitude),
        latitude: CesiumMath.toDegrees(c.latitude),
        height: c.height,
      };
    });
  }

  private compute(mode: MeasureMode, positions: Cartesian3[]): Partial<Measurement> {
    const points = this.toPoints(positions);
    if (mode === "distance" && positions.length >= 2) {
      let d2 = 0;
      let d3 = 0;
      for (let i = 1; i < positions.length; i++) {
        const a = positions[i - 1];
        const b = positions[i];
        if (!a || !b) continue;
        d3 += Cartesian3.distance(a, b);
        const geodesic = new EllipsoidGeodesic(
          Cartographic.fromCartesian(a),
          Cartographic.fromCartesian(b),
        );
        d2 += geodesic.surfaceDistance;
      }
      return { distance2dM: d2, distance3dM: d3 };
    }
    if (mode === "area" && points.length >= 3) {
      const ring = points.map((p) => [p.longitude, p.latitude]);
      const first = ring[0];
      if (first) ring.push(first);
      return { areaM2: Math.abs(ringAreaM2(ring)) };
    }
    const [a, b] = points;
    const [pa, pb] = positions;
    if (mode === "height" && a && b && pa && pb) {
      return { heightDeltaM: b.height - a.height, distance3dM: Cartesian3.distance(pa, pb) };
    }
    if (mode === "elevation" && points[0]) {
      return { elevationM: points[0].height };
    }
    return {};
  }

  private describe(draft: Draft): string {
    const positions = this.positions(draft);
    const result = this.compute(draft.mode, positions);
    if (draft.mode === "distance" && result.distance2dM !== undefined) {
      return `${formatLength(result.distance3dM ?? 0, this.units)}  ·  ground ${formatLength(result.distance2dM, this.units)}`;
    }
    if (draft.mode === "area" && result.areaM2 !== undefined)
      return formatArea(result.areaM2, this.units);
    if (draft.mode === "height" && result.heightDeltaM !== undefined)
      return `Δh ${formatLength(result.heightDeltaM, this.units)}`;
    return "";
  }

  private complete(draft: Draft): void {
    const measurement: Measurement = {
      id: draft.id,
      mode: draft.mode,
      points: this.toPoints(draft.points),
      complete: true,
      createdAt: Date.now(),
      ...this.compute(draft.mode, draft.points),
    };
    const point = draft.points[0];
    if ((draft.mode === "point" || draft.mode === "elevation") && point) {
      draft.entities.push(
        this.viewer.entities.add({
          position: point,
          label: this.label(() => {
            const p = measurement.points[0];
            if (!p) return "";
            const elevation = measurement.elevationM ?? p.height;
            return draft.mode === "elevation"
              ? `Elevation ${formatLength(elevation, this.units)}`
              : `${p.latitude.toFixed(6)}, ${p.longitude.toFixed(6)}\n${formatLength(p.height, this.units)}`;
          }),
        }),
      );
      void this.refineElevation(measurement);
    }
    this.finished.set(draft.id, draft.entities);
    this.draft = null;
    this.events.emit("measurement", measurement);
    this.scene.requestRender();
  }

  private async refineElevation(measurement: Measurement): Promise<void> {
    const point = measurement.points[0];
    if (!point) return;
    try {
      const [sample] = await sampleTerrainMostDetailed(this.scene.terrainProvider, [
        Cartographic.fromDegrees(point.longitude, point.latitude),
      ]);
      if (sample && Number.isFinite(sample.height)) {
        this.events.emit("measurement", { ...measurement, elevationM: sample.height });
      }
    } catch {
      // Flat ellipsoid terrain has nothing to sample; the picked height stands.
    }
  }

  destroy(): void {
    this.stop();
    this.clearAll();
  }
}
