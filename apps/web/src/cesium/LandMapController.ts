import {
  Cartesian2,
  Cartesian3,
  Cartographic,
  Color,
  ColorMaterialProperty,
  ConstantPositionProperty,
  Entity,
  HeightReference,
  Math as CesiumMath,
  PolygonHierarchy,
  ScreenSpaceEventHandler,
  ScreenSpaceEventType,
} from "cesium";

import type { Footprint } from "@twin/contracts";
import { boundsOf, polygonsOf } from "@twin/geo";

import { MappedSnapIndex } from "@/features/land/mappedSnap";
import { useLandContext, type LandContextLayer } from "@/state/landContext";
import { nearestBoundaryPoint } from "@/features/land/boundarySnap";

import type { LandMode, LandPoint } from "@/state/land";

import type { CesiumSceneManager } from "./CesiumSceneManager";

const FILL = Color.fromCssColorString("#a8d7b4").withAlpha(0.13);
const LINE = Color.fromCssColorString("#b9edc6");
const PREFIX = "land-boundary:";

/** Land display persists while ordinary feature inspection continues beside it. */
export class LandMapController {
  private entities: Entity[] = [];
  private handler: ScreenSpaceEventHandler | null = null;
  private footprint: Footprint | null = null;
  private mode: LandMode = "browse";
  private dragging: { polygon: number; ring: number; vertex: number } | null = null;
  private cameraInputs = true;
  private snapEnabled = true;
  private snapMapped = true;
  private readonly mappedSnap = new MappedSnapIndex();
  private snapMarker: Entity | null = null;
  private readonly release = () => this.finishDrag();

  constructor(
    private readonly host: CesiumSceneManager,
    private readonly point: (point: LandPoint) => void,
    private readonly change: (boundary: Footprint) => void,
    private readonly snapTarget?: (label: string | null) => void,
  ) {
    window.addEventListener("pointerup", this.release);
    window.addEventListener("blur", this.release);
  }

  private groundPoint(position: Cartesian2): LandPoint | null {
    const { scene, viewer } = this.host;
    const ray = viewer.camera.getPickRay(position);
    const hit = ray ? scene.globe.pick(ray, scene) : undefined;
    const world = hit ?? viewer.camera.pickEllipsoid(position, scene.globe.ellipsoid);
    if (!world) return null;
    const at = Cartographic.fromCartesian(world);
    const point: LandPoint = [
      CesiumMath.toDegrees(at.longitude),
      CesiumMath.toDegrees(at.latitude),
    ];
    return point;
  }

  private ground(position: Cartesian2): LandPoint | null {
    const point = this.groundPoint(position);
    if (!point) {
      this.showSnap(null);
      return null;
    }
    if (this.mode === "pick" || this.mode === "candidates") return point;
    const { scene } = this.host;
    const distance = (target: LandPoint) => {
      const height = scene.globe.getHeight(Cartographic.fromDegrees(...target)) ?? 0;
      const screen = scene.cartesianToCanvasCoordinates(Cartesian3.fromDegrees(...target, height));
      return screen ? Math.hypot(screen.x - position.x, screen.y - position.y) : Infinity;
    };
    const boundary =
      this.snapEnabled && this.footprint
        ? nearestBoundaryPoint(this.footprint, point, this.dragging)
        : null;
    let best = boundary
      ? { point: boundary, label: "This boundary", distance: distance(boundary) }
      : null;
    if (this.snapMapped && !this.mappedSnap.empty) {
      const bounds = { minX: point[0], minY: point[1], maxX: point[0], maxY: point[1] };
      for (const x of [-24, 0, 24])
        for (const y of [-24, 0, 24]) {
          if (!x && !y) continue;
          const sample = this.groundPoint(new Cartesian2(position.x + x, position.y + y));
          if (!sample || Math.abs(sample[0] - point[0]) > 180) continue;
          bounds.minX = Math.min(bounds.minX, sample[0]);
          bounds.maxX = Math.max(bounds.maxX, sample[0]);
          bounds.minY = Math.min(bounds.minY, sample[1]);
          bounds.maxY = Math.max(bounds.maxY, sample[1]);
        }
      const mapped = this.mappedSnap.nearest(point, bounds, distance);
      if (mapped && (!best || mapped.distance < best.distance)) best = mapped;
    }
    if (best && best.distance <= 12) {
      this.showSnap(best.point, best.label);
      return best.point;
    }
    this.showSnap(null);
    return point;
  }

  setSnapping(enabled: boolean, mapped = true): void {
    if (this.snapEnabled === enabled && this.snapMapped === mapped) return;
    this.snapEnabled = enabled;
    this.snapMapped = mapped;
    this.showSnap(null);
  }

  setSnapLayers(layers: Record<string, LandContextLayer>): void {
    if (this.mappedSnap.sync(layers)) this.showSnap(null);
  }

  private showSnap(point: LandPoint | null, label: string | null = null): void {
    this.snapTarget?.(point ? label : null);
    if (!point && !this.snapMarker) return;
    if (point && this.snapMarker) {
      this.snapMarker.position = new ConstantPositionProperty(Cartesian3.fromDegrees(...point));
      this.host.scene.requestRender();
      return;
    }
    if (this.snapMarker) this.host.viewer.entities.remove(this.snapMarker);
    this.snapMarker = point
      ? this.host.viewer.entities.add({
          id: `${PREFIX}snap`,
          position: Cartesian3.fromDegrees(...point),
          point: {
            pixelSize: 17,
            color: Color.TRANSPARENT,
            outlineColor: Color.GOLD,
            outlineWidth: 3,
            heightReference: HeightReference.CLAMP_TO_GROUND,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
        })
      : null;
    if (!this.host.isDestroyed) this.host.scene.requestRender();
  }

  setMode(mode: LandMode): void {
    if (this.mode === mode) return;
    this.finishDrag();
    this.mode = mode;
    this.showSnap(null);
    this.handler?.destroy();
    this.handler = null;
    const canvas = this.host.scene.canvas;
    canvas.style.cursor = mode === "browse" ? "" : "crosshair";
    if (mode === "browse") return;
    const handler = new ScreenSpaceEventHandler(canvas);
    this.handler = handler;
    handler.setInputAction((event: ScreenSpaceEventHandler.PositionedEvent) => {
      if (this.mode === "edit") return;
      if (this.mode === "candidates") {
        const picks: unknown[] = this.host.scene.drillPick(event.position, 32);
        const picked = picks.find(
          (candidate) =>
            candidate &&
            typeof candidate === "object" &&
            "id" in candidate &&
            candidate.id instanceof Entity &&
            candidate.id.id.startsWith("land-context:candidates/"),
        );
        if (
          picked &&
          typeof picked === "object" &&
          "id" in picked &&
          picked.id instanceof Entity &&
          picked.id.id.startsWith("land-context:candidates/")
        ) {
          const id = picked.id.id.slice("land-context:candidates/".length).split("#")[0];
          if (id) useLandContext.getState().toggleCandidate(id);
          return;
        }
      }
      const point = this.ground(event.position);
      if (point) this.point(point);
    }, ScreenSpaceEventType.LEFT_CLICK);
    handler.setInputAction((event: ScreenSpaceEventHandler.PositionedEvent) => {
      if (this.mode !== "edit") return;
      const picked: unknown = this.host.scene.pick(event.position);
      if (
        !picked ||
        typeof picked !== "object" ||
        !("id" in picked) ||
        !(picked.id instanceof Entity)
      )
        return;
      const match = /^land-boundary:handle:(\d+):(\d+):(\d+)$/.exec(picked.id.id);
      if (!match) return;
      this.dragging = {
        polygon: Number(match[1]),
        ring: Number(match[2]),
        vertex: Number(match[3]),
      };
      this.cameraInputs = this.host.scene.screenSpaceCameraController.enableInputs;
      this.host.scene.screenSpaceCameraController.enableInputs = false;
    }, ScreenSpaceEventType.LEFT_DOWN);
    handler.setInputAction((event: ScreenSpaceEventHandler.MotionEvent) => {
      if (this.mode === "edit" && !this.dragging) return;
      const point = this.ground(event.endPosition);
      if (!point || !this.dragging || !this.footprint) return;
      const copy = structuredClone(this.footprint);
      const { polygon, ring, vertex } = this.dragging;
      const coordinates = polygonsOf(copy)[polygon]?.[ring];
      if (!coordinates) return;
      coordinates[vertex] = point;
      if (vertex === 0) coordinates[coordinates.length - 1] = [...point];
      this.show(copy, [], true);
    }, ScreenSpaceEventType.MOUSE_MOVE);
    handler.setInputAction(() => this.finishDrag(), ScreenSpaceEventType.LEFT_UP);
  }

  private finishDrag(): void {
    if (!this.dragging) return;
    this.dragging = null;
    this.showSnap(null);
    this.host.scene.screenSpaceCameraController.enableInputs = this.cameraInputs;
    if (this.footprint) this.change(this.footprint);
  }

  show(footprint: Footprint | null, points: LandPoint[], editing: boolean): void {
    this.footprint = footprint;
    this.entities.forEach((entity) => this.host.viewer.entities.remove(entity));
    this.entities = [];
    const add = (options: Entity.ConstructorOptions) => {
      this.entities.push(this.host.viewer.entities.add(options));
    };
    const positions = (ring: number[][]) =>
      Cartesian3.fromDegreesArray(ring.flatMap((p) => [p[0] ?? 0, p[1] ?? 0]));
    if (footprint) {
      polygonsOf(footprint).forEach((polygon, pi) => {
        const outer = polygon[0];
        if (!outer) return;
        add({
          id: `${PREFIX}fill:${pi}`,
          polygon: {
            hierarchy: new PolygonHierarchy(
              positions(outer),
              polygon.slice(1).map((ring) => new PolygonHierarchy(positions(ring))),
            ),
            material: FILL,
          },
        });
        polygon.forEach((ring, ri) => {
          add({
            id: `${PREFIX}ring:${pi}:${ri}`,
            polyline: { positions: positions(ring), width: 2, material: LINE, clampToGround: true },
          });
          if (editing)
            ring.slice(0, -1).forEach((point, vi) =>
              add({
                id: `${PREFIX}handle:${pi}:${ri}:${vi}`,
                position: Cartesian3.fromDegrees(point[0] ?? 0, point[1] ?? 0),
                point: {
                  pixelSize: 11,
                  color: Color.WHITE,
                  outlineColor: LINE,
                  outlineWidth: 2,
                  heightReference: HeightReference.CLAMP_TO_GROUND,
                  disableDepthTestDistance: Number.POSITIVE_INFINITY,
                },
              }),
            );
        });
      });
    }
    if (points.length >= 2)
      add({
        id: `${PREFIX}drawing`,
        polyline: { positions: positions(points), width: 3, material: LINE, clampToGround: true },
      });
    if (points.length >= 3 && this.mode === "draw")
      add({
        id: `${PREFIX}draft-fill`,
        polygon: {
          hierarchy: new PolygonHierarchy(positions(points)),
          material: new ColorMaterialProperty(FILL),
        },
      });
    points.forEach((point, index) =>
      add({
        id: `${PREFIX}drawing:${index}`,
        position: Cartesian3.fromDegrees(...point),
        point: {
          pixelSize: 8,
          color: LINE,
          heightReference: HeightReference.CLAMP_TO_GROUND,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      }),
    );
    this.host.scene.requestRender();
  }

  frame(footprint: Footprint): void {
    const b = boundsOf(footprint);
    this.host.camera.flyToRectangle(b.west, b.south, b.east, b.north);
  }

  destroy(): void {
    this.finishDrag();
    this.handler?.destroy();
    this.showSnap(null);
    this.entities.forEach((entity) => this.host.viewer.entities.remove(entity));
    this.host.scene.canvas.style.cursor = "";
    window.removeEventListener("pointerup", this.release);
    window.removeEventListener("blur", this.release);
  }
}
