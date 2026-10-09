import {
  type Cartesian2,
  Cartesian3,
  Cartographic,
  Color,
  ColorMaterialProperty,
  Entity,
  HeightReference,
  Math as CesiumMath,
  PolygonHierarchy,
  ScreenSpaceEventHandler,
  ScreenSpaceEventType,
} from "cesium";

import type { Footprint } from "@twin/contracts";
import { boundsOf, polygonsOf } from "@twin/geo";

import { useLandContext } from "@/state/landContext";

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
  private readonly release = () => this.finishDrag();

  constructor(
    private readonly host: CesiumSceneManager,
    private readonly point: (point: LandPoint) => void,
    private readonly change: (boundary: Footprint) => void,
  ) {
    window.addEventListener("pointerup", this.release);
    window.addEventListener("blur", this.release);
  }

  private ground(position: Cartesian2): LandPoint | null {
    const { scene, viewer } = this.host;
    const ray = viewer.camera.getPickRay(position);
    const hit = ray ? scene.globe.pick(ray, scene) : undefined;
    const world = hit ?? viewer.camera.pickEllipsoid(position, scene.globe.ellipsoid);
    if (!world) return null;
    const at = Cartographic.fromCartesian(world);
    return [CesiumMath.toDegrees(at.longitude), CesiumMath.toDegrees(at.latitude)];
  }

  setMode(mode: LandMode): void {
    if (this.mode === mode) return;
    this.finishDrag();
    this.mode = mode;
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
      if (!this.dragging || !this.footprint) return;
      const point = this.ground(event.endPosition);
      if (!point) return;
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
    this.entities.forEach((entity) => this.host.viewer.entities.remove(entity));
    this.host.scene.canvas.style.cursor = "";
    window.removeEventListener("pointerup", this.release);
    window.removeEventListener("blur", this.release);
  }
}
