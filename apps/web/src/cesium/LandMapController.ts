import {
  Cartesian2,
  Cartesian3,
  Cartographic,
  Color,
  ColorMaterialProperty,
  ConstantPositionProperty,
  ConstantProperty,
  Ellipsoid,
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
import { repaintUntilDrawn } from "./repaintUntilDrawn";

const FILL = Color.fromCssColorString("#a8d7b4").withAlpha(0.13);
const LINE = Color.fromCssColorString("#b9edc6");
const PREFIX = "land-boundary:";
/** Handles beyond this many keep the globe-clamped placement rather than a scene sample. */
const MAX_SAMPLED_HANDLES = 400;

/** Below the Dead Sea shore or above Everest, a depth reading is not the ground. */
const LOWEST_GROUND_M = -500;
const HIGHEST_GROUND_M = 9_000;

const key = (point: number[]) => `${point[0]},${point[1]}`;

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
  /** Surface height of each point picked or sampled here, keyed by its degrees. */
  private heights = new Map<string, number>();
  private points: LandPoint[] = [];
  private sampling = 0;
  /** Height of the last surface the depth buffer showed under the pointer. */
  private surfaceHeight: number | undefined;
  private readonly release = () => this.finishDrag();
  private stopRepaint: () => void = () => undefined;

  constructor(
    private readonly host: CesiumSceneManager,
    private readonly point: (point: LandPoint) => void,
    private readonly change: (boundary: Footprint) => void,
    private readonly snapTarget?: (label: string | null) => void,
  ) {
    window.addEventListener("pointerup", this.release);
    window.addEventListener("blur", this.release);
  }

  /**
   * The surface under a canvas pixel. The depth buffer comes first: photorealistic tiles hide
   * the globe, so a globe pick lands on the terrain beneath them and drifts from the clicked
   * spot as the camera tilts. Far from the ground the analytic globe pick is the precise one.
   */
  private surface(position: Cartesian2): { point: LandPoint; height: number } | null {
    const { scene, viewer } = this.host;
    const ray = viewer.camera.getPickRay(position);
    const far = this.host.camera.pose().altitude > 20_000;
    const globe = ray ? scene.globe.pick(ray, scene) : undefined;
    let world: Cartesian3 | undefined = far ? globe : undefined;
    if (!world && scene.pickPositionSupported) {
      const hit = scene.pickPosition(position);
      const near = (viewer.camera.frustum as { near?: number }).near ?? 1;
      const height = hit ? Cartographic.fromCartesian(hit).height : NaN;
      if (hit && Cartesian3.distance(hit, viewer.camera.positionWC) > 2 * near) {
        // Depth left over from a frame drawn before the camera moved puts the point deep
        // underground; terrain where the globe is drawn, or Earth's range, rules that out.
        const floor = globe ? Cartographic.fromCartesian(globe).height - 100 : LOWEST_GROUND_M;
        if (height > floor && height < HIGHEST_GROUND_M) {
          world = hit;
          this.surfaceHeight = height;
        }
      } else if (hit && this.surfaceHeight !== undefined) {
        // Dots drawn over everything write their depth on the near plane, so a click on one
        // (or a handle being dragged) reads back the camera; the surface just seen stands in.
        world = viewer.camera.pickEllipsoid(position, this.raised(this.surfaceHeight));
      }
    }
    world ??= globe;
    // No terrain drawn there yet: the bare ellipsoid still says where the click is.
    world ??= viewer.camera.pickEllipsoid(position, scene.globe.ellipsoid);
    if (!world) return null;
    const at = Cartographic.fromCartesian(world);
    return {
      point: [CesiumMath.toDegrees(at.longitude), CesiumMath.toDegrees(at.latitude)],
      height: at.height,
    };
  }

  private ground(position: Cartesian2): { point: LandPoint; height: number } | null {
    const hit = this.surface(position);
    if (!hit) {
      this.showSnap(null);
      return null;
    }
    const { point, height } = hit;
    if (this.mode === "pick" || this.mode === "candidates") return hit;
    const { scene, viewer } = this.host;
    // Snap targets lie within a few pixels of the pointer, so the surface height just picked
    // stands in for theirs; the globe's own height is wrong wherever 3D tiles cover it.
    const distance = (target: LandPoint) => {
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
      // The search window: neighbouring pixels cast onto an ellipsoid at the picked height,
      // which matches the surface locally without another depth-buffer read per sample.
      const local = this.raised(height);
      const bounds = { minX: point[0], minY: point[1], maxX: point[0], maxY: point[1] };
      for (const x of [-24, 0, 24])
        for (const y of [-24, 0, 24]) {
          if (!x && !y) continue;
          const world = viewer.camera.pickEllipsoid(
            new Cartesian2(position.x + x, position.y + y),
            local,
          );
          if (!world) continue;
          const at = Cartographic.fromCartesian(world);
          const lon = CesiumMath.toDegrees(at.longitude);
          const lat = CesiumMath.toDegrees(at.latitude);
          if (Math.abs(lon - point[0]) > 180) continue;
          bounds.minX = Math.min(bounds.minX, lon);
          bounds.maxX = Math.max(bounds.maxX, lon);
          bounds.minY = Math.min(bounds.minY, lat);
          bounds.maxY = Math.max(bounds.maxY, lat);
        }
      const mapped = this.mappedSnap.nearest(point, bounds, distance);
      if (mapped && (!best || mapped.distance < best.distance)) best = mapped;
    }
    if (best && best.distance <= 12) {
      this.showSnap(best.point, best.label, height);
      return { point: best.point, height };
    }
    this.showSnap(null);
    return hit;
  }

  /** The globe's ellipsoid lifted to a surface height: the ground there, to a close approximation. */
  private raised(height: number): Ellipsoid {
    const { radii } = this.host.scene.globe.ellipsoid;
    return new Ellipsoid(radii.x + height, radii.y + height, radii.z + height);
  }

  /**
   * Where a dot for this point is drawn. CLAMP_TO_GROUND follows only the globe, which sits
   * beneath photorealistic tiles; a dot there floats off its line as the camera tilts, so a
   * point drawn here keeps the height of the surface it was picked from.
   */
  private dot(
    point: number[],
    height = this.heights.get(key(point)),
  ): { position: Cartesian3; heightReference: HeightReference } {
    const [lon = 0, lat = 0] = point;
    return height === undefined
      ? {
          position: Cartesian3.fromDegrees(lon, lat),
          heightReference: HeightReference.CLAMP_TO_GROUND,
        }
      : {
          position: Cartesian3.fromDegrees(lon, lat, height),
          heightReference: HeightReference.NONE,
        };
  }

  /** Surface heights for the handles of a boundary opened for editing, read from the scene. */
  private sampleHandles(footprint: Footprint): void {
    const { scene } = this.host;
    if (!scene.sampleHeightSupported) return;
    const points = polygonsOf(footprint)
      .flatMap((polygon) => polygon.flatMap((ring) => ring.slice(0, -1)))
      .filter((point) => !this.heights.has(key(point)))
      .slice(0, MAX_SAMPLED_HANDLES);
    if (!points.length) return;
    const ticket = ++this.sampling;
    const at = points.map(([lon = 0, lat = 0]) => Cartographic.fromDegrees(lon, lat));
    scene
      .sampleHeightMostDetailed(at)
      .then((sampled) => {
        if (ticket !== this.sampling || this.host.isDestroyed || this.footprint !== footprint)
          return;
        sampled.forEach((carto, index) => {
          const point = points[index];
          if (point && carto && Number.isFinite(carto.height))
            this.heights.set(key(point), carto.height);
        });
        this.show(footprint, this.points, true);
      })
      .catch(() => undefined);
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

  private showSnap(point: LandPoint | null, label: string | null = null, height?: number): void {
    this.snapTarget?.(point ? label : null);
    if (!point && !this.snapMarker) return;
    const at = point ? this.dot(point, height) : null;
    if (at && this.snapMarker?.point) {
      this.snapMarker.position = new ConstantPositionProperty(at.position);
      this.snapMarker.point.heightReference = new ConstantProperty(at.heightReference);
      this.host.scene.requestRender();
      return;
    }
    if (this.snapMarker) this.host.viewer.entities.remove(this.snapMarker);
    this.snapMarker = at
      ? this.host.viewer.entities.add({
          id: `${PREFIX}snap`,
          position: at.position,
          point: {
            pixelSize: 17,
            color: Color.TRANSPARENT,
            outlineColor: Color.GOLD,
            outlineWidth: 3,
            heightReference: at.heightReference,
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
      const hit = this.ground(event.position);
      if (!hit) return;
      this.heights.set(key(hit.point), hit.height);
      this.point(hit.point);
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
      const hit = this.ground(event.endPosition);
      if (!hit || !this.dragging || !this.footprint) return;
      const copy = structuredClone(this.footprint);
      const { polygon, ring, vertex } = this.dragging;
      const coordinates = polygonsOf(copy)[polygon]?.[ring];
      if (!coordinates) return;
      const { point } = hit;
      this.heights.set(key(point), hit.height);
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
    if (footprint && editing && !this.dragging && footprint !== this.footprint)
      this.sampleHandles(footprint);
    this.footprint = footprint;
    this.points = points;
    // Only points still on the map keep a height; a drag leaves one behind per move.
    const kept = new Map<string, number>();
    const keep = (point: number[]) => {
      const height = this.heights.get(key(point));
      if (height !== undefined) kept.set(key(point), height);
    };
    points.forEach(keep);
    if (footprint)
      polygonsOf(footprint).forEach((polygon) => polygon.forEach((r) => r.forEach(keep)));
    this.heights = kept;
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
            ring.slice(0, -1).forEach((point, vi) => {
              const at = this.dot(point);
              add({
                id: `${PREFIX}handle:${pi}:${ri}:${vi}`,
                position: at.position,
                point: {
                  pixelSize: 11,
                  color: Color.WHITE,
                  outlineColor: LINE,
                  outlineWidth: 2,
                  heightReference: at.heightReference,
                  disableDepthTestDistance: Number.POSITIVE_INFINITY,
                },
              });
            });
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
    points.forEach((point, index) => {
      const at = this.dot(point);
      add({
        id: `${PREFIX}drawing:${index}`,
        position: at.position,
        point: {
          pixelSize: 8,
          color: LINE,
          heightReference: at.heightReference,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
    });
    // Clamped outlines are built over several frames that nothing else asks for.
    this.stopRepaint();
    this.stopRepaint = repaintUntilDrawn(this.host);
  }

  frame(footprint: Footprint): void {
    const b = boundsOf(footprint);
    this.host.camera.flyToRectangle(b.west, b.south, b.east, b.north);
  }

  destroy(): void {
    this.stopRepaint();
    this.finishDrag();
    this.handler?.destroy();
    this.showSnap(null);
    this.entities.forEach((entity) => this.host.viewer.entities.remove(entity));
    this.host.scene.canvas.style.cursor = "";
    window.removeEventListener("pointerup", this.release);
    window.removeEventListener("blur", this.release);
  }
}
