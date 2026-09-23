import {
  CallbackProperty,
  type Cartesian2,
  Cartesian3,
  Cartographic,
  Color,
  ColorMaterialProperty,
  type Entity,
  HeightReference,
  Math as CesiumMath,
  PolygonHierarchy,
  PolylineDashMaterialProperty,
  ScreenSpaceEventHandler,
  ScreenSpaceEventType,
  type Scene,
  type Viewer,
} from "cesium";

import type { Footprint } from "@twin/contracts";
import { flattenRing, outerRings } from "@twin/geo";

import type { Emitter } from "@/lib/emitter";

import type { SceneEvents } from "./types";

/** Entity id prefixes the editor owns; the selection manager ignores them. */
export const AREA_HANDLE_PREFIX = "area-edit:";
export const AREA_CANDIDATE_PREFIX = "area-candidate:";

const HANDLE = Color.fromCssColorString("#ffffff");
const HANDLE_RING = Color.fromCssColorString("#7fd8c0");
const MID = HANDLE_RING.withAlpha(0.55);
const MOVE = Color.fromCssColorString("#f0b254");
const EDIT_LINE = HANDLE_RING;
const EDIT_FILL = HANDLE_RING.withAlpha(0.1);
const CANDIDATE_LINE = Color.fromCssColorString("#9db8f0");
const CANDIDATE_FILL = CANDIDATE_LINE.withAlpha(0.08);
const CANDIDATE_HOVER_FILL = CANDIDATE_LINE.withAlpha(0.28);
const MIN_VERTICES = 3;

export interface AreaCandidateOutline {
  id: string;
  name: string;
  footprint: Footprint;
}

/** Longitude, latitude and the ground height the handle sits at. */
type Vertex = [number, number, number];

type Handle = { kind: "vertex"; index: number } | { kind: "mid"; index: number } | { kind: "move" };

interface Drag {
  handle: Handle;
  /** Ring vertex under the pointer at drag start, for the move handle's delta. */
  origin: { longitude: number; latitude: number };
  ring: Vertex[];
}

/**
 * Hands the operator the outline of an area on the map: white vertex handles drag a corner,
 * translucent midpoint handles pull a new corner out of an edge, the amber centre handle
 * moves the whole area, and a right-click on a corner removes it. Camera inputs pause while
 * a handle is held so a drag never pans the world. The edited ring is emitted on release; the
 * store owns the zone and the map follows it.
 *
 * The same manager draws candidate outlines the finder proposes (dashed, blue); a click on
 * one emits its id so the console can adopt it.
 */
export class AreaEditor {
  private readonly scene: Scene;
  private handler: ScreenSpaceEventHandler | null = null;
  private zoneId: string | null = null;
  private ring: Vertex[] = [];
  private readonly entities: Entity[] = [];
  private readonly handles: Entity[] = [];
  private drag: Drag | null = null;
  private hovered: string | null = null;
  private readonly candidates = new Map<string, Entity[]>();
  private candidateHover: string | null = null;
  private pick: (() => void) | null = null;
  private readonly onKey = (event: KeyboardEvent) => {
    if (event.key === "Escape" && this.zoneId) this.events.emit("area-edit-end", this.zoneId);
  };

  constructor(
    private readonly viewer: Viewer,
    private readonly events: Emitter<SceneEvents>,
  ) {
    this.scene = viewer.scene;
  }

  get editing(): string | null {
    return this.zoneId;
  }

  /**
   * "Click the ground you mean": the next click on the map resolves with the ground under it
   * and its place in the view (normalized); Esc or a second call resolves null.
   */
  pickGround(): Promise<{ longitude: number; latitude: number; x: number; y: number } | null> {
    this.cancelPick();
    return new Promise((resolve) => {
      const canvas = this.viewer.canvas;
      const handler = new ScreenSpaceEventHandler(canvas);
      const finish = (result: Awaited<ReturnType<AreaEditor["pickGround"]>>) => {
        handler.destroy();
        window.removeEventListener("keydown", onKey);
        canvas.style.cursor = "";
        this.pick = null;
        this.events.emit("ground-pick-mode", false);
        resolve(result);
      };
      const onKey = (event: KeyboardEvent) => {
        if (event.key === "Escape") finish(null);
      };
      handler.setInputAction((e: ScreenSpaceEventHandler.PositionedEvent) => {
        const ground = this.ground(e.position);
        if (!ground) return;
        finish({
          longitude: ground.longitude,
          latitude: ground.latitude,
          x: e.position.x / canvas.clientWidth,
          y: e.position.y / canvas.clientHeight,
        });
      }, ScreenSpaceEventType.LEFT_CLICK);
      window.addEventListener("keydown", onKey);
      canvas.style.cursor = "crosshair";
      this.pick = () => finish(null);
      this.events.emit("ground-pick-mode", true);
    });
  }

  /** Leaves pick mode, resolving the pending pick with null. */
  cancelPick(): void {
    this.pick?.();
  }

  /** Starts editing `zoneId` with `footprint`'s outer ring, or stops when null. */
  edit(zoneId: string | null, footprint?: Footprint): void {
    this.stopEditing();
    const ring = footprint ? outerRings(footprint)[0] : undefined;
    if (!zoneId || !ring || ring.length < MIN_VERTICES + 1) {
      this.ensureHandler();
      return;
    }
    this.zoneId = zoneId;
    this.ring = this.verticesOf(ring);
    this.entities.push(
      this.viewer.entities.add({
        id: `${AREA_HANDLE_PREFIX}fill`,
        polygon: {
          hierarchy: new CallbackProperty(
            () => new PolygonHierarchy(this.positions(this.ring)),
            false,
          ),
          material: EDIT_FILL,
          heightReference: HeightReference.CLAMP_TO_GROUND,
        },
      }),
      this.viewer.entities.add({
        id: `${AREA_HANDLE_PREFIX}outline`,
        polyline: {
          positions: new CallbackProperty(() => {
            const p = this.positions(this.ring);
            const first = p[0];
            return first ? [...p, first] : p;
          }, false),
          width: 2.6,
          material: EDIT_LINE,
          clampToGround: true,
        },
      }),
    );
    this.rebuildHandles();
    this.ensureHandler();
    window.addEventListener("keydown", this.onKey);
    this.scene.requestRender();
  }

  /** Replaces the ring being edited (the store changed it, e.g. a size slider). */
  update(footprint: Footprint): void {
    const ring = outerRings(footprint)[0];
    if (!this.zoneId || !ring || this.drag) return;
    this.ring = this.verticesOf(ring);
    this.rebuildHandles();
    this.scene.requestRender();
  }

  /** Draws (or clears) the finder's candidate outlines; each is clickable. */
  showCandidates(candidates: AreaCandidateOutline[]): void {
    for (const entities of this.candidates.values())
      for (const entity of entities) this.viewer.entities.remove(entity);
    this.candidates.clear();
    this.candidateHover = null;
    for (const candidate of candidates) {
      const ring = outerRings(candidate.footprint)[0];
      if (!ring || ring.length < 4) continue;
      const positions = Cartesian3.fromDegreesArray(flattenRing(ring));
      const first = positions[0];
      if (!first) continue;
      const id = `${AREA_CANDIDATE_PREFIX}${candidate.id}`;
      this.candidates.set(candidate.id, [
        this.viewer.entities.add({
          id,
          polygon: {
            hierarchy: positions,
            material: CANDIDATE_FILL,
            heightReference: HeightReference.CLAMP_TO_GROUND,
          },
        }),
        this.viewer.entities.add({
          id: `${id}:line`,
          polyline: {
            positions: [...positions, first],
            width: 2.2,
            material: new PolylineDashMaterialProperty({ color: CANDIDATE_LINE, dashLength: 12 }),
            clampToGround: true,
          },
        }),
      ]);
    }
    this.ensureHandler();
    this.scene.requestRender();
  }

  /** Brightens one candidate (the console's hover) or none. */
  highlightCandidate(id: string | null): void {
    if (this.candidateHover === id) return;
    for (const [candidateId, entities] of this.candidates) {
      const polygon = entities[0]?.polygon;
      if (!polygon) continue;
      polygon.material = new ColorMaterialProperty(
        candidateId === id ? CANDIDATE_HOVER_FILL : CANDIDATE_FILL,
      );
    }
    this.candidateHover = id;
    this.scene.requestRender();
  }

  private ensureHandler(): void {
    if (this.handler || (!this.zoneId && this.candidates.size === 0)) return;
    this.handler = new ScreenSpaceEventHandler(this.viewer.canvas);
    this.handler.setInputAction(
      (e: ScreenSpaceEventHandler.PositionedEvent) => this.onDown(e.position),
      ScreenSpaceEventType.LEFT_DOWN,
    );
    this.handler.setInputAction(
      (e: ScreenSpaceEventHandler.MotionEvent) => this.onMove(e.endPosition),
      ScreenSpaceEventType.MOUSE_MOVE,
    );
    this.handler.setInputAction(() => this.onUp(), ScreenSpaceEventType.LEFT_UP);
    this.handler.setInputAction(
      (e: ScreenSpaceEventHandler.PositionedEvent) => this.onRightClick(e.position),
      ScreenSpaceEventType.RIGHT_CLICK,
    );
    this.handler.setInputAction(
      (e: ScreenSpaceEventHandler.PositionedEvent) => this.onClick(e.position),
      ScreenSpaceEventType.LEFT_CLICK,
    );
  }

  private dropHandler(): void {
    if (this.zoneId || this.candidates.size > 0) return;
    this.handler?.destroy();
    this.handler = null;
    this.viewer.canvas.style.cursor = "";
  }

  private stopEditing(): void {
    if (this.drag) this.endDrag(false);
    for (const entity of [...this.entities, ...this.handles]) this.viewer.entities.remove(entity);
    this.entities.length = 0;
    this.handles.length = 0;
    this.zoneId = null;
    this.ring = [];
    this.hovered = null;
    window.removeEventListener("keydown", this.onKey);
    this.dropHandler();
    this.scene.requestRender();
  }

  /**
   * Rings close on their first vertex; the editor works on the open list. Handles are placed
   * at the sampled ground height rather than clamped, so they render and pick on the first
   * frame whatever the tiles are doing; the fill and outline stay clamped to the surface.
   */
  private verticesOf(ring: (number | undefined)[][]): Vertex[] {
    return ring
      .slice(0, -1)
      .filter((c): c is [number, number] => c.length >= 2 && c[0] !== undefined)
      .map(([lon, lat]) => [lon, lat, this.heightAt(lon, lat)]);
  }

  private heightAt(longitude: number, latitude: number): number {
    const carto = Cartographic.fromDegrees(longitude, latitude);
    let height: number | undefined;
    if (this.scene.sampleHeightSupported) {
      try {
        height = this.scene.sampleHeight(carto, [...this.entities, ...this.handles]);
      } catch {
        height = undefined;
      }
    }
    if (height === undefined || !Number.isFinite(height))
      height = this.scene.globe.getHeight(carto);
    return height !== undefined && Number.isFinite(height) ? height : 0;
  }

  private positions(ring: Vertex[]): Cartesian3[] {
    return ring.map(([lon, lat, h]) => Cartesian3.fromDegrees(lon, lat, h));
  }

  private rebuildHandles(): void {
    for (const entity of this.handles) this.viewer.entities.remove(entity);
    this.handles.length = 0;
    const n = this.ring.length;
    for (let i = 0; i < n; i++) {
      this.handles.push(
        this.viewer.entities.add({
          id: `${AREA_HANDLE_PREFIX}vertex:${i}`,
          position: new CallbackProperty(() => {
            const v = this.ring[i];
            return v ? Cartesian3.fromDegrees(v[0], v[1], v[2]) : undefined;
          }, false) as never,
          point: {
            pixelSize: 11,
            color: HANDLE,
            outlineColor: HANDLE_RING,
            outlineWidth: 2.5,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
        }),
        this.viewer.entities.add({
          id: `${AREA_HANDLE_PREFIX}mid:${i}`,
          position: new CallbackProperty(() => {
            const a = this.ring[i];
            const b = this.ring[(i + 1) % this.ring.length];
            return a && b
              ? Cartesian3.fromDegrees((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)
              : undefined;
          }, false) as never,
          point: {
            pixelSize: 8,
            color: MID,
            outlineColor: HANDLE_RING.withAlpha(0.8),
            outlineWidth: 1.5,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
        }),
      );
    }
    this.handles.push(
      this.viewer.entities.add({
        id: `${AREA_HANDLE_PREFIX}move`,
        position: new CallbackProperty(() => {
          const c = this.centroid();
          return c ? Cartesian3.fromDegrees(c[0], c[1], c[2]) : undefined;
        }, false) as never,
        point: {
          pixelSize: 14,
          color: MOVE,
          outlineColor: Color.fromCssColorString("#0a0e09").withAlpha(0.7),
          outlineWidth: 2,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      }),
    );
  }

  private centroid(): Vertex | null {
    if (this.ring.length === 0) return null;
    let lon = 0;
    let lat = 0;
    let height = 0;
    for (const [x, y, h] of this.ring) {
      lon += x;
      lat += y;
      height += h;
    }
    const n = this.ring.length;
    return [lon / n, lat / n, height / n];
  }

  private handleAt(window: Cartesian2): Handle | null {
    const picked: unknown = this.scene.pick(window);
    const id = pickedId(picked);
    if (!id?.startsWith(AREA_HANDLE_PREFIX)) return null;
    const [, kind, index] = id.split(":");
    if (kind === "vertex") return { kind: "vertex", index: Number(index) };
    if (kind === "mid") return { kind: "mid", index: Number(index) };
    if (kind === "move") return { kind: "move" };
    return null;
  }

  private candidateAt(window: Cartesian2): string | null {
    const id = pickedId(this.scene.pick(window));
    if (!id?.startsWith(AREA_CANDIDATE_PREFIX)) return null;
    return id.slice(AREA_CANDIDATE_PREFIX.length).replace(/:line$/, "");
  }

  /** Ground under the pointer as degrees and height, from the depth buffer or the globe. */
  private ground(
    window: Cartesian2,
  ): { longitude: number; latitude: number; height: number } | null {
    let position: Cartesian3 | undefined;
    if (this.scene.pickPositionSupported) position = this.scene.pickPosition(window);
    if (!position) {
      const ray = this.viewer.camera.getPickRay(window);
      position = ray ? this.scene.globe.pick(ray, this.scene) : undefined;
    }
    // No terrain loaded yet: the bare ellipsoid still says where the click is.
    position ??= this.viewer.camera.pickEllipsoid(window, this.scene.globe.ellipsoid);
    if (!position) return null;
    const carto = Cartographic.fromCartesian(position);
    return {
      longitude: CesiumMath.toDegrees(carto.longitude),
      latitude: CesiumMath.toDegrees(carto.latitude),
      height: carto.height,
    };
  }

  private onDown(window: Cartesian2): void {
    if (!this.zoneId) return;
    const handle = this.handleAt(window);
    if (!handle) return;
    const origin = this.ground(window);
    if (!origin) return;
    if (handle.kind === "mid") {
      // Pull a new corner out of the edge and keep dragging it.
      const a = this.ring[handle.index];
      const b = this.ring[(handle.index + 1) % this.ring.length];
      if (!a || !b) return;
      this.ring.splice(handle.index + 1, 0, [
        (a[0] + b[0]) / 2,
        (a[1] + b[1]) / 2,
        (a[2] + b[2]) / 2,
      ]);
      this.rebuildHandles();
      this.drag = { handle: { kind: "vertex", index: handle.index + 1 }, origin, ring: [] };
    } else {
      this.drag = { handle, origin, ring: this.ring.map(([x, y, h]) => [x, y, h]) };
    }
    this.scene.screenSpaceCameraController.enableInputs = false;
    this.viewer.canvas.style.cursor = "grabbing";
    this.events.emit("area-edit-drag", true);
  }

  private onMove(window: Cartesian2): void {
    if (this.drag) {
      const ground = this.ground(window);
      if (!ground) return;
      const { handle } = this.drag;
      if (handle.kind === "vertex") {
        this.ring[handle.index] = [ground.longitude, ground.latitude, ground.height];
      } else if (handle.kind === "move") {
        const dLon = ground.longitude - this.drag.origin.longitude;
        const dLat = ground.latitude - this.drag.origin.latitude;
        this.ring = this.drag.ring.map(([x, y, h]) => [x + dLon, y + dLat, h]);
      }
      this.scene.requestRender();
      return;
    }
    if (this.zoneId) {
      const handle = this.handleAt(window);
      const key = handle ? `${handle.kind}:${"index" in handle ? handle.index : ""}` : null;
      if (key !== this.hovered) {
        this.hovered = key;
        this.viewer.canvas.style.cursor = handle ? (handle.kind === "move" ? "move" : "grab") : "";
      }
      if (handle) return;
    }
    if (this.candidates.size > 0) {
      const id = this.candidateAt(window);
      this.highlightCandidate(id);
      this.events.emit("area-candidate-hover", id);
      if (!this.zoneId || !this.hovered) this.viewer.canvas.style.cursor = id ? "pointer" : "";
    }
  }

  private onUp(): void {
    if (this.drag) this.endDrag(true);
  }

  private endDrag(commit: boolean): void {
    if (!this.drag) return;
    const wasMove = this.drag.handle.kind === "move";
    this.drag = null;
    this.scene.screenSpaceCameraController.enableInputs = true;
    this.viewer.canvas.style.cursor = "";
    this.events.emit("area-edit-drag", false);
    if (wasMove) this.rebuildHandles();
    if (commit) this.emitRing();
  }

  private onRightClick(window: Cartesian2): void {
    if (!this.zoneId || this.drag) return;
    const handle = this.handleAt(window);
    if (handle?.kind !== "vertex" || this.ring.length <= MIN_VERTICES) return;
    this.ring.splice(handle.index, 1);
    this.rebuildHandles();
    this.emitRing();
  }

  private onClick(window: Cartesian2): void {
    if (this.drag || this.candidates.size === 0) return;
    const id = this.candidateAt(window);
    if (id) this.events.emit("area-candidate-pick", id);
  }

  private emitRing(): void {
    const zoneId = this.zoneId;
    const first = this.ring[0];
    if (!zoneId || !first) return;
    const ring: [number, number][] = this.ring.map(([x, y]) => [x, y]);
    ring.push([first[0], first[1]]);
    this.events.emit("area-edit", {
      zoneId,
      footprint: { type: "Polygon", coordinates: [ring] },
    });
    this.scene.requestRender();
  }

  destroy(): void {
    this.cancelPick();
    this.stopEditing();
    this.showCandidates([]);
    this.handler?.destroy();
    this.handler = null;
  }
}

function pickedId(picked: unknown): string | null {
  if (!picked || typeof picked !== "object" || !("id" in picked)) return null;
  const { id } = picked;
  if (typeof id === "string") return id;
  if (id && typeof id === "object" && "id" in id) {
    const { id: inner } = id;
    return typeof inner === "string" ? inner : null;
  }
  return null;
}
