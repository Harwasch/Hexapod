import {
  BoundingSphere,
  Cartesian2,
  Cartesian3,
  Cartographic,
  Color,
  ColorMaterialProperty,
  ConstantProperty,
  type Entity,
  HeightReference,
  PolylineDashMaterialProperty,
  SceneTransforms,
  type Scene,
  type Viewer,
} from "cesium";

import { boundingRadiusM, centerOf, flattenRing, outerRings } from "@twin/geo";

import type { Emitter } from "@/lib/emitter";
import type { Machine, Project, Zone } from "@/missions/types";

import type { CameraController } from "./CameraController";
import type { SceneEvents } from "./types";

export const ZONE_ENTITY_PREFIX = "mission:zone:";

const TONE: Record<Zone["tone"], Color> = {
  teal: Color.fromCssColorString("#7fd8c0"),
  amber: Color.fromCssColorString("#f0b254"),
  blue: Color.fromCssColorString("#9db8f0"),
};

/** Screen-space anchor for a DOM overlay (machine marker or zone chip). */
export interface OverlayAnchor {
  id: string;
  x: number;
  y: number;
  visible: boolean;
}

export type FrameListener = (anchors: readonly OverlayAnchor[]) => void;

/**
 * Draws mission geometry (zones, tracks) as entities in the world and projects
 * machine/zone anchors to the screen every frame for DOM overlays that keep the
 * design's exact marker and chip styling.
 */
export class MissionManager {
  private readonly scene: Scene;
  private project: Project | null = null;
  private readonly zoneEntities: Entity[] = [];
  private readonly trackEntities: Entity[] = [];
  private zonesVisible = true;
  private tracksVisible = true;
  private readonly listeners = new Set<FrameListener>();
  private readonly removeFrame: () => void;
  private readonly heightCache = new Map<string, { height: number; at: number }>();
  private readonly scratchWindow = new Cartesian2();
  private readonly scratchNormal = new Cartesian3();
  private readonly scratchToCamera = new Cartesian3();

  constructor(
    private readonly viewer: Viewer,
    private readonly events: Emitter<SceneEvents>,
    private readonly camera: CameraController,
  ) {
    this.scene = viewer.scene;
    this.removeFrame = this.scene.postRender.addEventListener(
      () => this.project && this.listeners.size > 0 && this.projectAnchors(),
    );
  }

  get current(): Project | null {
    return this.project;
  }

  setProject(project: Project | null): void {
    this.clearEntities();
    this.project = project;
    this.heightCache.clear();
    if (!project) return;
    for (const zone of project.zones) {
      const ring = outerRings(zone.footprint)[0];
      if (!ring) continue;
      const positions = Cartesian3.fromDegreesArray(flattenRing(ring));
      const first = positions[0];
      if (!first) continue;
      const tone = TONE[zone.tone];
      this.zoneEntities.push(
        this.viewer.entities.add({
          id: `${ZONE_ENTITY_PREFIX}${zone.id}`,
          show: this.zonesVisible,
          polygon: {
            hierarchy: positions,
            material: tone.withAlpha(zone.treated ? 0.22 : 0.14),
            heightReference: HeightReference.CLAMP_TO_GROUND,
            classificationType: undefined,
          },
          polyline: {
            positions: [...positions, first],
            width: zone.treated ? 2.4 : 1.8,
            material: zone.treated
              ? tone
              : new PolylineDashMaterialProperty({ color: tone, dashLength: 14 }),
            clampToGround: true,
          },
        }),
      );
    }
    for (const machine of project.machines) {
      if (!machine.track || machine.track.length < 2) continue;
      const flat = machine.track.flatMap((p) => [p.longitude, p.latitude]);
      const positions = Cartesian3.fromDegreesArray(flat);
      this.trackEntities.push(
        this.viewer.entities.add({
          id: `mission:track:${machine.id}`,
          show: this.tracksVisible,
          polyline: {
            positions,
            width: 5,
            material: Color.fromCssColorString("#0a0e09").withAlpha(0.5),
            clampToGround: true,
          },
        }),
        this.viewer.entities.add({
          id: `mission:track-dash:${machine.id}`,
          show: this.tracksVisible,
          polyline: {
            positions,
            width: 2.2,
            material: new PolylineDashMaterialProperty({ color: TONE.teal, dashLength: 16 }),
            clampToGround: true,
          },
        }),
      );
    }
    this.scene.requestRender();
  }

  setLayer(key: "zones" | "tracks", visible: boolean): void {
    if (key === "zones") {
      this.zonesVisible = visible;
      for (const entity of this.zoneEntities) entity.show = visible;
    } else {
      this.tracksVisible = visible;
      for (const entity of this.trackEntities) entity.show = visible;
    }
    this.scene.requestRender();
  }

  /** Highlights the selected zone outline. */
  setSelectedZone(zoneId: string | null): void {
    for (const entity of this.zoneEntities) {
      const zone = this.project?.zones.find((z) => `${ZONE_ENTITY_PREFIX}${z.id}` === entity.id);
      if (!zone || !entity.polyline) continue;
      const selected = zone.id === zoneId;
      entity.polyline.width = new ConstantProperty(selected ? 3 : zone.treated ? 2.4 : 1.8);
      entity.polyline.material =
        selected || zone.treated
          ? new ColorMaterialProperty(TONE[zone.tone])
          : new PolylineDashMaterialProperty({ color: TONE[zone.tone], dashLength: 14 });
    }
    this.scene.requestRender();
  }

  onFrame(listener: FrameListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  flyToZone(zoneId: string): void {
    const zone = this.project?.zones.find((z) => z.id === zoneId);
    if (!zone) return;
    const center = centerOf(zone.footprint);
    const radius = Math.max(boundingRadiusM(zone.footprint), 80);
    const height = this.surfaceHeight(center.longitude, center.latitude, `zone-fly-${zone.id}`);
    this.camera.flyToBoundingSphere(
      new BoundingSphere(Cartesian3.fromDegrees(center.longitude, center.latitude, height), radius),
      { heading: 20, pitch: -40, rangeMultiplier: 2.6, durationS: 1.8 },
    );
  }

  flyToMachine(machineId: string): void {
    const machine = this.project?.machines.find((m) => m.id === machineId);
    if (!machine) return;
    const height = this.surfaceHeight(
      machine.position.longitude,
      machine.position.latitude,
      `machine-fly-${machine.id}`,
    );
    this.camera.flyToBoundingSphere(
      new BoundingSphere(
        Cartesian3.fromDegrees(machine.position.longitude, machine.position.latitude, height),
        40,
      ),
      { heading: machine.headingDeg + 180, pitch: -32, rangeMultiplier: 3.5, durationS: 1.6 },
    );
  }

  flyToProject(): void {
    const project = this.project;
    if (!project || project.zones.length === 0) return;
    const centers = project.zones.map((z) => centerOf(z.footprint));
    const lon = centers.reduce((a, c) => a + c.longitude, 0) / centers.length;
    const lat = centers.reduce((a, c) => a + c.latitude, 0) / centers.length;
    const height = this.surfaceHeight(lon, lat, "project-fly");
    this.camera.flyToBoundingSphere(
      new BoundingSphere(Cartesian3.fromDegrees(lon, lat, height), 1100),
      { heading: 15, pitch: -45, rangeMultiplier: 2.2, durationS: 2 },
    );
  }

  private surfaceHeight(longitude: number, latitude: number, key: string): number {
    const now = performance.now();
    const cached = this.heightCache.get(key);
    if (cached && now - cached.at < 1500) return cached.height;
    const carto = Cartographic.fromDegrees(longitude, latitude);
    let height: number | undefined;
    if (this.scene.sampleHeightSupported) {
      try {
        height = this.scene.sampleHeight(carto, this.viewer.entities.values);
      } catch {
        height = undefined;
      }
    }
    height ??= this.scene.globe.getHeight(carto) ?? 0;
    this.heightCache.set(key, { height, at: now });
    return height;
  }

  private projectAnchors(): void {
    const project = this.project;
    if (!project) return;
    const cameraPosition = this.viewer.camera.positionWC;
    const anchors: OverlayAnchor[] = [];
    const push = (id: string, lon: number, lat: number, lift: number) => {
      const height = this.surfaceHeight(lon, lat, id) + lift;
      const world = Cartesian3.fromDegrees(lon, lat, height);
      const visible = this.isAboveHorizon(world, cameraPosition);
      const window = visible
        ? SceneTransforms.worldToWindowCoordinates(this.scene, world, this.scratchWindow)
        : undefined;
      anchors.push({ id, x: window?.x ?? -9999, y: window?.y ?? -9999, visible: Boolean(window) });
    };
    for (const machine of project.machines)
      push(`machine:${machine.id}`, machine.position.longitude, machine.position.latitude, 1.2);
    if (this.zonesVisible)
      for (const zone of project.zones)
        push(`zone:${zone.id}`, zone.anchor.longitude, zone.anchor.latitude, 0.5);
    for (const listener of this.listeners) listener(anchors);
  }

  /** True when the point faces the camera (not hidden behind the planet's limb). */
  private isAboveHorizon(point: Cartesian3, cameraPosition: Cartesian3): boolean {
    const normal = Cartesian3.normalize(point, this.scratchNormal);
    const toCamera = Cartesian3.normalize(
      Cartesian3.subtract(cameraPosition, point, this.scratchToCamera),
      this.scratchToCamera,
    );
    return Cartesian3.dot(normal, toCamera) > -0.02;
  }

  machine(id: string): Machine | undefined {
    return this.project?.machines.find((m) => m.id === id);
  }

  private clearEntities(): void {
    for (const entity of [...this.zoneEntities, ...this.trackEntities])
      this.viewer.entities.remove(entity);
    this.zoneEntities.length = 0;
    this.trackEntities.length = 0;
  }

  destroy(): void {
    this.removeFrame();
    this.listeners.clear();
    this.clearEntities();
  }
}
