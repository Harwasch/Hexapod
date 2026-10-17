import {
  BoundingSphere,
  Cartesian2,
  Cartesian3,
  Cartographic,
  Color,
  ColorMaterialProperty,
  ConstantProperty,
  type Entity,
  HorizontalOrigin,
  PolygonHierarchy,
  PolylineDashMaterialProperty,
  SceneTransforms,
  sampleTerrainMostDetailed,
  type Scene,
  type CesiumWidget,
  VerticalOrigin,
} from "cesium";

import {
  boundingRadiusM,
  centerOf,
  destination,
  flattenRing,
  footprintLanes,
  laneStrokes,
  polygonsOf,
  type CoverageLane,
} from "@twin/geo";

import type { Emitter } from "@/lib/emitter";
import { eyeOf, frameGround, projectGroundPoints, viewportOptions } from "@/missions/framing";
import { PREVIEW_MAX_PASSES, PREVIEW_SWATH_M } from "@/missions/preview";
import { scoutLanes, scoutOf } from "@/missions/scout";
import { siteAnchor } from "@/missions/sitePin";
import type { LonLat, Machine, PlanOverlay, Project, Zone } from "@/missions/types";

import type { CameraController } from "./CameraController";
import type { SceneEvents } from "./types";

export const ZONE_ENTITY_PREFIX = "mission:zone:";
export const PLAN_ENTITY_PREFIX = "mission:plan:";
/** One colour per machine so passes read as "who drives where". */
const MACHINE_PALETTE = ["#7fd8c0", "#f0b254", "#9db8f0", "#e79ac8", "#c6e07a", "#c3a6f0"].map(
  (css) => Color.fromCssColorString(css),
);
const ROUTE_COLOR = Color.fromCssColorString("#9db8f0");
/** A scout area's passes: pale, dashed, a forecast of where the fog would lift. */
const FOG_LIFT = Color.fromCssColorString("#e8eef0");
/** How far below its zone's point the forecast label hangs: past the zone chip (about 31 px). */
const FORECAST_LABEL_OFFSET_PX = 40;

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
const DETAILED_HEIGHT_TIMEOUT_MS = 1500;
const MAX_TILESET_TERRAIN_GAP_M = 60;
/** Anchor heights refresh slowly while moving (terrain only) and rarely at rest (with a pick). */
const MOVING_HEIGHT_TTL_MS = 1500;
const RESTING_HEIGHT_TTL_MS = 4000;

/** A lane's identity in a zone: its ends, to a centimetre or so. */
function laneKey(zoneId: string, lane: CoverageLane): string {
  const p = (q: LonLat) => `${q.longitude.toFixed(7)},${q.latitude.toFixed(7)}`;
  return `${zoneId}|${p(lane.a)}|${p(lane.b)}`;
}

function isPlausibleHeight(height: number | undefined): height is number {
  return height !== undefined && Number.isFinite(height) && height > -500 && height < 9000;
}

/**
 * Picks a ground height from a tileset sample and a terrain sample. Splat tilesets report the
 * surface of whatever gaussians are loaded, which at coarse LOD can float hundreds of metres above
 * the ground, and terrain tiles still loading report placeholders kilometres below sea level; the
 * tileset wins only when it agrees with the terrain to within a building's height.
 */
function reconcileHeights(
  fromTileset: number | undefined,
  fromTerrain: number | undefined,
): number | undefined {
  const tileset = isPlausibleHeight(fromTileset) ? fromTileset : undefined;
  const terrain = isPlausibleHeight(fromTerrain) ? fromTerrain : undefined;
  if (tileset !== undefined && terrain !== undefined) {
    return Math.abs(tileset - terrain) <= MAX_TILESET_TERRAIN_GAP_M ? tileset : terrain;
  }
  return tileset ?? terrain;
}

export class MissionManager {
  private readonly scene: Scene;
  private project: Project | null = null;
  private readonly zoneEntities: Entity[] = [];
  private readonly trackEntities: Entity[] = [];
  private zonesVisible = true;
  /** Zones another layer draws (the selected Land's boundary): no entity, no chip anchor. */
  private hiddenZoneIds: ReadonlySet<string> = new Set();
  private tracksVisible = true;
  private readonly planEntities: Entity[] = [];
  /** The lanes drawn for each zone of the plan overlay (`previewLanes`). */
  private readonly drawnLanes = new Map<string, CoverageLane[]>();
  /** Each drawn lane's number in its zone counting the left-out ones, so numbers stay put. */
  private readonly drawnNumbers = new Map<string, number[]>();
  /**
   * Lanes the operator left out of the draft's preview (the rehearsal's "Leave this lane
   * out"), by zone and lane geometry, so a redraw of the same outline keeps them out and a
   * changed outline drops them. Only the preview: a plan's zones and hours are unchanged.
   */
  private readonly leftOut = new Map<string, CoverageLane & { zoneId: string }>();
  /** Off while a rehearsal draws the lanes in its own states (pending, done, a raised hand). */
  private passesShown = true;
  private overlay: PlanOverlay | null = null;
  private readonly listeners = new Set<FrameListener>();
  private readonly removeFrame: () => void;
  private lastGoodHeight: number | undefined;
  private readonly heightCache = new Map<
    string,
    { height: number; at: number; sampled?: boolean }
  >();
  /**
   * The terrain under each overlay anchor at full detail, sampled once per project. Under
   * photorealistic tiles the globe may never load its own terrain there, and its placeholder
   * heights are kilometres off; without this an anchor stood at the last height that answered
   * anywhere, or at sea level: on a ranch 300 m up, chips drew a hundred pixels off their zones.
   */
  private readonly groundHeights = new Map<string, number>();
  private exploring = false;
  private readonly removeExplore: () => void;
  private readonly scratchWindow = new Cartesian2();
  private readonly scratchNormal = new Cartesian3();
  private readonly scratchToCamera = new Cartesian3();

  constructor(
    private readonly viewer: CesiumWidget,
    private readonly events: Emitter<SceneEvents>,
    private readonly camera: CameraController,
  ) {
    this.scene = viewer.scene;
    this.removeFrame = this.scene.postRender.addEventListener(
      () => this.project && this.listeners.size > 0 && this.projectAnchors(),
    );
    // Walking or flying moves the camera in steps too small for `camera.changed`, so
    // `isMoving` stays false; a height sample (a render pass) is held off for the duration.
    this.removeExplore = this.events.on("explore", (on) => {
      this.exploring = on;
    });
  }

  get current(): Project | null {
    return this.project;
  }

  /** Zones to leave undrawn from the next `setProject` on (`missions/landGround.ts`). */
  setHiddenZones(ids: ReadonlySet<string>): void {
    this.hiddenZoneIds = ids;
  }

  setProject(project: Project | null): void {
    const overlay = this.overlay;
    this.clearEntities();
    this.project = project;
    this.heightCache.clear();
    this.groundHeights.clear();
    if (!project) return;
    this.sampleGround(project);
    // Zones may have been added or removed; the plan overlay is redrawn at the end.
    // Unless the overlay was cleared or replaced meanwhile: a stale one must not come back.
    if (overlay)
      queueMicrotask(
        () => this.project === project && this.overlay === overlay && this.showPlan(overlay),
      );
    const drawn = new Set<string>();
    for (const zone of project.zones) {
      const [ring, ...holes] = polygonsOf(zone.footprint)[0] ?? [];
      // One entity per id: a second zone with an id already drawn would throw in Cesium.
      if (!ring || this.hiddenZoneIds.has(zone.id) || drawn.has(zone.id)) continue;
      drawn.add(zone.id);
      const positions = Cartesian3.fromDegreesArray(flattenRing(ring));
      const first = positions[0];
      if (!first) continue;
      const tone = TONE[zone.tone];
      this.zoneEntities.push(
        this.viewer.entities.add({
          id: `${ZONE_ENTITY_PREFIX}${zone.id}`,
          show: this.zonesVisible,
          polygon: {
            // Holes stay open: the woods and the homestead inside a grassland zone are not in it.
            hierarchy: new PolygonHierarchy(
              positions,
              holes.map(
                (hole) => new PolygonHierarchy(Cartesian3.fromDegreesArray(flattenRing(hole))),
              ),
            ),
            material: tone.withAlpha(zone.treated ? 0.22 : 0.14),
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

  /** Colour used for a machine's passes, stable for the project. */
  machineColor(machineId: string): Color {
    const index = this.project?.machines.findIndex((m) => m.id === machineId) ?? -1;
    return MACHINE_PALETTE[(index < 0 ? 0 : index) % MACHINE_PALETTE.length] ?? ROUTE_COLOR;
  }

  /**
   * Draws a plan on the world: every zone in scope with its coverage passes in the colour of
   * the machine assigned, a numbered marker per step, and the route between zones in step
   * order. `null` clears it. Passes come from `footprintLanes` over every part of a zone with
   * its holes left out, so nothing is drawn over a homestead or a cleared fog hole. Three
   * styles: a mapped line (a fence) is drawn as the line it follows; a scout area ("Lift the
   * fog") is drawn as its real flight lines, dashed and labelled a forecast; anything else as
   * preview passes (see PREVIEW_SWATH_M). `previewLanes` hands out exactly what is drawn.
   */
  showPlan(overlay: PlanOverlay | null): void {
    // A different plan, or none: lanes left out of the last one's preview come back.
    const zones = (o: PlanOverlay | null) => (o?.zones.map((z) => z.zoneId) ?? []).join(",");
    if (!overlay || zones(overlay) !== zones(this.overlay)) this.leftOut.clear();
    this.overlay = overlay;
    for (const entity of this.planEntities) this.viewer.entities.remove(entity);
    this.planEntities.length = 0;
    this.drawnLanes.clear();
    this.drawnNumbers.clear();
    this.setSelectedZone(null);
    const project = this.project;
    if (!overlay || !project) {
      this.scene.requestRender();
      return;
    }
    const centers: Cartesian3[] = [];
    overlay.zones.forEach((item, index) => {
      const zone = project.zones.find((z) => z.id === item.zoneId);
      if (!zone || polygonsOf(zone.footprint).length === 0) return;
      const color = this.machineColor(item.machineIds[0] ?? "");
      // A mapped line (a fence) is followed, not covered: draw the line, not passes over its
      // corridor.
      const line = zone.land?.line?.coordinates ?? null;
      const scout = zone.scout ? scoutOf(zone.scout) : null;
      const all: CoverageLane[] = line
        ? []
        : scout
          ? scoutLanes(zone.footprint, scout)
          : footprintLanes(zone.footprint, {
              swathM: PREVIEW_SWATH_M,
              maxPasses: PREVIEW_MAX_PASSES,
            }).lanes;
      const lanes: CoverageLane[] = [];
      const numbers: number[] = [];
      all.forEach((lane, index) => {
        if (this.leftOut.has(laneKey(zone.id, lane))) return;
        lanes.push(lane);
        numbers.push(index + 1);
      });
      this.drawnLanes.set(zone.id, lanes);
      this.drawnNumbers.set(zone.id, numbers);
      if (line && line.length >= 2) {
        this.planEntities.push(
          this.viewer.entities.add({
            id: `${PLAN_ENTITY_PREFIX}line:${zone.id}`,
            polyline: {
              positions: Cartesian3.fromDegreesArray(line.flatMap((p) => [p[0] ?? 0, p[1] ?? 0])),
              width: 3.2,
              material: color,
              clampToGround: true,
            },
          }),
        );
      }
      // Consecutive passes join at the turn; lanes of one pass split by a hole stay apart.
      laneStrokes(lanes).forEach((stroke, strokeIndex) => {
        if (stroke.length < 2) return;
        this.planEntities.push(
          this.viewer.entities.add({
            // The first stroke keeps the zone's plain id; one split by a hole or a notch
            // adds numbered ones.
            id: `${PLAN_ENTITY_PREFIX}passes:${zone.id}${strokeIndex ? `:${strokeIndex}` : ""}`,
            show: this.passesShown,
            polyline: {
              positions: Cartesian3.fromDegreesArray(
                stroke.flatMap((p) => [p.longitude, p.latitude]),
              ),
              width: scout ? 1.8 : 1.6,
              material: scout
                ? new PolylineDashMaterialProperty({ color: FOG_LIFT, dashLength: 10 })
                : color.withAlpha(0.8),
              clampToGround: true,
            },
          }),
        );
      });
      const outline = this.zoneEntities.find((e) => e.id === `${ZONE_ENTITY_PREFIX}${zone.id}`);
      if (outline?.polyline) {
        outline.polyline.width = new ConstantProperty(3.2);
        outline.polyline.material = new ColorMaterialProperty(scout ? FOG_LIFT : color);
      }
      const center = centerOf(zone.footprint);
      centers.push(Cartesian3.fromDegrees(center.longitude, center.latitude));
      // Labels stand on the ground as the zone chips do (`surfaceHeight`): clamping a label
      // needs the globe's terrain, which photorealistic tiles hide, and then it sank to the
      // ellipsoid, hundreds of metres under a ranch and off its zone.
      const position = Cartesian3.fromDegrees(
        center.longitude,
        center.latitude,
        this.surfaceHeight(center.longitude, center.latitude, `plan:${zone.id}`) + 3,
      );
      this.planEntities.push(
        this.viewer.entities.add({
          id: `${PLAN_ENTITY_PREFIX}step:${zone.id}`,
          position,
          label: {
            text: `${index + 1}`,
            font: "600 13px Azeret Mono, monospace",
            fillColor: Color.fromCssColorString("#0a0e09"),
            showBackground: true,
            backgroundColor: color,
            backgroundPadding: new Cartesian2(7, 4),
            // Above the zone's chip, which starts at the same point and runs right.
            horizontalOrigin: HorizontalOrigin.LEFT,
            verticalOrigin: VerticalOrigin.BOTTOM,
            pixelOffset: new Cartesian2(0, -16),
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
        }),
      );
      if (scout) {
        // Said where the lines are: these passes are what a flight would do, not evidence.
        this.planEntities.push(
          this.viewer.entities.add({
            id: `${PLAN_ENTITY_PREFIX}forecast:${zone.id}`,
            position,
            label: {
              text: "Would lift fog · forecast",
              font: "500 12px Inter, system-ui, sans-serif",
              fillColor: Color.fromCssColorString("#0a0e09"),
              showBackground: true,
              backgroundColor: FOG_LIFT.withAlpha(0.92),
              backgroundPadding: new Cartesian2(7, 4),
              // Below the zone's chip, an HTML button about 31 px tall that hangs from the same
              // point and is drawn over the canvas: clear all of it, or it hides the label.
              horizontalOrigin: HorizontalOrigin.LEFT,
              verticalOrigin: VerticalOrigin.TOP,
              pixelOffset: new Cartesian2(0, FORECAST_LABEL_OFFSET_PX),
              disableDepthTestDistance: Number.POSITIVE_INFINITY,
            },
          }),
        );
      }
    });
    if (centers.length >= 2) {
      this.planEntities.push(
        this.viewer.entities.add({
          id: `${PLAN_ENTITY_PREFIX}route`,
          polyline: {
            positions: centers,
            width: 2.4,
            material: new PolylineDashMaterialProperty({ color: ROUTE_COLOR, dashLength: 18 }),
            clampToGround: true,
          },
        }),
      );
    }
    this.scene.requestRender();
  }

  /**
   * The lanes the plan overlay draws for a zone, in driving order (empty for a mapped line or
   * a zone not on the map): what a scout export flies and a rehearsal runs.
   */
  previewLanes(zoneId: string): readonly CoverageLane[] {
    return this.drawnLanes.get(zoneId) ?? [];
  }

  /** The number of each of `previewLanes(zoneId)`, from 1, with left-out lanes still counted. */
  previewLaneNumbers(zoneId: string): readonly number[] {
    return this.drawnNumbers.get(zoneId) ?? [];
  }

  /** Takes one drawn lane out of the draft's preview and redraws it without. */
  leaveOutLane(zoneId: string, lane: CoverageLane): void {
    this.leftOut.set(laneKey(zoneId, lane), { ...lane, zoneId });
    if (this.overlay) this.showPlan(this.overlay);
  }

  /** Every lane left out of the preview comes back. */
  restoreLanes(): void {
    if (this.leftOut.size === 0) return;
    this.leftOut.clear();
    if (this.overlay) this.showPlan(this.overlay);
  }

  /** The lanes left out of the current preview, in the order they were left out. */
  leftOutLanes(): readonly (CoverageLane & { zoneId: string })[] {
    return [...this.leftOut.values()];
  }

  /** Shows or hides the drawn passes (zone outlines and step markers stay). */
  setPreviewPassesShown(shown: boolean): void {
    this.passesShown = shown;
    for (const entity of this.planEntities)
      if (typeof entity.id === "string" && entity.id.startsWith(`${PLAN_ENTITY_PREFIX}passes:`))
        entity.show = shown;
    this.scene.requestRender();
  }

  /**
   * Flies to a point on the ground, framing `radiusM` around it; like `flyToZones`, the framed
   * centre moves screen-right by `sideOffset` radii so the point lands beside the plan window.
   */
  flyToPoint(point: LonLat, radiusM = 80, sideOffset = 0.75): void {
    const heading = 20;
    const h = (heading * Math.PI) / 180;
    const target = destination(
      point,
      (Math.atan2(Math.cos(h), -Math.sin(h)) * 180) / Math.PI,
      radiusM * sideOffset,
    );
    const key = `point-fly:${target.longitude.toFixed(5)},${target.latitude.toFixed(5)}`;
    void this.detailedHeight(target.longitude, target.latitude, key).then((height) => {
      this.camera.flyToBoundingSphere(
        new BoundingSphere(
          Cartesian3.fromDegrees(target.longitude, target.latitude, height),
          radiusM * (1 + sideOffset * 0.4),
        ),
        { heading, pitch: -40, rangeMultiplier: 2.6, durationS: 1.4 },
      );
    });
  }

  /**
   * Flies to fit every listed zone. The plan window covers the right of the screen, so the
   * framed centre is pushed towards screen-right by `sideOffset` radii and the zones land in
   * the open part of the view.
   */
  flyToZones(zoneIds: string[], sideOffset = 0.75): void {
    const zones = (this.project?.zones ?? []).filter((z) => zoneIds.includes(z.id));
    if (zones.length === 0) return;
    const centers = zones.map((z) => centerOf(z.footprint));
    let lon = centers.reduce((a, c) => a + c.longitude, 0) / centers.length;
    let lat = centers.reduce((a, c) => a + c.latitude, 0) / centers.length;
    const mid = Cartesian3.fromDegrees(lon, lat);
    let radius = 120;
    for (const zone of zones) {
      const c = centerOf(zone.footprint);
      radius = Math.max(
        radius,
        Cartesian3.distance(mid, Cartesian3.fromDegrees(c.longitude, c.latitude)) +
          boundingRadiusM(zone.footprint),
      );
    }
    const heading = 20;
    // Screen-right on the ground for a camera at `heading`: east cos(h), north -sin(h).
    const h = (heading * Math.PI) / 180;
    const shift = destination(
      { longitude: lon, latitude: lat },
      (Math.atan2(Math.cos(h), -Math.sin(h)) * 180) / Math.PI,
      radius * sideOffset,
    );
    lon = shift.longitude;
    lat = shift.latitude;
    void this.detailedHeight(lon, lat, "plan-fly").then((height) => {
      this.camera.flyToBoundingSphere(
        new BoundingSphere(
          Cartesian3.fromDegrees(lon, lat, height),
          radius * (1 + sideOffset * 0.4),
        ),
        { heading, pitch: -42, rangeMultiplier: 2.2, durationS: 1.8 },
      );
    });
  }

  onFrame(listener: FrameListener): () => void {
    this.listeners.add(listener);
    // A new listener has nodes waiting for a position (overlays that just mounted, the site
    // pin that replaced them); in request-render mode the next frame may be a while coming.
    this.scene.requestRender();
    return () => this.listeners.delete(listener);
  }

  flyToZone(zoneId: string): void {
    const zone = this.project?.zones.find((z) => z.id === zoneId);
    if (!zone) return;
    const center = centerOf(zone.footprint);
    const radius = Math.max(boundingRadiusM(zone.footprint), 80);
    void this.detailedHeight(center.longitude, center.latitude, `zone-fly-${zone.id}`).then(
      (height) => {
        this.camera.flyToBoundingSphere(
          new BoundingSphere(
            Cartesian3.fromDegrees(center.longitude, center.latitude, height),
            radius,
          ),
          { heading: 20, pitch: -40, rangeMultiplier: 2.6, durationS: 1.8 },
        );
      },
    );
  }

  flyToMachine(machineId: string): void {
    const machine = this.project?.machines.find((m) => m.id === machineId);
    if (!machine) return;
    const { longitude, latitude } = machine.position;
    void this.detailedHeight(longitude, latitude, `machine-fly-${machine.id}`).then((height) => {
      this.camera.flyToBoundingSphere(
        new BoundingSphere(Cartesian3.fromDegrees(longitude, latitude, height), 60),
        { heading: machine.headingDeg + 180, pitch: -34, rangeMultiplier: 4, durationS: 1.6 },
      );
    });
  }

  /**
   * Flies to the whole project, fitted to this window: every zone outline, chip and machine
   * inside the part of the screen the HUD leaves open (`missions/framing.ts`), so the site pin
   * and "the whole ranch" land framed on a phone held upright as on a desktop.
   */
  flyToProject(): void {
    const project = this.project;
    if (!project || project.zones.length === 0) return;
    const framing = frameGround(
      projectGroundPoints(project),
      viewportOptions(this.camera.viewport(), 15, -40),
    );
    const { longitude, latitude } = framing.target;
    void this.detailedHeight(longitude, latitude, "project-fly").then((height) => {
      const eye = eyeOf(framing, height);
      this.camera.flyTo(eye.longitude, eye.latitude, eye.height, {
        heading: eye.heading,
        pitch: eye.pitch,
        durationS: 2,
      });
    });
  }

  /**
   * Height for a camera target. The synchronous estimate only knows about tiles that are already
   * loaded, which right after a site flight can be coarse terrain tens of metres off; ask for the
   * most detailed terrain and tileset samples, but never wait longer than a beat for them.
   */
  private async detailedHeight(longitude: number, latitude: number, key: string): Promise<number> {
    const quick = this.surfaceHeight(longitude, latitude, key);
    const timeout = new Promise<undefined>((resolve) => {
      setTimeout(() => resolve(undefined), DETAILED_HEIGHT_TIMEOUT_MS);
    });
    const fromTileset = this.scene.sampleHeightSupported
      ? this.scene
          .sampleHeightMostDetailed([Cartographic.fromDegrees(longitude, latitude)])
          .then((result) => result[0]?.height)
          .catch(() => undefined)
      : Promise.resolve(undefined);
    const fromTerrain = sampleTerrainMostDetailed(this.viewer.terrainProvider, [
      Cartographic.fromDegrees(longitude, latitude),
    ])
      .then((result) => result[0]?.height)
      .catch(() => undefined);
    let tilesetHeight: number | undefined;
    let terrainHeight: number | undefined;
    void fromTileset.then((h) => (tilesetHeight = h));
    void fromTerrain.then((h) => (terrainHeight = h));
    await Promise.race([Promise.all([fromTileset, fromTerrain]), timeout]);
    const detailed = reconcileHeights(tilesetHeight, terrainHeight);
    if (detailed === undefined) return quick;
    this.lastGoodHeight = detailed;
    this.heightCache.set(key, { height: detailed, at: performance.now() });
    return detailed;
  }

  /** Where each overlay anchor of `project` stands, keyed as `projectAnchors` keys them. */
  private anchorPoints(project: Project): { key: string; longitude: number; latitude: number }[] {
    const points = [
      ...project.machines.map((m) => ({ key: `machine:${m.id}`, ...m.position })),
      ...project.zones.map((z) => ({ key: `zone:${z.id}`, ...z.anchor })),
    ];
    const site = siteAnchor(project);
    if (site) points.push({ key: `site:${project.id}`, ...site });
    return points;
  }

  /**
   * Samples the most detailed terrain under every anchor once; see `groundHeights`. A terrain
   * provider still loading (the app's first seconds) is asked again a little later.
   */
  private sampleGround(project: Project, attempt = 0): void {
    const points = this.anchorPoints(project);
    if (points.length === 0) return;
    sampleTerrainMostDetailed(
      this.viewer.terrainProvider,
      points.map((p) => Cartographic.fromDegrees(p.longitude, p.latitude)),
    )
      .then((samples) => {
        if (this.project !== project) return;
        samples.forEach((sample, i) => {
          const key = points[i]?.key;
          if (key && isPlausibleHeight(sample.height)) this.groundHeights.set(key, sample.height);
        });
        // Guesses made before it answered give way at the next frame; a drawn-surface sample stays.
        for (const key of this.groundHeights.keys()) {
          if (!this.heightCache.get(key)?.sampled) this.heightCache.delete(key);
        }
        this.scene.requestRender();
      })
      .catch(() => {
        if (attempt < 3 && this.project === project) {
          setTimeout(
            () => this.project === project && this.sampleGround(project, attempt + 1),
            2000,
          );
        }
      });
  }

  /**
   * Cheap ground estimate for overlay anchors and flights. `globe.getHeight` is a CPU lookup
   * into loaded terrain; `scene.sampleHeight` is a render pass, so it runs only when the camera
   * is at rest and at most once per anchor per few seconds.
   */
  private surfaceHeight(longitude: number, latitude: number, key: string): number {
    const now = performance.now();
    const cached = this.heightCache.get(key);
    const ttl = this.camera.isMoving ? MOVING_HEIGHT_TTL_MS : RESTING_HEIGHT_TTL_MS;
    // A height sampled from the drawn surface is kept: sampling again is a render pass and a
    // read-back that waits for the GPU (half a second of main thread in a trace, every few
    // seconds at rest), for a machine that has not moved.
    if (cached && (cached.sampled || now - cached.at < ttl)) return cached.height;
    const carto = Cartographic.fromDegrees(longitude, latitude);
    let fromTileset: number | undefined;
    if (this.scene.sampleHeightSupported && !this.camera.isMoving && !this.exploring) {
      try {
        fromTileset = this.scene.sampleHeight(carto, this.viewer.entities.values);
      } catch {
        fromTileset = undefined;
      }
    }
    const fromTerrain = this.groundHeights.get(key) ?? this.scene.globe.getHeight(carto);
    let height = reconcileHeights(fromTileset, fromTerrain);
    if (height === undefined) height = this.lastGoodHeight ?? 0;
    else this.lastGoodHeight = height;
    this.heightCache.set(key, { height, at: now, sampled: fromTileset !== undefined });
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
        if (!this.hiddenZoneIds.has(zone.id))
          push(`zone:${zone.id}`, zone.anchor.longitude, zone.anchor.latitude, 0.5);
    // The site pin that stands for all of the above at globe scale (`missions/sitePin.ts`).
    const site = siteAnchor(project);
    if (site) push(`site:${project.id}`, site.longitude, site.latitude, 0);
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
    for (const entity of [...this.zoneEntities, ...this.trackEntities, ...this.planEntities])
      this.viewer.entities.remove(entity);
    this.zoneEntities.length = 0;
    this.trackEntities.length = 0;
    this.planEntities.length = 0;
    // Zone outlines are rebuilt from scratch, so the highlight state is gone with them.
  }

  destroy(): void {
    this.removeFrame();
    this.removeExplore();
    this.listeners.clear();
    this.clearEntities();
  }
}
