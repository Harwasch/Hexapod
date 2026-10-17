import {
  CallbackPositionProperty,
  Cartesian2,
  Cartesian3,
  ClassificationType,
  Color,
  ConstantProperty,
  type CesiumWidget,
  type Entity,
  HorizontalOrigin,
  NearFarScalar,
  PolygonHierarchy,
  PolylineDashMaterialProperty,
  type Scene,
  StripeMaterialProperty,
  StripeOrientation,
  VerticalOrigin,
} from "cesium";
import type { Position } from "geojson";

import { flattenRing, polygonsOf, type Footprint, type LonLat } from "@twin/geo";

import type { Rehearsal, RehearsalTrack } from "@/missions/rehearsal";
import {
  clusterHands,
  clusterLabel,
  HAND_KIND_TITLE as HAND_KIND_NAME,
  type HandCluster,
} from "@/missions/rehearsalClusters";
import { positionAt, REHEARSAL_SPEEDS, type RehearsalSpeed } from "@/missions/rehearsalClock";

import type { MissionManager } from "./MissionManager";

/** Every entity the rehearsal draws. */
export const REHEARSAL_ENTITY_PREFIX = "rehearsal:";
/** A cluster of raised hands, named by its first hand: a click opens that hand in the card. */
export const REHEARSAL_HAND_PREFIX = `${REHEARSAL_ENTITY_PREFIX}hand:`;
/** The hand open in the card, marked on its own: a click keeps it open. */
const REHEARSAL_FOCUS_PREFIX = `${REHEARSAL_ENTITY_PREFIX}focus:`;
/**
 * Ground a workable-ground redraft left out, hatched (`showLeftOut`). It outlives a rehearsal
 * (the redraft stops the run that asked for it) and goes when the plan card does.
 */
const LEFT_OUT_PREFIX = `${REHEARSAL_ENTITY_PREFIX}left-out:`;
/** The ghost of one of the project's machines: a click selects that machine. */
export const REHEARSAL_GHOST_PREFIX = `${REHEARSAL_ENTITY_PREFIX}ghost:`;
/** An assumed generic machine's ghost: nothing to select, so a click goes to what is under it. */
const REHEARSAL_SIM_PREFIX = `${REHEARSAL_ENTITY_PREFIX}sim:`;

/**
 * What a click on a rehearsal entity means (SelectionManager): a hand opens in the card, a
 * project machine's ghost selects the machine, and everything else the rehearsal draws (lanes,
 * creeks, generic ghosts) is looked through to the zone, machine or place under it. Null for an
 * entity that is not the rehearsal's.
 */
export function rehearsalPick(
  id: string,
): { kind: "hand"; handId: string } | { kind: "machine"; machineId: string } | "through" | null {
  if (id.startsWith(REHEARSAL_HAND_PREFIX))
    return { kind: "hand", handId: id.slice(REHEARSAL_HAND_PREFIX.length) };
  if (id.startsWith(REHEARSAL_FOCUS_PREFIX))
    return { kind: "hand", handId: id.slice(REHEARSAL_FOCUS_PREFIX.length) };
  if (id.startsWith(REHEARSAL_GHOST_PREFIX))
    return { kind: "machine", machineId: id.slice(REHEARSAL_GHOST_PREFIX.length) };
  return id.startsWith(REHEARSAL_ENTITY_PREFIX) ? "through" : null;
}

const PENDING = Color.fromCssColorString("#c9d1cc").withAlpha(0.75);
const HAND = Color.fromCssColorString("#f0b254");
const WATER = Color.fromCssColorString("#6ec3f0").withAlpha(0.9);
const INK = "#0a0e09";
/** The hatch of left-out ground, by why it was left out. */
const LEFT_OUT_COLOR: Record<LeftOutReason, string> = {
  steep: "#e4572e",
  "creek-buffer": "#6ec3f0",
  sliver: "#c9d1cc",
  "no-data": "#c9d1cc",
};
/** Only these carry a label on the map; slivers and no-data ground are small and listed. */
const LABELLED_REASONS: readonly LeftOutReason[] = ["steep", "creek-buffer"];
/** React hears the clock at most this often while playing; state changes go at once. */
const FRAME_INTERVAL_MS = 100;
/**
 * A frame that took longer than this advances the clock by no more: a tab in the background
 * (no frames at all) does not jump to the end, while a slow machine still plays at its speed.
 */
const MAX_STEP_S = 1;

export interface RehearsalFrame {
  /** Simulated seconds from the start. */
  t: number;
  durationS: number;
  playing: boolean;
  speed: RehearsalSpeed;
  /** Hands the ghosts have reached so far. */
  raised: number;
  ended: boolean;
}

export type RehearsalFrameListener = (frame: RehearsalFrame | null) => void;

export type LeftOutReason = "steep" | "creek-buffer" | "sliver" | "no-data";

/** Ground left out of a workable-ground redraft, for the hatch. */
export interface LeftOutGround {
  reason: LeftOutReason;
  /** e.g. "Left out: too steep". */
  label: string;
  footprint: Footprint;
}

export interface RehearsalLook {
  /**
   * A machine's colour: the project's for its own machines; for a generic one (or one the
   * project does not list) a palette slot after the project's own, so no two ghosts share one.
   */
  colorOf: (machineId: string) => Color;
  /** Ellipsoid height of the ground at a point, metres (ghosts and hands stand on it). */
  heightAt: (longitude: number, latitude: number) => number;
}

function svgUri(svg: string): string {
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

/** A ghost: a translucent disc in the machine's colour with a dark core. */
function ghostImage(color: Color): string {
  const css = color.toCssHexString();
  return svgUri(
    `<svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 30 30"><circle cx="15" cy="15" r="12" fill="${css}" fill-opacity="0.55" stroke="${INK}" stroke-opacity="0.8" stroke-width="2" stroke-dasharray="4 3"/><circle cx="15" cy="15" r="4" fill="${INK}" fill-opacity="0.75"/></svg>`,
  );
}

const HAND_GLYPH = `<g transform="translate(6 12) scale(0.92)" fill="none" stroke="${INK}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 11V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2"/><path d="M14 10V4a2 2 0 0 0-2-2a2 2 0 0 0-2 2v2"/><path d="M10 10.5V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2v8"/><path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/></g>`;

/**
 * A raised hand: Lucide's hand on an amber disc, with the count of a cluster's raised hands in
 * a dark badge at its top right (none for one). `ring` marks the hand open in the card.
 */
function handImage(count: number, ring = false): string {
  const badge =
    count > 1
      ? (() => {
          const text = String(count);
          const width = Math.max(18, 7 * text.length + 8);
          const x = 46 - width;
          return `<rect x="${x}" y="1" width="${width}" height="18" rx="9" fill="${INK}" stroke="#f0b254" stroke-width="1.5"/><text x="${x + width / 2}" y="14" text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-size="12" font-weight="700" fill="#f6f1e4">${text}</text>`;
        })()
      : "";
  const halo = ring
    ? `<circle cx="17" cy="23" r="16.5" fill="none" stroke="#f6f1e4" stroke-width="3"/>`
    : "";
  return svgUri(
    `<svg xmlns="http://www.w3.org/2000/svg" width="48" height="40" viewBox="0 0 48 40">${halo}<circle cx="17" cy="23" r="15" fill="#f0b254" stroke="${INK}" stroke-width="2"/>${HAND_GLYPH}${badge}</svg>`,
  );
}

/** Every badge drawn once: a count's image is the same image each time. */
const handImages = new Map<string, string>();
function cachedHandImage(count: number, ring = false): string {
  const key = `${count}:${ring ? 1 : 0}`;
  let image = handImages.get(key);
  if (!image) {
    image = handImage(count, ring);
    handImages.set(key, image);
  }
  return image;
}

/** A point inside a ring's polygon: the middle of its widest span across its middle latitude. */
function interiorPoint(rings: readonly Position[][]): LonLat | null {
  const outer = rings[0] ?? [];
  let south = Infinity;
  let north = -Infinity;
  for (const [, lat] of outer) {
    if (lat === undefined) continue;
    south = Math.min(south, lat);
    north = Math.max(north, lat);
  }
  if (!Number.isFinite(south)) return null;
  const y = (south + north) / 2;
  const xs: number[] = [];
  for (const ring of rings)
    for (let i = 0; i + 1 < ring.length; i++) {
      const [x0, y0] = ring[i] ?? [];
      const [x1, y1] = ring[i + 1] ?? [];
      if (x0 === undefined || y0 === undefined || x1 === undefined || y1 === undefined) continue;
      if (y0 <= y === y1 <= y) continue;
      xs.push(x0 + ((y - y0) * (x1 - x0)) / (y1 - y0));
    }
  xs.sort((a, b) => a - b);
  let best: LonLat | null = null;
  let widest = -1;
  for (let i = 0; i + 1 < xs.length; i += 2) {
    const a = xs[i] ?? 0;
    const b = xs[i + 1] ?? 0;
    if (b - a > widest) {
      widest = b - a;
      best = { longitude: (a + b) / 2, latitude: y };
    }
  }
  return best;
}

/** About how far apart the hatch's stripes are, metres. */
const HATCH_SPACING_M = 12;

/** A ring's east-west width, metres (the stripes run across it). */
function ringWidthM(ring: readonly Position[]): number {
  let west = Infinity;
  let east = -Infinity;
  let latitude = 0;
  for (const [lon, lat] of ring) {
    if (lon === undefined) continue;
    west = Math.min(west, lon);
    east = Math.max(east, lon);
    latitude = lat ?? latitude;
  }
  return Number.isFinite(west) ? (east - west) * Math.cos((latitude * Math.PI) / 180) * 111_320 : 0;
}

/** A polygon's size for picking where to label, in square degrees (outer ring, shoelace). */
function ringArea(ring: readonly Position[]): number {
  let twice = 0;
  for (let i = 0; i + 1 < ring.length; i++) {
    const [x0 = 0, y0 = 0] = ring[i] ?? [];
    const [x1 = 0, y1 = 0] = ring[i + 1] ?? [];
    twice += x0 * y1 - x1 * y0;
  }
  return Math.abs(twice) / 2;
}

function laneId(zoneId: string, index: number): string {
  return `${zoneId}:${index}`;
}

/**
 * Plays a dress rehearsal (`missions/rehearsal.ts`) on the map: ghost machines on the plan's
 * drawn lanes, the lanes going from pending (dashed grey) to done (the machine's colour, or
 * amber where a hand went up), and raised hands once a ghost reaches them. Hands stand in
 * clusters (`missions/rehearsalClusters.ts`): one marker per cluster at its first hand, with
 * the count of its hands raised so far, so a steep ranch shows a dozen markers and not a
 * hundred and fifty; the hand open in the card has a marker of its own. It has its own
 * animation-frame loop, which asks the scene for a frame only while playing (the scene renders
 * on request), at 300 or 600 simulated seconds per second. The plan's own passes are hidden
 * while it is loaded and come back on `stop`.
 *
 * It also hatches the ground a workable-ground redraft left out (`showLeftOut`), which stays
 * through the redraft's own rehearsal and goes when the plan card says so.
 */
export class RehearsalManager {
  private readonly scene: Scene;
  private rehearsal: Rehearsal | null = null;
  private look: RehearsalLook | null = null;
  private t = 0;
  private speed: RehearsalSpeed = REHEARSAL_SPEEDS[0];
  private playing = false;
  /** The pending animation frame while playing. */
  private request: number | null = null;
  private lastTick: number | null = null;
  private lastEmit = 0;
  private lastRaised = -1;
  private focused: string | null = null;
  private readonly pending = new Map<string, Entity>();
  private readonly done = new Map<string, Entity>();
  /** One marker per cluster of hands, and what it shows now. */
  private readonly clusters: {
    cluster: HandCluster;
    entity: Entity;
    count: number;
    label: string | null;
  }[] = [];
  /** The hand open in the card, marked on its own. */
  private focusEntity: Entity | null = null;
  /** The hatched left-out ground and its labels (`showLeftOut`). */
  private readonly leftOut: Entity[] = [];
  /** The ghosts, and the creek lines drawn with them. */
  private readonly ghosts: Entity[] = [];
  private readonly listeners = new Set<RehearsalFrameListener>();

  constructor(
    private readonly viewer: CesiumWidget,
    private readonly mission: MissionManager,
  ) {
    this.scene = viewer.scene;
  }

  get loaded(): Rehearsal | null {
    return this.rehearsal;
  }

  /** Draws a rehearsal at `t` (paused), replacing any other. */
  load(rehearsal: Rehearsal, look: RehearsalLook, t = 0): void {
    const playing = this.playing;
    this.clear();
    this.rehearsal = rehearsal;
    this.look = look;
    this.mission.setPreviewPassesShown(false);
    const positions = (points: readonly LonLat[]) =>
      Cartesian3.fromDegreesArray(points.flatMap((p) => [p.longitude, p.latitude]));
    // The creeks the buffer was checked against, so a creek hand points at something drawn.
    rehearsal.creeks.forEach((creek, index) => {
      if (creek.coordinates.length < 2) return;
      this.ghosts.push(
        this.viewer.entities.add({
          id: `${REHEARSAL_ENTITY_PREFIX}creek:${index}`,
          name: creek.name ?? "Mapped creek",
          polyline: {
            positions: positions(creek.coordinates),
            width: 2,
            material: WATER,
            clampToGround: true,
          },
        }),
      );
    });
    const flagged = new Set(rehearsal.hands.map((h) => laneId(h.zoneId, h.index)));
    for (const lane of rehearsal.lanes) {
      const id = laneId(lane.zoneId, lane.index);
      this.pending.set(
        id,
        this.viewer.entities.add({
          id: `${REHEARSAL_ENTITY_PREFIX}lane:${id}`,
          polyline: {
            positions: positions(lane.points),
            width: 2,
            material: new PolylineDashMaterialProperty({ color: PENDING, dashLength: 24 }),
            clampToGround: true,
          },
        }),
      );
      // Drawn once and shown when done: adding and removing a ground polyline rebuilds its
      // whole batch, every finished lane with it, on each lane and each scrub.
      const warn = flagged.has(id);
      this.done.set(
        id,
        this.viewer.entities.add({
          id: `${REHEARSAL_ENTITY_PREFIX}done:${id}`,
          show: false,
          polyline: {
            positions: positions(lane.points),
            width: warn ? 3 : 2.4,
            material: warn ? HAND : look.colorOf(lane.machineId).withAlpha(0.9),
            clampToGround: true,
          },
        }),
      );
    }
    for (const cluster of clusterHands(rehearsal.hands)) {
      this.clusters.push({
        cluster,
        count: 0,
        label: null,
        entity: this.handEntity(`${REHEARSAL_HAND_PREFIX}${cluster.first.id}`, cluster.first.at, {
          name: `Raised hands: ${HAND_KIND_NAME[cluster.kind]}, ${cluster.hands.length} lane${cluster.hands.length === 1 ? "" : "s"}`,
          image: cachedHandImage(1),
          show: false,
        }),
      });
    }
    rehearsal.tracks.forEach((track, index) => {
      const color = look.colorOf(track.machine.id);
      this.ghosts.push(
        this.viewer.entities.add({
          id: `${track.machine.generic ? REHEARSAL_SIM_PREFIX : REHEARSAL_GHOST_PREFIX}${track.machine.id}`,
          name: `${track.machine.name} (simulated rehearsal)`,
          position: new CallbackPositionProperty(() => this.ghostPosition(track), false),
          billboard: {
            image: ghostImage(color),
            width: 28,
            height: 28,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
          label: {
            text: track.machine.label,
            font: "600 12px Azeret Mono, monospace",
            fillColor: Color.fromCssColorString(INK),
            showBackground: true,
            backgroundColor: color.withAlpha(0.85),
            backgroundPadding: new Cartesian2(6, 3),
            horizontalOrigin: HorizontalOrigin.LEFT,
            verticalOrigin: VerticalOrigin.CENTER,
            // Machines sharing a zone start side by side: their labels alternate up and down.
            pixelOffset: new Cartesian2(17, index % 2 === 0 ? -7 : 7),
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
        }),
      );
    });
    this.t = Math.min(Math.max(0, t), rehearsal.durationS);
    // A hand open in the card keeps its marker through a replan (a lane left out, the
    // ephemeral drainages counted), if the new run still raises it.
    if (this.focused) this.focus(this.focused, false);
    this.sync();
    this.emit(true);
    this.scene.requestRender();
    if (playing && this.t < rehearsal.durationS) this.play();
  }

  play(speed: RehearsalSpeed = this.speed): void {
    const rehearsal = this.rehearsal;
    if (!rehearsal) return;
    this.speed = speed;
    if (this.t >= rehearsal.durationS) this.seek(0);
    if (this.playing) {
      this.emit(true);
      return;
    }
    this.playing = true;
    this.lastTick = null;
    this.request = requestAnimationFrame(this.tick);
    this.emit(true);
  }

  pause(): void {
    if (!this.playing) return;
    this.playing = false;
    if (this.request !== null) cancelAnimationFrame(this.request);
    this.request = null;
    this.emit(true);
  }

  setSpeed(speed: RehearsalSpeed): void {
    this.speed = speed;
    this.emit(true);
  }

  /** Jumps the clock (simulated seconds); the end shows every hand and the whole run. */
  seek(t: number): void {
    const rehearsal = this.rehearsal;
    if (!rehearsal) return;
    this.t = Math.min(Math.max(0, t), rehearsal.durationS);
    if (this.t >= rehearsal.durationS) this.pause();
    this.sync();
    this.emit(true);
    this.scene.requestRender();
  }

  /**
   * Marks one hand with its own labelled marker (null: none; the latest cluster speaks) and
   * flies to it.
   */
  focus(handId: string | null, fly = true): void {
    this.focused = handId;
    if (this.focusEntity) this.viewer.entities.remove(this.focusEntity);
    this.focusEntity = null;
    const hand = this.rehearsal?.hands.find((h) => h.id === handId);
    if (hand) {
      this.focusEntity = this.handEntity(`${REHEARSAL_FOCUS_PREFIX}${hand.id}`, hand.at, {
        name: `Raised hand: ${hand.label}`,
        image: cachedHandImage(1, true),
        show: true,
        label: hand.label,
      });
    }
    this.sync();
    this.scene.requestRender();
    if (hand && fly) this.mission.flyToPoint(hand.at, 160);
  }

  /**
   * Hatches ground a workable-ground redraft left out, by reason, each reason's largest piece
   * labelled ("Left out: too steep · derived"). Replaces what was hatched; `[]` clears it.
   * `heightAt` stands the labels on the ground, as the ghosts stand.
   */
  showLeftOut(
    pieces: readonly LeftOutGround[],
    heightAt: (longitude: number, latitude: number) => number = () => 0,
  ): void {
    for (const entity of this.leftOut) this.viewer.entities.remove(entity);
    this.leftOut.length = 0;
    // Stripes repeat across a polygon's own extent, so a polygon gets about one stripe per
    // HATCH_SPACING_M: the count rounded to a power of two, so the materials (one per reason and
    // count, shared, each drawn in one batch) stay few.
    const materials = new Map<string, StripeMaterialProperty>();
    const material = (reason: LeftOutReason, ring: Position[]) => {
      const repeat =
        2 ** Math.round(Math.log2(Math.max(2, Math.min(256, ringWidthM(ring) / HATCH_SPACING_M))));
      const key = `${reason}:${repeat}`;
      let found = materials.get(key);
      if (!found) {
        const color = Color.fromCssColorString(LEFT_OUT_COLOR[reason]);
        found = new StripeMaterialProperty({
          evenColor: color.withAlpha(0.5),
          oddColor: color.withAlpha(0.06),
          orientation: StripeOrientation.VERTICAL,
          repeat,
        });
        materials.set(key, found);
      }
      return found;
    };
    const largest = new Map<LeftOutReason, { area: number; rings: Position[][]; label: string }>();
    pieces.forEach((piece, pieceIndex) => {
      polygonsOf(piece.footprint).forEach(([ring, ...holes], part) => {
        if (!ring || ring.length < 4) return;
        this.leftOut.push(
          this.viewer.entities.add({
            id: `${LEFT_OUT_PREFIX}${pieceIndex}:${part}`,
            name: `${piece.label} (derived)`,
            polygon: {
              hierarchy: new PolygonHierarchy(
                Cartesian3.fromDegreesArray(flattenRing(ring)),
                holes.map(
                  (hole) => new PolygonHierarchy(Cartesian3.fromDegreesArray(flattenRing(hole))),
                ),
              ),
              material: material(piece.reason, ring),
              classificationType: ClassificationType.BOTH,
            },
          }),
        );
        const area = ringArea(ring);
        if (area > (largest.get(piece.reason)?.area ?? 0))
          largest.set(piece.reason, { area, rings: [ring, ...holes], label: piece.label });
      });
    });
    for (const reason of LABELLED_REASONS) {
      const best = largest.get(reason);
      const at = best ? interiorPoint(best.rings) : null;
      if (!best || !at) continue;
      this.leftOut.push(
        this.viewer.entities.add({
          id: `${LEFT_OUT_PREFIX}label:${reason}`,
          position: Cartesian3.fromDegrees(
            at.longitude,
            at.latitude,
            heightAt(at.longitude, at.latitude) + 3,
          ),
          label: {
            text: `${best.label} · derived`,
            font: "500 12px Inter, system-ui, sans-serif",
            fillColor: Color.fromCssColorString(INK),
            showBackground: true,
            backgroundColor: Color.fromCssColorString(LEFT_OUT_COLOR[reason]).withAlpha(0.92),
            backgroundPadding: new Cartesian2(7, 4),
            horizontalOrigin: HorizontalOrigin.CENTER,
            verticalOrigin: VerticalOrigin.BOTTOM,
            // Lifted clear of a zone chip, which hangs from its zone's point at the same height.
            pixelOffset: new Cartesian2(0, -30),
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
        }),
      );
    }
    this.scene.requestRender();
  }

  /** Clears the map and puts the plan's own passes back. */
  stop(): void {
    const had = this.rehearsal !== null;
    this.clear();
    this.rehearsal = null;
    this.look = null;
    this.focused = null;
    this.t = 0;
    if (had) {
      this.mission.setPreviewPassesShown(true);
      for (const listener of this.listeners) listener(null);
    }
    this.scene.requestRender();
  }

  onFrame(listener: RehearsalFrameListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  destroy(): void {
    this.stop();
    this.showLeftOut([]);
    this.listeners.clear();
  }

  /** A hand marker: the amber hand standing on the ground, its label (if any) above it. */
  private handEntity(
    id: string,
    at: LonLat,
    options: { name: string; image: string; show: boolean; label?: string },
  ): Entity {
    return this.viewer.entities.add({
      id,
      name: options.name,
      show: options.show,
      position: this.standing(at, 2),
      billboard: {
        image: options.image,
        width: 40,
        height: 34,
        // A steep ranch has a dozen markers: far ones shrink, the near ones stay readable.
        scaleByDistance: new NearFarScalar(400, 1, 6000, 0.6),
        // The disc's centre (17, 23 of 48 x 40) on the hand's point.
        horizontalOrigin: HorizontalOrigin.LEFT,
        verticalOrigin: VerticalOrigin.BOTTOM,
        pixelOffset: new Cartesian2(-14, 6),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
      label: {
        text: options.label ?? "",
        show: options.label !== undefined,
        font: "500 12px Inter, system-ui, sans-serif",
        fillColor: Color.fromCssColorString(INK),
        showBackground: true,
        backgroundColor: HAND.withAlpha(0.94),
        backgroundPadding: new Cartesian2(7, 4),
        horizontalOrigin: HorizontalOrigin.CENTER,
        verticalOrigin: VerticalOrigin.BOTTOM,
        // Centred above the hand, so it reads the same beside the plan window as anywhere.
        pixelOffset: new Cartesian2(0, -34),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
  }

  private readonly tick = (now: number): void => {
    const rehearsal = this.rehearsal;
    if (!this.playing || !rehearsal) return;
    const step = this.lastTick === null ? 0 : Math.min((now - this.lastTick) / 1000, MAX_STEP_S);
    this.lastTick = now;
    this.t = Math.min(rehearsal.durationS, this.t + step * this.speed);
    if (this.t >= rehearsal.durationS) {
      this.playing = false;
      this.request = null;
    }
    this.sync();
    this.scene.requestRender();
    this.emit(!this.playing);
    if (this.playing) this.request = requestAnimationFrame(this.tick);
  };

  private standing(at: LonLat, lift: number): Cartesian3 {
    const height = this.look?.heightAt(at.longitude, at.latitude) ?? 0;
    return Cartesian3.fromDegrees(at.longitude, at.latitude, height + lift);
  }

  private ghostPosition(track: RehearsalTrack): Cartesian3 | undefined {
    const where = positionAt(track, this.t);
    return where ? this.standing(where.at, 1.5) : undefined;
  }

  /** The lanes done by now and the hands reached by now, drawn as such. */
  private sync(): void {
    const rehearsal = this.rehearsal;
    if (!rehearsal) return;
    for (const lane of rehearsal.lanes) {
      const id = laneId(lane.zoneId, lane.index);
      // At the end every lane is done, whatever rounding left in its finishing time.
      const finished = this.t >= lane.t1 || this.t >= rehearsal.durationS;
      const done = this.done.get(id);
      if (done && done.show !== finished) done.show = finished;
      const pending = this.pending.get(id);
      if (pending?.show === finished) pending.show = !finished;
    }
    // The focused hand speaks alone (its own marker); otherwise the cluster of the latest hand
    // to go up, since the markers of neighbouring clusters would overlap their labels.
    const counts: number[] = [];
    let speaking: number | null = null;
    let latestT = -Infinity;
    this.clusters.forEach(({ cluster }, index) => {
      const up = cluster.hands.filter((h) => this.t >= h.t);
      counts.push(up.length);
      const last = up[up.length - 1];
      if (last && last.t >= latestT) {
        latestT = last.t;
        speaking = index;
      }
    });
    if (this.focused) speaking = null;
    this.clusters.forEach((item, index) => {
      const count = counts[index] ?? 0;
      const shown = count > 0;
      if (item.entity.show !== shown) item.entity.show = shown;
      if (shown && item.count !== count && item.entity.billboard)
        item.entity.billboard.image = new ConstantProperty(cachedHandImage(count));
      item.count = count;
      const label = shown && index === speaking ? clusterLabel(item.cluster, count) : null;
      if (item.label !== label && item.entity.label) {
        item.entity.label.show = new ConstantProperty(label !== null);
        if (label !== null) item.entity.label.text = new ConstantProperty(label);
        item.label = label;
      }
    });
  }

  /** Where the rehearsal is now (null when none is loaded); listeners hear it throttled. */
  get frame(): RehearsalFrame | null {
    const rehearsal = this.rehearsal;
    if (!rehearsal) return null;
    return {
      t: this.t,
      durationS: rehearsal.durationS,
      playing: this.playing,
      speed: this.speed,
      raised: rehearsal.hands.filter((h) => this.t >= h.t).length,
      ended: this.t >= rehearsal.durationS,
    };
  }

  private emit(force: boolean): void {
    const frame = this.frame;
    if (!frame) return;
    const now = performance.now();
    if (!force && frame.raised === this.lastRaised && now - this.lastEmit < FRAME_INTERVAL_MS)
      return;
    this.lastEmit = now;
    this.lastRaised = frame.raised;
    for (const listener of this.listeners) listener(frame);
  }

  private clear(): void {
    if (this.request !== null) cancelAnimationFrame(this.request);
    this.request = null;
    this.playing = false;
    for (const entity of [
      ...this.pending.values(),
      ...this.done.values(),
      ...this.clusters.map((item) => item.entity),
      ...(this.focusEntity ? [this.focusEntity] : []),
      ...this.ghosts,
    ])
      this.viewer.entities.remove(entity);
    this.pending.clear();
    this.done.clear();
    this.clusters.length = 0;
    this.focusEntity = null;
    this.ghosts.length = 0;
    this.lastRaised = -1;
  }
}
