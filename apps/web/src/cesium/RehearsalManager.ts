import {
  CallbackPositionProperty,
  Cartesian2,
  Cartesian3,
  Color,
  ConstantProperty,
  type CesiumWidget,
  type Entity,
  HorizontalOrigin,
  NearFarScalar,
  PolylineDashMaterialProperty,
  type Scene,
  VerticalOrigin,
} from "cesium";

import type { LonLat } from "@twin/geo";

import type { RaisedHand, Rehearsal, RehearsalTrack } from "@/missions/rehearsal";
import { positionAt, REHEARSAL_SPEEDS, type RehearsalSpeed } from "@/missions/rehearsalClock";

import type { MissionManager } from "./MissionManager";

/** Every entity the rehearsal draws. */
export const REHEARSAL_ENTITY_PREFIX = "rehearsal:";
/** A raised hand: a click opens it in the plan card. */
export const REHEARSAL_HAND_PREFIX = `${REHEARSAL_ENTITY_PREFIX}hand:`;
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
  if (id.startsWith(REHEARSAL_GHOST_PREFIX))
    return { kind: "machine", machineId: id.slice(REHEARSAL_GHOST_PREFIX.length) };
  return id.startsWith(REHEARSAL_ENTITY_PREFIX) ? "through" : null;
}

const PENDING = Color.fromCssColorString("#c9d1cc").withAlpha(0.75);
const HAND = Color.fromCssColorString("#f0b254");
const WATER = Color.fromCssColorString("#6ec3f0").withAlpha(0.9);
const INK = "#0a0e09";
/**
 * How many raised hands carry their label at once when none is focused: the latest only, since
 * the hands of one lane often stand close together and their labels would overlap.
 */
const LABELLED_HANDS = 1;
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

/** A raised hand: Lucide's hand on an amber disc. */
const HAND_IMAGE = svgUri(
  `<svg xmlns="http://www.w3.org/2000/svg" width="34" height="34" viewBox="0 0 34 34"><circle cx="17" cy="17" r="15" fill="#f0b254" stroke="${INK}" stroke-width="2"/><g transform="translate(6 6) scale(0.92)" fill="none" stroke="${INK}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 11V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2"/><path d="M14 10V4a2 2 0 0 0-2-2a2 2 0 0 0-2 2v2"/><path d="M10 10.5V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2v8"/><path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/></g></svg>`,
);

function laneId(zoneId: string, index: number): string {
  return `${zoneId}:${index}`;
}

/**
 * Plays a dress rehearsal (`missions/rehearsal.ts`) on the map: ghost machines on the plan's
 * drawn lanes, the lanes going from pending (dashed grey) to done (the machine's colour, or
 * amber where a hand went up), and a hand at each lane's worst point once a ghost reaches it.
 * It has its own animation-frame loop, which asks the scene for a frame only while playing
 * (the scene renders on request), at 300 or 600 simulated seconds per second. The plan's own
 * passes are hidden while it is loaded and come back on `stop`.
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
  private readonly hands = new Map<string, Entity>();
  private readonly labelShown = new Map<string, boolean>();
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
    for (const hand of rehearsal.hands) {
      this.hands.set(
        hand.id,
        this.viewer.entities.add({
          id: `${REHEARSAL_HAND_PREFIX}${hand.id}`,
          name: `Raised hand: ${hand.label}`,
          show: false,
          position: this.standing(hand.at, 2),
          billboard: {
            image: HAND_IMAGE,
            width: 28,
            height: 28,
            // Many hands on a steep ranch: far ones shrink, the near ones stay readable.
            scaleByDistance: new NearFarScalar(400, 1, 6000, 0.55),
            verticalOrigin: VerticalOrigin.BOTTOM,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
          label: {
            text: hand.label,
            show: false,
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
        }),
      );
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

  /** Puts one hand's label up (null: none besides the latest) and flies to it. */
  focus(handId: string | null, fly = true): void {
    this.focused = handId;
    this.sync();
    this.scene.requestRender();
    const hand = this.rehearsal?.hands.find((h) => h.id === handId);
    if (hand && fly) this.mission.flyToPoint(hand.at, 160);
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
    this.listeners.clear();
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
    const raised = rehearsal.hands.filter((h) => this.t >= h.t);
    // The focused hand speaks alone; otherwise the latest to go up.
    const labelled = new Set(
      this.focused ? [this.focused] : raised.slice(-LABELLED_HANDS).map((h) => h.id),
    );
    const up = new Set(raised.map((h) => h.id));
    for (const hand of rehearsal.hands) this.showHand(hand, up.has(hand.id), labelled);
  }

  private showHand(hand: RaisedHand, up: boolean, labelled: ReadonlySet<string>): void {
    const entity = this.hands.get(hand.id);
    if (!entity) return;
    const shown = up || hand.id === this.focused;
    if (entity.show !== shown) entity.show = shown;
    const label = shown && labelled.has(hand.id);
    if (entity.label && this.labelShown.get(hand.id) !== label) {
      entity.label.show = new ConstantProperty(label);
      this.labelShown.set(hand.id, label);
    }
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
      ...this.hands.values(),
      ...this.ghosts,
    ])
      this.viewer.entities.remove(entity);
    this.pending.clear();
    this.done.clear();
    this.hands.clear();
    this.labelShown.clear();
    this.ghosts.length = 0;
    this.lastRaised = -1;
  }
}
