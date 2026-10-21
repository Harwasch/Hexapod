/**
 * The other dates of a site a per-cell view rule draws (scanRule.ts `dates`), streamed by the
 * renderer that draws the site's scan (playcanvasBackend.ts) into its one app, beside the scan's
 * own tiles: one sort for every date, which is what lets one cell show one date and the next
 * cell another with a clean seam between (ADR 0017). Loaded on demand with the first rule that
 * names another date.
 *
 * **Streaming.** Each date is a tileset of its own: its tiles are chosen from the view as the
 * scan's are (view/stream.ts), within the budget the renderer draws to now (`budget`, the
 * adaptive budget the host hands it: frames that slow with two dates lower it for both), so a
 * date's resident splats are about the scan's -- two dates hold about twice what one does, as
 * under CesiumJS, where each date's tileset refines on its own. The view is the renderer's own
 * pose (`poseView`): the host plans the scan's cut from Cesium's camera, and this the dates'
 * from the same camera as the renderer draws it.
 *
 * **What travels with a date.** Its inferred layers (scanLayers.ts) and split objects
 * (scanObjects.ts) are streamed with it, placed from its root (`RuleDate.root`), so each is
 * where the date's tiles are. A date's tile is the renderer's `loadDate`: drawn as a scan tile
 * (the cell views, the sun) but bound to none of the scan's objects, motion or picks -- the
 * per-scan hooks of a date the rule draws do not act on it, as under CesiumJS.
 *
 * **Frames.** A date's tiles go under its placement `P` in the scan's frame; the view is taken
 * into the date's frame by `P⁻¹` (scanLayers.ts `layerView`).
 */

import { createLogger } from "@/lib/log";
import { TileStreamer, type View } from "@/view/stream";
import { parseTileset, type Sphere, type TileNode } from "@/view/tiles";

import { LAYER_BUDGET_SHARE, layerView, ScanLayers, type LayerBackend } from "./scanLayers";
import { ScanObjects } from "./scanObjects";
import type { RuleDate } from "./scanRule";
import type { ScanBackend, ScanPose } from "./types";

const log = createLogger("rule-dates");

/** How often a date's cut is planned again while the camera moves (ms), as the scan's. */
export const DATE_REPLAN_MS = 150;
/** Tiles of one date fetched at once. */
const DATE_FETCHES_AT_ONCE = 2;
/** Loaded tiles a date keeps beyond what it draws, as the scan's cache. */
const DATE_CACHE_FACTOR = 1.5;

/** What of a back-end the dates use: a layer's, and a date's own tile. */
export type DatesBackend<M> = LayerBackend<M> &
  Pick<ScanBackend<M>, "add" | "remove" | "dispose" | "place"> & {
    /** Fetches and decodes a tile of another date: a scan tile bound to none of its objects. */
    loadDate(tilesetUrl: string, tile: TileNode, signal?: AbortSignal): Promise<M>;
  };

export interface RuleDatesOptions {
  /** A tile came (or one is due again): the renderer draws, and plans again in that frame. */
  arrived(): void;
  /** What a frame shows changed without a tile coming: the renderer draws. */
  changed(): void;
  /** Gaussians each date may stream now. */
  budget(): number;
  fetchJson?: (url: string) => Promise<unknown>;
}

/** What the dates report (scanRule.ts `ScanRuleReading.dates`). */
export interface RuleDatesStatus {
  dates: number;
  tiles: number;
  gaussians: number;
  loading: number;
}

type Vec3 = [number, number, number];

function sub(a: readonly number[], b: readonly number[]): Vec3 {
  return [(a[0] ?? 0) - (b[0] ?? 0), (a[1] ?? 0) - (b[1] ?? 0), (a[2] ?? 0) - (b[2] ?? 0)];
}
function dot(a: readonly number[], b: readonly number[]): number {
  return (a[0] ?? 0) * (b[0] ?? 0) + (a[1] ?? 0) * (b[1] ?? 0) + (a[2] ?? 0) * (b[2] ?? 0);
}
function cross(a: readonly number[], b: readonly number[]): Vec3 {
  const [ax = 0, ay = 0, az = 0] = a;
  const [bx = 0, by = 0, bz = 0] = b;
  return [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx];
}
function unit(v: readonly number[]): Vec3 {
  const length = Math.hypot(v[0] ?? 0, v[1] ?? 0, v[2] ?? 0) || 1;
  return [(v[0] ?? 0) / length, (v[1] ?? 0) / length, (v[2] ?? 0) / length];
}

/**
 * The view a renderer draws from `pose` (the scan's frame), as view/stream.ts plans for: the
 * eye, the pixels a metre spans a metre away, the frustum's test of a tile's sphere (its four
 * sides and its far plane; behind the eye is out), and the middle of the view.
 */
export function poseView(pose: ScanPose): View {
  const forward = unit(pose.direction);
  const right = unit(cross(forward, pose.up));
  const up = cross(right, forward);
  const tanV = Math.tan(pose.fovy / 2);
  const aspect = pose.width / Math.max(1, pose.height);
  const tanH = tanV * aspect;
  const slackV = Math.sqrt(1 + tanV * tanV);
  const slackH = Math.sqrt(1 + tanH * tanH);
  return {
    eye: [...pose.eye],
    projection: pose.height / (2 * tanV),
    centre: { forward, halfDiagonal: Math.atan(tanV * Math.hypot(1, aspect)) },
    visible: ({ center, radius }: Sphere) => {
      const d = sub(center, pose.eye);
      const z = dot(d, forward);
      if (z < -radius || z - radius > pose.far) return false;
      return (
        Math.abs(dot(d, right)) - z * tanH <= radius * slackH &&
        Math.abs(dot(d, up)) - z * tanV <= radius * slackV
      );
    },
  };
}

interface DrawnDate<M> {
  date: RuleDate;
  key: string;
  /** Null until its tileset has been read. */
  streamer: TileStreamer<M> | null;
  layers: ScanLayers<M>;
  objects: ScanObjects<M>;
  onScreen: Set<M>;
  stopped: boolean;
}

/** The rule's other dates in one renderer: follows the list, plans each from the pose. */
export class RuleDates<M> {
  #dates: DrawnDate<M>[] = [];
  #arrived = true;
  #lastPlan = Number.NEGATIVE_INFINITY;
  #lastEye: number[] | null = null;
  #lastDirection: number[] | null = null;
  #dueBy: number | null = null;
  #stopped = false;

  constructor(
    private readonly backend: DatesBackend<M>,
    private readonly options: RuleDatesOptions,
  ) {}

  /** Draws these dates (none: `[]`), keeping each one already drawn as it was. */
  setDates(dates: readonly RuleDate[]): void {
    if (this.#stopped) return;
    const keyOf = (d: RuleDate) => JSON.stringify([d.url, d.assetId, d.placement, d.root]);
    const next = new Map(dates.map((d) => [keyOf(d), d]));
    for (const drawn of this.#dates.filter((d) => !next.has(d.key))) this.#drop(drawn);
    this.#dates = this.#dates.filter((d) => next.has(d.key));
    for (const [key, date] of next) {
      if (this.#dates.some((d) => d.key === key)) continue;
      this.#dates.push(this.#start(key, date));
    }
    this.#arrived = true;
    this.options.changed();
  }

  /**
   * Plans every date's tiles for the renderer's `pose`, at most every `DATE_REPLAN_MS` and only
   * when the eye moved or a tile came; places the dates' split objects at their poses.
   */
  plan(pose: ScanPose, now: number = performance.now()): void {
    if (this.#stopped) return;
    let moved = false;
    for (const date of this.#dates) moved = date.objects.tick() || moved;
    if (moved) this.options.changed();
    const turned =
      this.#lastEye === null ||
      this.#lastDirection === null ||
      sub(pose.eye, this.#lastEye).some((v) => Math.abs(v) > 1e-3) ||
      sub(pose.direction, this.#lastDirection).some((v) => Math.abs(v) > 1e-5);
    if (!turned && !this.#arrived) {
      this.#dueBy = null;
      return;
    }
    if (now - this.#lastPlan < DATE_REPLAN_MS) {
      this.#dueBy = this.#lastPlan + DATE_REPLAN_MS;
      return;
    }
    this.#dueBy = null;
    this.#lastPlan = now;
    this.#arrived = false;
    this.#lastEye = [...pose.eye];
    this.#lastDirection = [...pose.direction];
    const view = poseView(pose);
    const budget = Math.max(1, this.options.budget());
    for (const date of this.#dates) {
      date.streamer?.setBudget(budget, budget * DATE_CACHE_FACTOR);
      date.streamer?.update(layerView(view, date.date.placement));
      date.layers.update(view);
    }
  }

  /** When a plan held back by `DATE_REPLAN_MS` is due (`performance.now()` ms), or null. */
  dueBy(): number | null {
    return this.#dueBy;
  }

  status(): RuleDatesStatus {
    let tiles = 0;
    let gaussians = 0;
    let loading = 0;
    for (const date of this.#dates) {
      const layers = date.layers.status();
      tiles += date.onScreen.size + layers.tiles;
      gaussians += (date.streamer?.drawnGaussians ?? 0) + layers.gaussians;
      loading += (date.streamer?.loading ?? 0) + layers.loading;
    }
    return { dates: this.#dates.length, tiles, gaussians, loading };
  }

  stop(): void {
    this.#stopped = true;
    for (const date of this.#dates.splice(0)) this.#drop(date);
  }

  #start(key: string, date: RuleDate): DrawnDate<M> {
    const { backend, options } = this;
    const drawn: DrawnDate<M> = {
      date,
      key,
      streamer: null,
      layers: new ScanLayers<M>(backend, date.assetId, date.url, date.extras, date.root, {
        arrived: () => this.#arrive(),
        changed: () => options.changed(),
        budget: () => options.budget() * LAYER_BUDGET_SHARE,
        ...(options.fetchJson ? { fetchJson: options.fetchJson } : {}),
      }),
      objects: new ScanObjects<M>(
        {
          load: (url, tile, signal) => backend.loadDate(url, tile, signal),
          add: (mesh) => backend.add(mesh),
          remove: (mesh) => backend.remove(mesh),
          dispose: (mesh) => backend.dispose(mesh),
          ...(backend.place ? { place: backend.place.bind(backend) } : {}),
        },
        date.assetId,
        () => options.changed(),
      ),
      onScreen: new Set(),
      stopped: false,
    };
    const fetchJson =
      options.fetchJson ??
      (async (url: string): Promise<unknown> => {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`answered ${String(response.status)}`);
        return (await response.json()) as unknown;
      });
    void drawn.objects.load(date.url, date.extras, date.root, fetchJson);
    void fetchJson(date.url)
      .then((json) => {
        if (drawn.stopped) return;
        const budget = Math.max(1, options.budget());
        const streamer = new TileStreamer<M>(
          parseTileset(json),
          {
            load: (tile, signal) => backend.loadDate(date.url, tile, signal),
            show: (_tile, mesh) => {
              drawn.onScreen.add(mesh);
              if (date.placement) backend.place?.(mesh, date.placement);
              backend.add(mesh);
              options.changed();
            },
            hide: (_tile, mesh) => {
              drawn.onScreen.delete(mesh);
              backend.remove(mesh);
              options.changed();
            },
            dispose: (mesh) => backend.dispose(mesh),
            failed: (tile, error) =>
              log.warn("a date's tile did not load; its parent stays", {
                asset: date.assetId,
                tile: tile.uri,
                error: error instanceof Error ? error.message : String(error),
              }),
          },
          {
            budget,
            cacheBudget: budget * DATE_CACHE_FACTOR,
            concurrency: DATE_FETCHES_AT_ONCE,
          },
        );
        streamer.onArrival = () => this.#arrive();
        drawn.streamer = streamer;
        this.#arrive();
      })
      .catch((error: unknown) =>
        log.warn("a date the rule draws did not load; its cells draw nothing", {
          asset: date.assetId,
          error: error instanceof Error ? error.message : String(error),
        }),
      );
    return drawn;
  }

  #arrive(): void {
    this.#arrived = true;
    this.options.arrived();
  }

  #drop(date: DrawnDate<M>): void {
    date.stopped = true;
    date.layers.stop();
    date.objects.stop();
    const streamer = date.streamer;
    date.streamer = null;
    // Hides what it shows (`hide`), then disposes every tile it holds.
    streamer?.stop();
    for (const mesh of date.onScreen) this.backend.remove(mesh);
    date.onScreen.clear();
  }
}
