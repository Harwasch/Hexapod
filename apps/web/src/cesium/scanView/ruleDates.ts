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
 *
 * The streamer, the layers' and objects' classes and the parser are the globe's own, lent at
 * runtime (`kit`, ruleKit.ts) rather than imported: imported, the bundler would move them out
 * of the globe's first-load entry, and the first load would grow.
 */

import { createLogger } from "@/lib/log";
import type { TileStreamer, View } from "@/view/stream";
import type { Sphere, TileNode } from "@/view/tiles";

import type { LayerBackend, ScanLayers } from "./scanLayers";
import type { ScanObjects } from "./scanObjects";

import type { NamedRuleKit, RuleDate, ScanRuleView } from "./scanRule";
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
  /** Its tileset.json, once read: what a streamer is made from again after an idle spell. */
  json: unknown;
  /** Null until its tileset has been read, and while it holds no share of the budget. */
  streamer: TileStreamer<M> | null;
  layers: ScanLayers<M>;
  objects: ScanObjects<M>;
  onScreen: Set<M>;
  stopped: boolean;
}

/** What of a rule the dates are planned from (scanRule.ts `ScanRuleView`). */
export type RuleDatesPlan = Pick<
  ScanRuleView,
  "dates" | "plane" | "cellM" | "owners" | "scanOwns"
> & {
  /**
   * Shares fixed rather than worked out from the plane: two dates compared split the one budget
   * evenly (playcanvasBackend.ts `COMPARE_SHARES`).
   */
  shares?: { readonly scan: number; readonly dates: readonly number[] };
};

/** Cells sampled at most for the shares, the plane thinned evenly past it. */
export const SHARE_SAMPLES = 16_384;

/**
 * Each holder's share of the cells in view that it owns -- the scan the renderer draws
 * (`scan`: `plan.scanOwns`), and each date (its own source and its inferred layers'). A cell
 * is in view when a sphere about its middle, on the plane's ground (z = 0), wide enough for the
 * cells it stands for, meets the frustum (`visible`, or every cell without one). Null when no
 * holder owns a cell in view: the shares stand as they were.
 */
export function viewShares(
  plan: RuleDatesPlan,
  visible: ((sphere: Sphere) => boolean) | null,
): { scan: number; dates: number[] } | null {
  const [i0, j0, ni, nj] = plan.plane;
  const holderOf = new Map<number, number>();
  for (const source of plan.scanOwns) holderOf.set(source, 0);
  plan.dates.forEach((date, k) => {
    for (const source of date.owns) holderOf.set(source, k + 1);
  });
  const counts = new Array<number>(plan.dates.length + 1).fill(0);
  const step = Math.max(1, Math.ceil(Math.sqrt((ni * nj) / SHARE_SAMPLES)));
  const radius = plan.cellM * step * 0.75;
  let total = 0;
  for (let cj = Math.floor(step / 2); cj < nj; cj += step)
    for (let ci = Math.floor(step / 2); ci < ni; ci += step) {
      // A cell nobody owns (or past the bytes) is no holder's.
      const holder = holderOf.get(plan.owners.data[cj * ni + ci] ?? -1);
      if (holder === undefined) continue;
      const center: [number, number, number] = [
        (i0 + ci + 0.5) * plan.cellM,
        (j0 + cj + 0.5) * plan.cellM,
        0,
      ];
      if (visible && !visible({ center, radius })) continue;
      counts[holder] = (counts[holder] ?? 0) + 1;
      total += 1;
    }
  if (total === 0) return null;
  return { scan: (counts[0] ?? 0) / total, dates: counts.slice(1).map((c) => c / total) };
}

/**
 * The rule's other dates in one renderer: follows the rule, plans each from the pose, and
 * splits one budget between every date it draws -- the scan the renderer draws included --
 * by each one's share of the cells in view it owns (`viewShares`), so all of them together
 * hold what one scan alone would. A date that owns nothing in view streams nothing: its tiles
 * are let go and nothing of it is fetched until it owns something in view again. The scan's
 * own share reaches the renderer's host through `scanShare` (its streamer is the host's).
 */
export class RuleDates<M> {
  #dates: DrawnDate<M>[] = [];
  #plan: RuleDatesPlan | null = null;
  /** The scan's share of the budget, and each date's by key. */
  #scanShare = 1;
  readonly #shares = new Map<string, number>();
  #arrived = true;
  #lastPlan = Number.NEGATIVE_INFINITY;
  #lastEye: number[] | null = null;
  #lastDirection: number[] | null = null;
  #lastBudget = Number.NaN;
  #dueBy: number | null = null;
  #stopped = false;

  constructor(
    private readonly backend: DatesBackend<M>,
    private readonly options: RuleDatesOptions,
    private readonly kit: NamedRuleKit,
  ) {}

  /**
   * Draws the rule's dates (none: `[]`), keeping each one already drawn as it was, and takes
   * its plane: the shares are worked out again at the next plan.
   */
  setRule(plan: RuleDatesPlan): void {
    if (this.#stopped) return;
    this.#plan = plan;
    const keyOf = (d: RuleDate) => JSON.stringify([d.url, d.assetId, d.placement, d.root]);
    const next = new Map(plan.dates.map((d) => [keyOf(d), d]));
    for (const drawn of this.#dates.filter((d) => !next.has(d.key))) this.#drop(drawn);
    this.#dates = this.#dates.filter((d) => next.has(d.key));
    for (const [key, date] of next) {
      const drawn = this.#dates.find((d) => d.key === key);
      if (drawn) drawn.date = date;
      else this.#dates.push(this.#start(key, date));
    }
    // Before any view, by the cells each owns at all.
    if (this.#lastEye === null) this.#reshare(null);
    this.#arrived = true;
    this.options.changed();
  }

  /** The share of the budget the scan the renderer draws takes now (1 without other dates). */
  scanShare(): number {
    return this.#dates.length === 0 ? 1 : this.#scanShare;
  }

  /**
   * Plans every date's tiles for the renderer's `pose`, at most every `DATE_REPLAN_MS`, when
   * the eye moved, a tile came, or the budget moved (the camera still: a host whose frames
   * slowed has every date hold less); places the dates' split objects at their poses.
   */
  plan(pose: ScanPose, now: number = performance.now()): void {
    if (this.#stopped) return;
    let moved = false;
    for (const date of this.#dates) moved = date.objects.tick() || moved;
    if (moved) this.options.changed();
    const budget = Math.max(1, this.options.budget());
    const turned =
      this.#lastEye === null ||
      this.#lastDirection === null ||
      sub(pose.eye, this.#lastEye).some((v) => Math.abs(v) > 1e-3) ||
      sub(pose.direction, this.#lastDirection).some((v) => Math.abs(v) > 1e-5);
    if (!turned && !this.#arrived && budget === this.#lastBudget) {
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
    this.#lastBudget = budget;
    this.#lastEye = [...pose.eye];
    this.#lastDirection = [...pose.direction];
    const view = poseView(pose);
    this.#reshare((sphere) => view.visible(sphere));
    for (const date of this.#dates) {
      const share = this.#shares.get(date.key) ?? 0;
      if (share <= 0) {
        this.#idle(date);
        continue;
      }
      if (!date.streamer && date.json !== undefined) this.#stream(date);
      date.streamer?.setBudget(share * budget, share * budget * DATE_CACHE_FACTOR);
      date.streamer?.update(this.kit.layerView(view, date.date.placement));
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

  /** Works the shares out again from what is `visible` (every cell with null). */
  #reshare(visible: ((sphere: Sphere) => boolean) | null): void {
    const plan = this.#plan;
    if (!plan) return;
    const found =
      plan.shares ?? viewShares({ ...plan, dates: this.#dates.map((d) => d.date) }, visible);
    if (!found) return;
    this.#dates.forEach((d, k) => this.#shares.set(d.key, found.dates[k] ?? 0));
    if (found.scan !== this.#scanShare) {
      this.#scanShare = found.scan;
      // The host reads the scan's share as its streamer's (`scanShare`): a frame applies it.
      this.options.changed();
    }
  }

  #start(key: string, date: RuleDate): DrawnDate<M> {
    const { backend, options, kit } = this;
    const drawn: DrawnDate<M> = {
      date,
      key,
      json: undefined,
      streamer: null,
      layers: new kit.ScanLayers<M>(backend, date.assetId, date.url, date.extras, date.root, {
        arrived: () => this.#arrive(),
        changed: () => options.changed(),
        // A layer travels with its date: a share of the date's share.
        budget: () => options.budget() * (this.#shares.get(key) ?? 0) * kit.layerBudgetShare,
        ...(options.fetchJson ? { fetchJson: options.fetchJson } : {}),
      }),
      objects: new kit.ScanObjects<M>(
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
        drawn.json = json;
        if ((this.#shares.get(drawn.key) ?? 0) > 0) this.#stream(drawn);
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

  /** Makes the date's streamer from its tileset.json, at its share of the budget. */
  #stream(drawn: DrawnDate<M>): void {
    const { backend, options, kit } = this;
    const { date } = drawn;
    const budget = Math.max(1, options.budget() * (this.#shares.get(drawn.key) ?? 0));
    const streamer = new kit.TileStreamer<M>(
      kit.parseTileset(drawn.json),
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
      { budget, cacheBudget: budget * DATE_CACHE_FACTOR, concurrency: DATE_FETCHES_AT_ONCE },
    );
    streamer.onArrival = () => this.#arrive();
    drawn.streamer = streamer;
  }

  /** A date that owns nothing in view: its tiles let go, nothing fetched. */
  #idle(drawn: DrawnDate<M>): void {
    const streamer = drawn.streamer;
    if (!streamer) return;
    drawn.streamer = null;
    // Hides what it shows (`hide`), then disposes every tile it holds.
    streamer.stop();
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
