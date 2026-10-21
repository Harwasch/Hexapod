/**
 * A per-cell view rule on the globe (SiteManager `setViewRule`; ADR 0017 "Per-cell view
 * rules"), loaded on demand the first time a rule is chosen.
 *
 * **One primitive draws every date.** The frame anchor's splat primitive (the host, the
 * date that made the site's frame) draws the other dates' tiles as companions
 * (`companions.ts`), in its one texture and one sort, and each date's own primitive draws
 * nothing (`drawnBy`). Their inferred layers follow their scan there (`inferredLayers.ts`).
 * Two dates drawn as two primitives would be sorted apart, one painted over the other wherever
 * both reach; one sort is what lets a cell show one date and its neighbour another with a
 * soft seam between.
 *
 * **Which date each cell shows** is `lib/compositor.ts`'s: each source's coverage (its
 * confidence grid, else its leaf tiles' boxes, fetched once) is read into an owner plane for
 * the rule and "As of" date, and the compositor's shader parts (`splatCompositor.ts`) draw the
 * owner's splats alone. Data age colours each cell by its owner's age through W4a's colour
 * part (`splatCellView.ts`), on a one-level grid of the plane; Highlight invented tints the
 * inferred owners.
 *
 * **What it turns off, and says.** The per-date analytic views go to Photo (a date's
 * Confidence reads its own grid, not the others'). The host's view cones are switched off
 * while the rule is on: the anchor's cones would fade the other dates' splats where the anchor
 * never looked. A Living Survey rig on any drawn date takes the scan out of incremental mode,
 * which drawing companions needs, so the rule ends and says why. Per-scan hooks on a date's
 * own primitive (hidden or highlighted objects, skins) do not act while it is a companion: its
 * primitive draws nothing then. Split objects are tilesets of their own beside their scan, out
 * of the host's sort: each is drawn whole by the date owning the cell under its declared origin
 * at its pose (`ruledObjects`, `objectAnchor`), in its own colours (Data age does not tint it).
 *
 * **Under PlayCanvas** (the app's default renderer) the same plane is handed to the renderer
 * drawing the site's scan (scanView/scanRule.ts), which draws every date in its one sort: the
 * scan's own tiles, the other dates streamed beside them (scanView/ruleDates.ts), their
 * inferred layers and split objects, each tile weighed by the plane as it is copied into the
 * renderer's work buffer, with the same seam fade. The plane is then worked out in the frame of
 * the scan PlayCanvas draws (SiteManager `scanTarget`, the date chosen, which the rule is As
 * of): every date's tiles are placed in it. Nothing is put on CesiumJS's primitives, which draw nothing there. A renderer
 * change carries the rule over, CesiumJS to PlayCanvas and back; under Spark or the WebGPU
 * trial a rule cannot be drawn (`ruleAssets`), and SiteManager ends it, saying so.
 *
 * **As captured** (state/asCaptured.ts) draws the rule chosen as Ground truth, with no inferred
 * owner, while it is on; the rule chosen stays the viewer's, and comes back when it is off.
 */

import type { Epoch, Representation, SiteAsset } from "@twin/contracts";
import { Matrix4, type Cesium3DTileset, type Scene } from "cesium";

import { cellGridMeta, cellTextureRows, loadCellGrid, type CellGrid } from "@/lib/cellGrid";
import type { CellLookView } from "@/lib/cellLook";
import {
  leafTileBoxes,
  NO_OWNER,
  objectAnchor,
  ownerAgeGrid,
  ownerAt,
  ownerPlane,
  type CompositorSource,
  type OwnerPlane,
  type TileBox,
  type TilesetNode,
  type ViewRule,
  type ViewRuleEnd,
} from "@/lib/compositor";
import { ageBucket, laterDates, loadLaterGrids } from "@/lib/dataAge";
import {
  epochOf,
  epochOptions,
  frameAnchor,
  observedAtFromUpload,
  shownOption,
} from "@/lib/epochs";
// Already reached through splatCompositor.ts: importing it moves nothing (ruleKit.ts).
import { resolveLayerUrl } from "@/lib/inferred";
import { createLogger } from "@/lib/log";
import { perCellDrawable } from "@/lib/viewModes";
import { effectiveInferredStyle, useAsCaptured } from "@/state/asCaptured";
import { useSettings } from "@/state/settings";
import { useViewMode } from "@/state/viewMode";
import { useViewRule, type LeftOut, type ViewRuleLegend } from "@/state/viewRule";

import { companionGroups, ruledObjects, setCompanions, type CompanionHost } from "./companions";
import type { RuleKit } from "./scanView/ruleKit";
import {
  lendRuleKit,
  namedKit,
  publishScanRule,
  ruleOwnerRows,
  type RuleDate,
  type RuleSource,
  type ScanRuleView,
} from "./scanView/scanRule";
import type { SplatRendererKind } from "./scanView/types";
import { cesiumCellViewGpu, SplatCellView, type CellViewGpu } from "./splatCellView";
import { invertAffine, type Mat4 } from "./splatFrames";
import { SEAM_REACH } from "@/lib/compositor";
import {
  cesiumCompositorGpu,
  SplatCompositor,
  type CompositorGpu,
  type CompositorPrimitive,
} from "./splatCompositor";
import { splatTilesetOf, type SplatPrimitive } from "./splatInternals";
import { MAX_SOURCES } from "./splatSources";
import { visibilityChainOf } from "./splatVisibility";

const log = createLogger("compositor");

/** A date the rule draws: its tileset, and the epoch it is of. */
export interface CompositorDate {
  tileset: Cesium3DTileset;
  epoch: Epoch;
}

/** The loaded site, as far as the rule reads it (SiteManager's own record of it). */
export interface CompositorSite {
  rule: { rule: ViewRule } | null;
  representation: Representation;
  epochs: readonly Epoch[];
  /** The date chosen (the capture-date control's): "As of" it. Null for the latest. */
  epochId: string | null;
  site: { assets: readonly SiteAsset[] };
  /**
   * Each loaded model, the asset whose placement it copies (`followAnchor`), if any, and
   * whether that is still being decided.
   */
  handles: ReadonlyMap<
    string,
    {
      tileset: Cesium3DTileset | null;
      follows?: string | undefined;
      followPending?: boolean | undefined;
    }
  >;
}

/**
 * What a site draws under a rule (SiteManager `drawnAssets`): the frame anchor, whose primitive
 * draws them all under CesiumJS, then each date up to the one chosen that copies its placement,
 * oldest first. Null when a rule cannot be drawn: not splats, under a renderer that draws no
 * rule (Spark, the WebGPU trial: lib/viewModes.ts `perCellDrawable`), fewer than two dates.
 */
export function ruleAssets(
  site: CompositorSite,
  representation: Representation,
  renderer: SplatRendererKind,
): SiteAsset[] | null {
  const options = epochOptions(site.epochs, site.site.assets, representation);
  const anchor = frameAnchor(site.epochs, site.site.assets, representation)?.asset;
  if (
    !perCellDrawable(renderer) ||
    !anchor ||
    options.length < 2 ||
    representation !== "gaussian-splat"
  )
    return null;
  const until = Date.parse(shownOption(options, site.epochId)?.epoch.observedAt ?? "");
  const dates = options.filter(
    (o) =>
      o.asset.id !== anchor.id &&
      Date.parse(o.epoch.observedAt) <= until &&
      site.handles.get(o.asset.id)?.follows === anchor.id,
  );
  return [anchor, ...dates.map((o) => o.asset)];
}

/**
 * The dates up to "As of" a rule cannot draw (beside the frame anchor), and why: `loading` while
 * the date's model loads or its placement is being compared with the anchor's, `frame` once it
 * is placed on its own (its tiles are not in the site frame). `asOf` marks the date chosen.
 */
export function leftOutDates(site: CompositorSite, representation: Representation): LeftOut[] {
  const options = epochOptions(site.epochs, site.site.assets, representation);
  const anchor = frameAnchor(site.epochs, site.site.assets, representation)?.asset;
  const chosen = shownOption(options, site.epochId);
  const until = Date.parse(chosen?.epoch.observedAt ?? "");
  const out: LeftOut[] = [];
  for (const o of options) {
    if (!anchor || o.asset.id === anchor.id || Date.parse(o.epoch.observedAt) > until) continue;
    const handle = site.handles.get(o.asset.id);
    if (handle?.follows === anchor.id) continue;
    const loading = !handle?.tileset || handle.followPending === true;
    out.push({
      epochId: o.epoch.id,
      reason: loading ? "loading" : "frame",
      asOf: o.epoch.id === chosen?.epoch.id,
    });
  }
  return out;
}

/** What the session reads of the scene, every frame. */
export interface CompositorContext {
  scene: Pick<Scene, "preUpdate" | "requestRender">;
  entry: CompositorSite;
  /** Who draws splat scans now: CesiumJS and PlayCanvas draw a rule (`perCellDrawable`). */
  renderer(): SplatRendererKind;
  /**
   * What the rule borrows from the globe's first load to draw under PlayCanvas (ruleKit.ts:
   * lent, not imported, so that nothing moves out of the first load's entry).
   */
  kit: RuleKit;
  /** Ends the rule, saying why. */
  end(reason: ViewRuleEnd): void;
}

export interface CompositorHandle {
  /** What the site draws under the rule now (`ruleAssets`), or null when it cannot be drawn. */
  assets(representation: Representation): SiteAsset[] | null;
  dispose(): void;
  /** The shader parts on the host now, for harnesses and tests. */
  compositor(): SplatCompositor | undefined;
  /**
   * For harnesses measuring the seam: the reach (`SEAM_REACH` when undefined; 0, centres
   * only), and the rule off (`enabled` false) to draw every date at once, under either renderer.
   */
  tune(options: { seamReach?: number | undefined; enabled?: boolean }): void;
}

export interface CompositorOptions {
  gpu?: CompositorGpu | undefined;
  cellGpu?: CellViewGpu | undefined;
  /** Reads a tileset's own tileset.json; by default through its Resource (ion's headers too). */
  fetchJson?: (url: string) => Promise<unknown>;
  loadGrid?: typeof loadCellGrid;
  now?: () => number;
}

type HostPrimitive = CompositorPrimitive &
  CompanionHost &
  Pick<SplatPrimitive, "incremental" | "drawnBy">;

/** What a source's coverage is read from. */
interface CoverageInput {
  /** Its tileset's URL, where its grid and its tileset.json are. */
  url: string | undefined;
  extras: unknown;
  inferred: boolean;
  /** Its tileset.json. */
  json(url: string): Promise<unknown>;
  /** Its root's frame to the plane's (column-major), from its tileset.json; none: the same. */
  frame(json: unknown): readonly number[] | undefined;
}

/** A tileset the rule draws, as both renderers see it. */
interface RuleEntry {
  /** What its coverage is kept by: its tileset, or (an inferred layer under PlayCanvas) its URL. */
  key: unknown;
  epoch: Epoch;
  inferred: boolean;
  /** The date an inferred layer travels with. */
  from?: unknown;
  /** Its tileset's URL as the renderer names it (PlayCanvas). */
  url?: string;
  cover: CoverageInput;
}

/** One source's coverage, once read (`done`). */
interface Coverage {
  grid?: CellGrid | null;
  boxes?: TileBox[] | null;
  done: boolean;
}

/** This session's place in the host's companion list. */
const DATES = Symbol("view-rule dates");

function primitiveOf(tileset: Cesium3DTileset): HostPrimitive | undefined {
  return splatTilesetOf(tileset).gaussianSplatPrimitive;
}

function urlOf(tileset: Cesium3DTileset): string | undefined {
  return (tileset as unknown as { resource?: { url?: string } }).resource?.url;
}

/** The root tile's transform as tileset.json declares it (the identity when it declares none). */
function rootTransform(tileset: Cesium3DTileset): Matrix4 {
  return (tileset.root as { transform?: Matrix4 } | undefined)?.transform ?? Matrix4.IDENTITY;
}

/** The root tile's transform on the globe now (its placement's model matrix included). */
function computedTransform(tileset: Cesium3DTileset): number[] {
  const m = (tileset.root as { computedTransform?: Mat4 } | undefined)?.computedTransform;
  return Array.from({ length: 16 }, (_, i) => m?.[i] ?? (i % 5 === 0 ? 1 : 0));
}

/** A tileset's URL as a dedicated renderer fetches it (ScanRendererHost `run`): absolute. */
function absoluteUrl(tileset: Cesium3DTileset): string | undefined {
  const url = urlOf(tileset);
  return url === undefined ? undefined : new URL(url, location.href).toString();
}

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** `a · b` (column-major 4x4). */
function times(a: readonly number[], b: readonly number[]): number[] {
  return Matrix4.toArray(
    Matrix4.multiply(Matrix4.fromArray([...a]), Matrix4.fromArray([...b]), new Matrix4()),
  );
}

/** `a⁻¹` (column-major 4x4, any affine: a runtime scale too). */
function inverse(a: readonly number[]): number[] {
  return Matrix4.toArray(Matrix4.inverse(Matrix4.fromArray([...a]), new Matrix4()));
}

/** `a⁻¹ · b`, or null when it is the identity (to a micrometre): where `b`'s frame is in `a`'s. */
function relative(a: readonly number[], b: readonly number[]): number[] | null {
  const m = times(inverse(a), b);
  return m.every((v, i) => Math.abs(v - (IDENTITY[i] ?? 0)) < 1e-6) ? null : m;
}

/** A tileset.json's `root.transform` (column-major), or the identity. */
function declaredRoot(json: unknown): number[] {
  const t = (json as { root?: { transform?: unknown } } | null)?.root?.transform;
  return Array.isArray(t) && t.length === 16 && t.every((v) => Number.isFinite(v))
    ? (t as number[])
    : IDENTITY;
}

/** A tileset.json by plain fetch (an inferred layer under PlayCanvas has no Resource here). */
async function fetchJsonOf(url: string): Promise<unknown> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`answered ${String(response.status)}`);
  return response.json();
}

/** A tileset's own tileset.json, through its Resource where it has one (ion's headers too). */
async function resourceJson(tileset: Cesium3DTileset, url: string): Promise<unknown> {
  const resource = (tileset as unknown as { resource?: { fetchJson?: () => Promise<unknown> } })
    .resource;
  if (typeof resource?.fetchJson === "function") return resource.fetchJson();
  const response = await fetch(url);
  if (!response.ok) throw new Error(`answered ${String(response.status)}`);
  return response.json();
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function startCompositor(
  context: CompositorContext,
  options: CompositorOptions = {},
): CompositorHandle {
  const gpu = "gpu" in options ? options.gpu : cesiumCompositorGpu();
  const cellGpu = "cellGpu" in options ? options.cellGpu : cesiumCellViewGpu();
  const loadGrid = options.loadGrid ?? loadCellGrid;
  const now = options.now ?? Date.now;
  const { scene, entry: site } = context;
  /** The loaded dates drawn, the frame anchor first; none until the anchor is in. */
  const datesNow = (): CompositorDate[] => {
    const dates: CompositorDate[] = [];
    for (const asset of ruleAssets(site, site.representation, context.renderer()) ?? []) {
      const tileset = site.handles.get(asset.id)?.tileset;
      const epoch = epochOf(asset, site.epochs);
      if (tileset && !tileset.isDestroyed() && epoch) dates.push({ tileset, epoch });
      else if (dates.length === 0) return [];
    }
    return dates;
  };
  /** "As of": the date chosen, else the latest, and its model. */
  const asOfNow = (): { epoch: Epoch; assetId: string } | null => {
    const option = shownOption(
      epochOptions(site.epochs, site.site.assets, "gaussian-splat"),
      site.epochId,
    );
    return option && { epoch: option.epoch, assetId: option.asset.id };
  };
  if (!gpu && context.renderer() === "cesium") {
    context.end("unsupported");
    return {
      assets: () => null,
      dispose: () => undefined,
      compositor: () => undefined,
      tune: () => undefined,
    };
  }
  // A date's own Confidence or Data age reads its own grid, not the other dates'.
  useViewMode.getState().set("photo");
  // PlayCanvas streams the other dates with the globe's own classes, lent (ruleKit.ts).
  lendRuleKit(context.kit);
  const kit = namedKit(context.kit);

  let disposed = false;
  /** Which renderer the rule is drawn by now; null before the first frame. */
  let mode: "cesium" | "playcanvas" | null = null;
  /** The tileset whose root frame the plane is in: CesiumJS's host, or PlayCanvas's scan. */
  let frameOf: Cesium3DTileset | undefined;
  let host: Cesium3DTileset | undefined;
  let primitive: HostPrimitive | undefined;
  let compositor: SplatCompositor | undefined;
  let ageView: SplatCellView | undefined;
  /** The host's view cones, switched off while the rule is on, and how they were. */
  let cones: { part: { enabled: boolean }; was: boolean } | undefined;
  const borrowed = new Set<HostPrimitive>();
  /** The dates drawn, by source index, whose split objects the rule draws (`ruledObjects`). */
  let ruling: Cesium3DTileset[] = [];
  const listening = new Map<Cesium3DTileset, () => void>();
  const coverage = new Map<unknown, Coverage>();
  let coverageVersion = 0;
  let sourcesVersion = 0;
  let sourcesKey = "";
  let planeKey = "";
  let plane: OwnerPlane | undefined;
  let ageKey = "";
  let later: { key: string; grids: CompositorSource[] } = { key: "", grids: [] };
  let inferred = new Set<unknown>();
  let legendKey = "";
  /** What PlayCanvas was last handed, and for which asset (`publishScanRule`). */
  let published: { assetId: string; key: string; age: CellLookView | null } | null = null;
  const tuning: { seamReach: number | undefined; enabled: boolean } = {
    seamReach: undefined,
    enabled: true,
  };

  const changed = (): void => {
    coverageVersion += 1;
    scene.requestRender();
  };

  /** What a source covers, read once: its grid, else its leaf tiles' boxes. */
  const coverageOf = (key: unknown, input: CoverageInput): Coverage => {
    const known = coverage.get(key);
    if (known) return known;
    const entry: Coverage = { done: false };
    coverage.set(key, entry);
    const { url } = input;
    const meta = input.inferred ? null : cellGridMeta(input.extras, "confidence");
    const read = async (): Promise<void> => {
      if (!url) return;
      if (meta) {
        try {
          entry.grid = await loadGrid(url, meta);
          return;
        } catch (error) {
          log.warn("a date's confidence grid did not load; its tile bounds stand in", {
            message: message(error),
          });
        }
      }
      // The boxes are in the tileset's root frame; the plane is in the frame's.
      const json = (await input.json(url)) as { root?: TilesetNode };
      entry.boxes = leafTileBoxes(json.root, input.frame(json));
    };
    void read()
      .catch((error: unknown) =>
        log.warn("a date's coverage could not be read; it owns no cell", {
          message: message(error),
        }),
      )
      .finally(() => {
        entry.done = true;
        if (!disposed) changed();
      });
    return entry;
  };

  /** Data age's "newer data here": the dates after "As of", whose grids are fetched once. */
  const laterSources = (asOf: { assetId: string } | null): CompositorSource[] => {
    if (!asOf || !frameOf) return [];
    const { later: dates } = laterDates(
      { epochs: site.epochs, assets: site.site.assets },
      asOf.assetId,
      "gaussian-splat",
    );
    const key = dates.map((d) => d.asset.id).join(",");
    if (key !== later.key) {
      const mine: typeof later = { key, grids: [] };
      later = mine;
      void loadLaterGrids(dates, Matrix4.toArray(rootTransform(frameOf))).then((grids) => {
        if (disposed || later !== mine) return;
        mine.grids = grids.flatMap((g) =>
          "grid" in g
            ? [
                {
                  source: NO_OWNER,
                  epochId: g.epoch.id,
                  observedAt: g.epoch.observedAt,
                  inferred: false,
                  grid: g.grid,
                },
              ]
            : [],
        );
        changed();
      });
    }
    return later.grids;
  };

  /**
   * Whether a split object of `date` is drawn under CesiumJS: whole, when that date owns the
   * plane's cell under the object's centre -- one date's copy of an object, never two.
   */
  const ownsObject =
    (date: Cesium3DTileset) =>
    (object: Cesium3DTileset): boolean => {
      const plane = compositor?.plane;
      const root = (host?.root as { computedTransform?: Mat4 } | undefined)?.computedTransform;
      const m = root && invertAffine(root);
      const own = (object.root as { computedTransform?: Mat4 } | undefined)?.computedTransform;
      if (!plane || !m || !own) return false;
      // Its declared origin at its pose, in the host's frame (`objectAnchor`).
      const [x, y] = objectAnchor(
        times(
          m,
          Array.from({ length: 16 }, (_, i) => own[i] ?? 0),
        ),
      );
      return ownerAt(plane, x, y) === ruling.indexOf(date);
    };

  /** Lets go of CesiumJS's host: its companions, the parts, the dates it drew, the cones. */
  const releaseCesium = (): void => {
    for (const tileset of ruling) ruledObjects.delete(tileset);
    ruling = [];
    if (primitive) setCompanions(primitive, DATES, []);
    for (const own of borrowed) if (own.drawnBy === primitive) own.drawnBy = undefined;
    borrowed.clear();
    compositor?.destroy();
    compositor = undefined;
    ageView?.destroy();
    ageView = undefined;
    if (cones) cones.part.enabled = cones.was;
    cones = undefined;
    primitive = undefined;
    host = undefined;
  };

  /** Takes the rule back from the renderer it was handed to (PlayCanvas). */
  const unpublish = (): void => {
    if (published) publishScanRule(published.assetId, null);
    published = null;
  };

  /** The plane is to be read in another frame (another host, another scan, another renderer). */
  const reframe = (tileset: Cesium3DTileset | undefined): void => {
    if (frameOf === tileset) return;
    frameOf = tileset;
    coverage.clear();
    planeKey = "";
    ageKey = "";
    plane = undefined;
  };

  /** The rule's choices now: which rule, as of when, whether inferred layers are drawn. */
  const choices = () => {
    const rule: ViewRule = useAsCaptured.getState().on
      ? "ground-truth"
      : (site.rule?.rule ?? "latest");
    const asOf = asOfNow();
    const style = effectiveInferredStyle();
    return { rule, asOf, asOfIso: asOf?.epoch.observedAt ?? null, style };
  };

  /**
   * The plane of `entries` (the first `MAX_SOURCES`, each its source index), worked out again
   * when the sources, their coverage, the rule's kind or "As of" changed; and what each source
   * is, for the legend.
   */
  const planeFor = (entries: readonly RuleEntry[]) => {
    const drawn = entries.slice(0, MAX_SOURCES);
    const key = drawn.map((e) => `${String(e.inferred)}:${e.url ?? ""}`).join(",");
    if (key !== sourcesKey) {
      sourcesKey = key;
      sourcesVersion += 1;
    }
    const pending = new Set<number>();
    const sources: CompositorSource[] = drawn.map((e, source) => {
      const c = coverageOf(e.key, e.cover);
      if (!c.done) pending.add(source);
      return {
        source,
        epochId: e.epoch.id,
        observedAt: e.epoch.observedAt,
        inferred: e.inferred,
        // Dated only by its upload: it fills where no dated date saw, and has no age.
        dated: !observedAtFromUpload(e.epoch),
        grid: c.grid ?? null,
        boxes: c.boxes ?? null,
      };
    });
    const { rule, asOf, asOfIso, style } = choices();
    const next = [
      rule === "ground-truth",
      asOfIso,
      sourcesVersion,
      coverageVersion,
      style !== "hide",
    ].join("|");
    let fresh = false;
    if (next !== planeKey || !plane) {
      planeKey = next;
      plane = ownerPlane(rule, sources, asOfIso, { allowInferred: style !== "hide" });
      fresh = true;
      scene.requestRender();
    }
    return { plane, fresh, drawn, sources, pending, rule, asOf, asOfIso, style };
  };

  /** The rule as CesiumJS draws it: every date on the frame anchor's primitive. */
  const syncCesium = (dates: readonly CompositorDate[]): void => {
    const first = dates[0];
    if (!first || !gpu) {
      if (!gpu) context.end("unsupported");
      return;
    }
    const own = primitiveOf(first.tileset);
    if (own?.incremental === false) {
      context.end("living");
      return;
    }
    if (!own) return;
    if (host !== first.tileset || primitive !== own) {
      releaseCesium();
      host = first.tileset;
      primitive = own;
      const root = first.tileset;
      compositor = new SplatCompositor(
        gpu,
        () => (root.root as { computedTransform?: Mat4 } | undefined)?.computedTransform,
      );
      compositor.enabled = tuning.enabled;
      compositor.seamReach = tuning.seamReach;
    }
    reframe(first.tileset);
    const target = own;
    const parts = compositor;
    if (!parts) return;
    // The other dates draw through the host, in its sort.
    const others = dates.slice(1).map((d) => d.tileset);
    for (const tileset of others) {
      const theirs = primitiveOf(tileset);
      if (theirs && theirs.drawnBy !== target) {
        theirs.drawnBy = target;
        borrowed.add(theirs);
      }
    }
    for (const theirs of borrowed) {
      if (others.some((t) => primitiveOf(t) === theirs)) continue;
      if (theirs.drawnBy === target) theirs.drawnBy = undefined;
      borrowed.delete(theirs);
    }
    setCompanions(target, DATES, others);
    // Each date's split objects: drawn by the date owning the cell under each.
    const next = dates.map((d) => d.tileset);
    for (const tileset of ruling) if (!next.includes(tileset)) ruledObjects.delete(tileset);
    for (const tileset of next)
      if (!ruling.includes(tileset)) ruledObjects.set(tileset, ownsObject(tileset));
    ruling = next;
    // Every source the host draws: the dates, then each date's inferred layers.
    const epochOf = new Map<unknown, Epoch>(dates.map((d) => [d.tileset, d.epoch]));
    const json = (tileset: Cesium3DTileset) => (url: string) =>
      options.fetchJson ? options.fetchJson(url) : resourceJson(tileset, url);
    // The boxes are in the tileset's root frame; the plane is in the host's.
    const frame = (tileset: Cesium3DTileset) => (): number[] | undefined =>
      host && host !== tileset
        ? Matrix4.toArray(
            Matrix4.multiply(
              Matrix4.inverseTransformation(rootTransform(host), new Matrix4()),
              rootTransform(tileset),
              new Matrix4(),
            ),
          )
        : undefined;
    const input = (tileset: Cesium3DTileset, isInferred: boolean): CoverageInput => ({
      url: urlOf(tileset),
      extras: (tileset.root as { extras?: unknown } | undefined)?.extras,
      inferred: isInferred,
      json: json(tileset),
      frame: frame(tileset),
    });
    const entries: (RuleEntry & { tileset: Cesium3DTileset })[] = dates.map((d) => ({
      key: d.tileset,
      tileset: d.tileset,
      epoch: d.epoch,
      inferred: false,
      cover: input(d.tileset, false),
    }));
    for (const { owner, from, tilesets } of companionGroups(target)) {
      const epoch = owner === DATES ? undefined : epochOf.get(from);
      if (epoch)
        for (const tileset of tilesets)
          entries.push({
            key: tileset,
            tileset,
            epoch,
            inferred: true,
            from,
            cover: input(tileset, true),
          });
    }
    if (!parts.installed && !parts.install(target)) {
      context.end("unsupported");
      return;
    }
    if (!cones) {
      const part = visibilityChainOf(target)?.parts.find(
        (p) => p.visibilityFunction === "splatViewConeVisibility",
      ) as { enabled?: unknown } | undefined;
      if (part && typeof part.enabled === "boolean") {
        cones = { part: part as { enabled: boolean }, was: part.enabled };
        part.enabled = false;
      }
    }
    if (parts.sources.setSources(entries.map((e) => e.tileset))) sourcesVersion += 1;
    const found = planeFor(entries);
    const { drawn, sources, pending, rule, asOf, asOfIso, style } = found;
    const nextInferred = new Set<unknown>(drawn.filter((e) => e.inferred).map((e) => e.key));
    if ([...nextInferred].some((t) => !inferred.has(t)) || nextInferred.size !== inferred.size)
      inferred = nextInferred;
    if (found.fresh || parts.plane !== found.plane) parts.setPlane(found.plane);
    // Inferred owners tinted: under Highlight invented, and for the viewer's own Highlight
    // on the layers of the dates the host draws -- the host's own layers carry their scan's
    // highlight (inferredLayers.ts), which is not drawn twice.
    const highlight = style === "highlight";
    let tinted = 0;
    drawn.forEach((e, i) => {
      const own = e.from === first.tileset;
      if (
        e.inferred &&
        ((rule === "highlight-invented" && !(highlight && own)) || (highlight && !own))
      )
        tinted |= 1 << i;
    });
    if (tinted !== parts.tinted) {
      parts.tinted = tinted;
      scene.requestRender();
    }
    // Data age: each cell by its owner's age, striped where a date after "As of" saw it.
    if (rule === "data-age" && cellGpu) {
      const newer = laterSources(asOf);
      const nextAge = `${planeKey}|${later.key}|${String(newer.length)}|${String(Math.floor(now() / 86_400_000))}`;
      if (nextAge !== ageKey || !ageView) {
        ageKey = nextAge;
        ageView?.destroy();
        const { grid, rg } = ownerAgeGrid(found.plane, [...sources, ...newer], asOfIso, now());
        const root = first.tileset;
        ageView = new SplatCellView(
          grid,
          cellGpu,
          () => (root.root as { computedTransform?: Mat4 } | undefined)?.computedTransform,
          () => inferred,
        );
        ageView.setDataAge(rg);
        ageView.mode = "data-age";
        scene.requestRender();
      }
      if (!ageView.installed && ageView.install(target)) scene.requestRender();
    } else if (ageView) {
      ageView.destroy();
      ageView = undefined;
      ageKey = "";
      scene.requestRender();
    }
    if (parts.sync()) scene.requestRender();
    publishLegend(found.plane, sources, pending, entries.length - drawn.length);
  };

  /**
   * The rule as PlayCanvas draws it: the plane, in the frame of the scan PlayCanvas draws, and
   * by URL what each tileset is -- every date, each one's inferred layers and split objects --
   * handed to that renderer (scanView/scanRule.ts), which streams the other dates beside the
   * scan and weighs every tile it draws by the plane.
   */
  const syncPlaycanvas = (dates: readonly CompositorDate[]): void => {
    // The scan PlayCanvas draws: the date chosen (SiteManager `scanTarget`; a model asked for
    // by itself, which it would draw instead, ends the rule).
    const chosen = asOfNow();
    const shown = chosen ? site.handles.get(chosen.assetId)?.tileset : null;
    const target = chosen && shown ? { assetId: chosen.assetId, tileset: shown } : null;
    if (!target || target.tileset.isDestroyed()) {
      unpublish();
      return;
    }
    if (published && published.assetId !== target.assetId) unpublish();
    reframe(target.tileset);
    // Each date where it is in the scan's frame (`P`), and the root its layers are placed from.
    const toScan = invertAffine(computedTransform(target.tileset));
    const placed = dates.map((d) => {
      const placement = toScan
        ? relative(computedTransform(target.tileset), computedTransform(d.tileset))
        : null;
      const own = Matrix4.toArray(rootTransform(d.tileset));
      const root = placement ? times(own, inverse(placement)) : own;
      return {
        ...d,
        url: absoluteUrl(d.tileset),
        assetId: [...site.handles].find(([, h]) => h.tileset === d.tileset)?.[0],
        extras: (d.tileset.root as { extras?: unknown } | undefined)?.extras,
        placement,
        root,
      };
    });
    const json = (url: string) => (options.fetchJson ?? fetchJsonOf)(url);
    const entries: RuleEntry[] = placed.map((d) => ({
      key: d.tileset,
      epoch: d.epoch,
      inferred: false,
      ...(d.url ? { url: d.url } : {}),
      cover: {
        url: urlOf(d.tileset),
        extras: d.extras,
        inferred: false,
        json: (url) => (options.fetchJson ? options.fetchJson(url) : resourceJson(d.tileset, url)),
        frame: () => d.placement ?? undefined,
      },
    }));
    // Each date's inferred layers, by the URL the renderer streams them from (scanLayers.ts).
    for (const d of placed) {
      if (!d.url || d.assetId === undefined) continue;
      const base = d.url;
      for (const ref of kit.pickedLayers(d.assetId, d.extras).refs) {
        const url = resolveLayerUrl(base, ref.uri);
        entries.push({
          key: url,
          epoch: d.epoch,
          inferred: true,
          from: d.tileset,
          url,
          cover: {
            url,
            extras: undefined,
            inferred: true,
            json,
            frame: (layer) => relative(d.root, declaredRoot(layer)) ?? undefined,
          },
        });
      }
    }
    const found = planeFor(entries);
    const { drawn, sources, pending, rule, asOf, asOfIso } = found;
    const bySource = new Map<string, RuleSource>();
    drawn.forEach((e, source) => {
      if (e.url) bySource.set(e.url, { source });
    });
    // Each date's split objects, drawn whole by the date owning the cell under each.
    placed.forEach((d, source) => {
      if (!d.url || source >= drawn.length) return;
      for (const ref of kit.splitObjectsOf(d.extras))
        bySource.set(resolveLayerUrl(d.url, ref.uri), { source, whole: true });
    });
    const tinted =
      rule === "highlight-invented"
        ? drawn.reduce((mask, e, i) => (e.inferred ? mask | (1 << i) : mask), 0)
        : 0;
    // Data age: each cell by its owner's age, striped where a date after "As of" saw it.
    let age: CellLookView | null = null;
    if (rule === "data-age") {
      const newer = laterSources(asOf);
      const nextAge = `${planeKey}|${later.key}|${String(newer.length)}|${String(Math.floor(now() / 86_400_000))}`;
      if (nextAge !== ageKey || !published?.age) {
        ageKey = nextAge;
        const { grid, rg } = ownerAgeGrid(found.plane, [...sources, ...newer], asOfIso, now());
        age = { mode: "data-age", rows: cellTextureRows(grid.meta, rg), meta: grid.meta };
      } else age = published.age;
    } else ageKey = "";
    const others: RuleDate[] = placed.flatMap((d, source) =>
      d.tileset !== target.tileset && d.url && d.assetId !== undefined && source < drawn.length
        ? [
            {
              url: d.url,
              assetId: d.assetId,
              extras: d.extras,
              placement: d.placement,
              root: d.root,
            },
          ]
        : [],
    );
    const native = placed.findIndex((d) => d.tileset === target.tileset);
    const reach = tuning.seamReach ?? SEAM_REACH;
    const key = JSON.stringify([
      planeKey,
      [...bySource],
      tinted,
      ageKey,
      reach,
      tuning.enabled,
      others.map((d) => [d.url, d.placement, d.root]),
    ]);
    if (published?.key !== key || published.assetId !== target.assetId) {
      const view: ScanRuleView = {
        plane: [found.plane.i0, found.plane.j0, found.plane.ni, found.plane.nj],
        cellM: found.plane.cellM,
        owners: ruleOwnerRows(found.plane),
        sources: bySource,
        native: native >= 0 && native < drawn.length ? native : NO_OWNER,
        tinted,
        reach,
        enabled: tuning.enabled,
        dates: others,
        age,
      };
      published = { assetId: target.assetId, key, age };
      publishScanRule(target.assetId, view);
      scene.requestRender();
    }
    publishLegend(found.plane, sources, pending, entries.length - drawn.length);
  };

  const sync = (): void => {
    if (disposed) return;
    const dates = datesNow();
    if (!dates[0]) return;
    for (const { tileset } of dates) {
      // The date's primitive appears with its first tile: told before it uploads anything.
      if (!listening.has(tileset)) listening.set(tileset, tileset.tileLoad.addEventListener(sync));
    }
    const rigged = dates.some(
      (d) => (d.tileset as unknown as { splatIncremental?: boolean }).splatIncremental === false,
    );
    if (rigged) {
      context.end("living");
      return;
    }
    const next = context.renderer() === "cesium" ? "cesium" : "playcanvas";
    if (next !== mode) {
      // Carried over to the other renderer: what the last one was handed is taken back.
      if (mode === "cesium") releaseCesium();
      if (mode === "playcanvas") unpublish();
      mode = next;
      reframe(undefined);
    }
    if (mode === "cesium") syncCesium(dates);
    else syncPlaycanvas(dates);
  };

  const publishLegend = (
    plane: OwnerPlane,
    sources: readonly CompositorSource[],
    pending: ReadonlySet<number>,
    left: number,
  ): void => {
    const byId = new Map<string, ViewRuleLegend["dates"][number]>();
    let inferredCells = 0;
    for (const s of sources) {
      const taken = plane.sources.find((p) => p.source === s.source);
      if (s.inferred) {
        inferredCells += taken?.cells ?? 0;
        continue;
      }
      if (!taken) continue;
      byId.set(s.epochId, {
        epochId: s.epochId,
        coverage: pending.has(s.source) ? "loading" : taken.coverage,
        ageBucket: s.dated === false ? null : ageBucket(s.observedAt, now()),
        cells: taken.cells,
      });
    }
    const dates = [...byId.values()];
    const legend: ViewRuleLegend = {
      cellM: plane.cellM,
      dates,
      inferredCells,
      left,
      leftOut: leftOutDates(site, site.representation),
    };
    const next = JSON.stringify(legend);
    if (next === legendKey) return;
    legendKey = next;
    useViewRule.setState({ legend });
  };

  const offUpdate = scene.preUpdate.addEventListener(sync);
  const offStyle = useSettings.subscribe((s, previous) => {
    if (s.inferredStyle !== previous.inferredStyle) scene.requestRender();
  });
  const offAsCaptured = useAsCaptured.subscribe(() => scene.requestRender());
  scene.requestRender();
  return {
    assets: (representation) => ruleAssets(site, representation, context.renderer()),
    compositor: () => compositor,
    tune: ({ seamReach, enabled }) => {
      tuning.seamReach = seamReach;
      if (enabled !== undefined) tuning.enabled = enabled;
      if (compositor) {
        compositor.seamReach = seamReach;
        compositor.enabled = tuning.enabled;
      }
      scene.requestRender();
    },
    dispose: () => {
      if (disposed) return;
      disposed = true;
      offUpdate();
      offStyle();
      offAsCaptured();
      for (const off of listening.values()) off();
      listening.clear();
      releaseCesium();
      unpublish();
      useViewRule.setState({ legend: null });
      scene.requestRender();
    },
  };
}
