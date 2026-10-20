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
 * own primitive (hidden or highlighted objects, skins, split objects) do not act while it is a
 * companion: its primitive draws nothing then.
 */

import type { Epoch, Representation, SiteAsset } from "@twin/contracts";
import { Matrix4, type Cesium3DTileset, type Scene } from "cesium";

import { cellGridMeta, loadCellGrid, type CellGrid } from "@/lib/cellGrid";
import {
  leafTileBoxes,
  NO_OWNER,
  ownerAgeGrid,
  ownerPlane,
  type CompositorSource,
  type OwnerPlane,
  type TileBox,
  type TilesetNode,
  type ViewRule,
  type ViewRuleEnd,
} from "@/lib/compositor";
import { ageBucket, laterDates, loadLaterGrids } from "@/lib/dataAge";
import { epochOf, epochOptions, frameAnchor, shownOption } from "@/lib/epochs";
import { createLogger } from "@/lib/log";
import { useSettings } from "@/state/settings";
import { useViewMode } from "@/state/viewMode";
import { useViewRule, type LeftOut, type ViewRuleLegend } from "@/state/viewRule";

import { companionGroups, setCompanions, type CompanionHost } from "./companions";
import { cesiumCellViewGpu, SplatCellView, type CellViewGpu } from "./splatCellView";
import type { Mat4 } from "./splatFrames";
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
 * draws them all, then each date up to the one chosen that copies its placement, oldest first.
 * Null when a rule cannot be drawn: not splats, under another renderer than CesiumJS, fewer
 * than two dates.
 */
export function ruleAssets(
  site: CompositorSite,
  representation: Representation,
  cesium: boolean,
): SiteAsset[] | null {
  const options = epochOptions(site.epochs, site.site.assets, representation);
  const anchor = frameAnchor(site.epochs, site.site.assets, representation)?.asset;
  if (!cesium || !anchor || options.length < 2 || representation !== "gaussian-splat") return null;
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
  /** Whether CesiumJS draws splat scans now (a rule is CesiumJS's alone). */
  cesium(): boolean;
  /** Ends the rule, saying why. */
  end(reason: ViewRuleEnd): void;
}

export interface CompositorHandle {
  /** What the site draws under the rule now (`ruleAssets`), or null when it cannot be drawn. */
  assets(representation: Representation): SiteAsset[] | null;
  dispose(): void;
  /** The shader parts on the host now, for harnesses and tests. */
  compositor(): SplatCompositor | undefined;
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
  /** The loaded dates drawn, the host first; none until the host is in. */
  const datesNow = (): CompositorDate[] => {
    const dates: CompositorDate[] = [];
    for (const asset of ruleAssets(site, site.representation, context.cesium()) ?? []) {
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
  if (!gpu) {
    context.end("unsupported");
    return { assets: () => null, dispose: () => undefined, compositor: () => undefined };
  }
  // A date's own Confidence or Data age reads its own grid, not the other dates'.
  useViewMode.getState().set("photo");

  let disposed = false;
  let host: Cesium3DTileset | undefined;
  let primitive: HostPrimitive | undefined;
  let compositor: SplatCompositor | undefined;
  let ageView: SplatCellView | undefined;
  /** The host's view cones, switched off while the rule is on, and how they were. */
  let cones: { part: { enabled: boolean }; was: boolean } | undefined;
  const borrowed = new Set<HostPrimitive>();
  const listening = new Map<Cesium3DTileset, () => void>();
  const coverage = new Map<Cesium3DTileset, Coverage>();
  let coverageVersion = 0;
  let sourcesVersion = 0;
  let planeKey = "";
  let ageKey = "";
  let later: { key: string; grids: CompositorSource[] } = { key: "", grids: [] };
  let inferred = new Set<unknown>();
  let legendKey = "";

  const changed = (): void => {
    coverageVersion += 1;
    scene.requestRender();
  };

  /** What `tileset` covers, read once: its grid, else its leaf tiles' boxes. */
  const coverageOf = (tileset: Cesium3DTileset, isInferred: boolean): Coverage => {
    const known = coverage.get(tileset);
    if (known) return known;
    const entry: Coverage = { done: false };
    coverage.set(tileset, entry);
    const url = urlOf(tileset);
    const extras = (tileset.root as { extras?: unknown } | undefined)?.extras;
    const meta = isInferred ? null : cellGridMeta(extras, "confidence");
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
      // The boxes are in the tileset's root frame; the plane is in the host's.
      const frame =
        host && host !== tileset
          ? Matrix4.toArray(
              Matrix4.multiply(
                Matrix4.inverseTransformation(rootTransform(host), new Matrix4()),
                rootTransform(tileset),
                new Matrix4(),
              ),
            )
          : undefined;
      const json = (await (options.fetchJson
        ? options.fetchJson(url)
        : resourceJson(tileset, url))) as { root?: TilesetNode };
      entry.boxes = leafTileBoxes(json.root, frame);
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
    if (!asOf || !host) return [];
    const { later: dates } = laterDates(
      { epochs: site.epochs, assets: site.site.assets },
      asOf.assetId,
      "gaussian-splat",
    );
    const key = dates.map((d) => d.asset.id).join(",");
    if (key !== later.key) {
      const mine: typeof later = { key, grids: [] };
      later = mine;
      void loadLaterGrids(dates, Matrix4.toArray(rootTransform(host))).then((grids) => {
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

  /** Lets go of the host: its companions, the parts, the dates it drew, the cones. */
  const release = (): void => {
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
    coverage.clear();
    planeKey = "";
    ageKey = "";
  };

  const sync = (): void => {
    if (disposed) return;
    const dates = datesNow();
    const first = dates[0];
    if (!first) return;
    for (const { tileset } of dates) {
      // The date's primitive appears with its first tile: told before it uploads anything.
      if (!listening.has(tileset)) listening.set(tileset, tileset.tileLoad.addEventListener(sync));
    }
    const own = primitiveOf(first.tileset);
    const rigged = dates.some(
      (d) => (d.tileset as unknown as { splatIncremental?: boolean }).splatIncremental === false,
    );
    if (rigged || own?.incremental === false) {
      context.end("living");
      return;
    }
    if (!own) return;
    if (host !== first.tileset || primitive !== own) {
      release();
      host = first.tileset;
      primitive = own;
      const root = first.tileset;
      compositor = new SplatCompositor(
        gpu,
        () => (root.root as { computedTransform?: Mat4 } | undefined)?.computedTransform,
      );
    }
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
    // Every source the host draws: the dates, then each date's inferred layers.
    const epochOf = new Map<unknown, Epoch>(dates.map((d) => [d.tileset, d.epoch]));
    const entries: {
      tileset: Cesium3DTileset;
      epoch: Epoch;
      inferred: boolean;
      from?: unknown;
    }[] = dates.map((d) => ({ ...d, inferred: false }));
    for (const { owner, from, tilesets } of companionGroups(target)) {
      const epoch = owner === DATES ? undefined : epochOf.get(from);
      if (epoch)
        for (const tileset of tilesets) entries.push({ tileset, epoch, inferred: true, from });
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
    const drawn = entries.slice(0, MAX_SOURCES);
    const nextInferred = new Set<unknown>(drawn.filter((e) => e.inferred).map((e) => e.tileset));
    if ([...nextInferred].some((t) => !inferred.has(t)) || nextInferred.size !== inferred.size)
      inferred = nextInferred;
    const pending = new Set<number>();
    const sources: CompositorSource[] = drawn.map((e, source) => {
      const c = coverageOf(e.tileset, e.inferred);
      if (!c.done) pending.add(source);
      return {
        source,
        epochId: e.epoch.id,
        observedAt: e.epoch.observedAt,
        inferred: e.inferred,
        grid: c.grid ?? null,
        boxes: c.boxes ?? null,
      };
    });
    const rule = site.rule?.rule ?? "latest";
    const asOf = asOfNow();
    const asOfIso = asOf?.epoch.observedAt ?? null;
    const style = useSettings.getState().inferredStyle;
    const key = [
      rule === "ground-truth",
      asOfIso,
      sourcesVersion,
      coverageVersion,
      style !== "hide",
    ].join("|");
    let plane: OwnerPlane | undefined = parts.plane;
    if (key !== planeKey || !plane) {
      planeKey = key;
      plane = ownerPlane(rule, sources, asOfIso, { allowInferred: style !== "hide" });
      parts.setPlane(plane);
      scene.requestRender();
    }
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
        const { grid, rg } = ownerAgeGrid(plane, [...sources, ...newer], asOfIso, now());
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
    publishLegend(plane, sources, pending, entries.length - drawn.length);
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
        ageBucket: ageBucket(s.observedAt, now()),
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
  scene.requestRender();
  return {
    assets: (representation) => ruleAssets(site, representation, context.cesium()),
    compositor: () => compositor,
    dispose: () => {
      if (disposed) return;
      disposed = true;
      offUpdate();
      offStyle();
      for (const off of listening.values()) off();
      listening.clear();
      release();
      useViewRule.setState({ legend: null });
      scene.requestRender();
    },
  };
}
