/**
 * A scan's inferred layers (lib/inferred.ts) drawn by the dedicated splat renderer that draws
 * the scan (ScanRendererHost), as `cesium/inferredLayers.ts` draws them under CesiumJS: the
 * layers of the fill method picked (state/variants.ts), or Today's `extras.inferredLayers`,
 * shown while Inferred is Show or Highlight (state/settings.ts), in Highlight's look, and faded
 * by their own view cones (layerLook.ts).
 *
 * **One sort.** Splats write no depth in any renderer, so what decides which of two splats is
 * in front is the order they are blended in -- and only one renderer's sort can order both.
 * A layer drawn by CesiumJS under an overlay that draws the scan was painted over by every
 * measured splat behind it (the rebuilt top of a spool see-through to its drum and bottom
 * flange from above); drawn here, each of its tiles is a mesh of the same renderer as the
 * scan's tiles, sorted with them: in front of what it is in front of, behind what it is
 * behind. CesiumJS's own copy is hidden meanwhile (`attachInferredLayers`, `drawn`).
 *
 * **Streaming.** A layer is a tileset of its own, beside the scan's: its tiles are fetched and
 * chosen from the view as the scan's are (view/stream.ts), within a share of the scan's budget
 * (`LAYER_BUDGET_SHARE`) -- a fill is one small tile; an image model's fill of a whole site is
 * a level-of-detail tree. A tile is decoded by the back-end (`loadLayer`) bound to none of the
 * scan's objects, and left out of picking: a click selects the measured scan's objects, as it
 * does under CesiumJS.
 *
 * **Frames.** A layer's root is the scan's root `S` (or another frame, `O`): the renderer draws
 * in the scan's frame, so a layer's tiles go under `P = S⁻¹ · O` (the identity for every layer
 * published so far), and the view it is streamed for is taken into its frame by `P⁻¹`. The
 * scan's model matrix -- its placement on the globe, a runtime scale -- is the overlay's camera
 * pose, so a layer is drawn wherever and at whatever size the scan is.
 *
 * **Supersedes** (lib/supersedes.ts) is not this module's: the measured splats a fill replaces
 * are hidden through the scan's instance ids (state/supersedes.ts, `linkScanInstances`).
 */

import { resolveLayerUrl, type InferredLayerRef, type InferredStyle } from "@/lib/inferred";
import { createLogger } from "@/lib/log";
import { variantsOf } from "@/lib/variants";
import { loadViewCones, viewConesMetaOf, type ViewConesMeta } from "@/lib/viewCones";
import { useSettings } from "@/state/settings";
import { onPickChange } from "@/state/variants";
import { TileStreamer, type View } from "@/view/stream";
import { parseTileset, type Sphere } from "@/view/tiles";

import { pickedLayers } from "../inferredLayers";
import { uniformScale } from "../placement";
import { invertAffine, type Mat4 } from "../splatFrames";
import { viewConesEnabled } from "../splatViewCones";
import { layerCones, type LayerLook } from "./layerLook";
import { multiply4, rootTransformOf } from "./scanObjects";
import type { ScanBackend } from "./types";

const log = createLogger("scan-layers");

/**
 * The share of the scan's streamed budget each layer may hold. A fill of one object is a few
 * thousand gaussians, whole at any budget; a whole site's (an image model's, 0.4M on the Camp
 * scan) is whole at a desktop's budget and coarser on a phone's.
 */
export const LAYER_BUDGET_SHARE = 0.25;
/** Tiles of one layer fetched at once. */
const LAYER_FETCHES_AT_ONCE = 2;

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** What of a back-end layers use. */
export type LayerBackend<M> = Pick<
  ScanBackend<M>,
  "add" | "remove" | "dispose" | "place" | "loadLayer" | "setLayerLook"
>;

export interface ScanLayersOptions {
  /** A layer or a tile of one came (or one is due again): the host plans and draws. */
  arrived(): void;
  /** What a frame shows changed without a tile coming (a style): the host draws. */
  changed(): void;
  /** Gaussians each layer may stream now. */
  budget(): number;
  fetchJson?: (url: string) => Promise<unknown>;
  loadCones?: (url: string, meta: ViewConesMeta) => Promise<Uint8Array>;
}

/** What `ScanRendererStatus.layers` reports. */
export interface ScanLayersStatus {
  /** Layers wanted now, those loaded (their tileset read), and those on screen. */
  wanted: number;
  loaded: number;
  drawn: number;
  /** Tiles on screen now, and their gaussians. */
  tiles: number;
  gaussians: number;
  /** Tilesets and tiles being fetched. */
  loading: number;
  /** Layers that did not load. */
  failed: number;
  style: InferredStyle;
}

/** `S⁻¹ · O`, or null when it is the identity (to a micrometre). */
export function layerPlacement(scanRoot: Mat4, layerRoot: Mat4): number[] | null {
  const placement = multiply4(invertAffine(scanRoot) ?? IDENTITY, layerRoot);
  return placement.every((v, i) => Math.abs(v - (IDENTITY[i] ?? 0)) < 1e-6) ? null : placement;
}

function apply(m: Mat4, p: readonly number[], w = 1): [number, number, number] {
  const [x = 0, y = 0, z = 0] = p;
  return [0, 1, 2].map(
    (r) => (m[r] ?? 0) * x + (m[4 + r] ?? 0) * y + (m[8 + r] ?? 0) * z + (m[12 + r] ?? 0) * w,
  ) as [number, number, number];
}

/** `view` (the scan's frame) as seen from a layer drawn under `placement` (its own frame). */
export function layerView(view: View, placement: Mat4 | null): View {
  if (placement === null) return view;
  const inverse = invertAffine(placement) ?? IDENTITY;
  const scale = uniformScale(placement);
  const centre = view.centre;
  const forward = centre ? apply(inverse, centre.forward, 0) : null;
  const length = forward ? Math.hypot(...forward) || 1 : 1;
  return {
    eye: apply(inverse, view.eye),
    projection: view.projection,
    visible: (bounds: Sphere) =>
      view.visible({ center: apply(placement, bounds.center), radius: bounds.radius * scale }),
    ...(centre && forward
      ? {
          centre: {
            forward: forward.map((v) => v / length) as [number, number, number],
            halfDiagonal: centre.halfDiagonal,
          },
        }
      : {}),
  };
}

interface DrawnLayer<M> {
  ref: InferredLayerRef;
  url: string;
  /** Null until its tileset has been read. */
  streamer: TileStreamer<M> | null;
  /** `P`, or null for the identity. */
  placement: number[] | null;
  look: LayerLook;
  /** Meshes the streamer has on screen (on the back-end's while the layers are visible). */
  onScreen: Set<M>;
  failed: boolean;
}

/**
 * One session's inferred layers: follows the pick and the style, streams each layer's tiles
 * from the view the host hands it (`update`), and hands the back-end their look.
 */
export class ScanLayers<M> {
  #layers: DrawnLayer<M>[] = [];
  #key: string | null = null;
  #refs: InferredLayerRef[] = [];
  /** Whether `#refs` have been asked for (they are, once a style shows them). */
  #requested = false;
  #serial = 0;
  #style: InferredStyle;
  #stopped = false;
  #pending = 0;
  readonly #offs: (() => void)[] = [];

  constructor(
    private readonly backend: LayerBackend<M>,
    private readonly assetId: string,
    private readonly tilesetUrl: string,
    private readonly extras: unknown,
    private readonly scanRoot: Mat4,
    private readonly options: ScanLayersOptions,
  ) {
    this.#style = useSettings.getState().inferredStyle;
    if (!backend.loadLayer || !backend.setLayerLook) return;
    if (variantsOf(extras).fill.length > 0)
      this.#offs.push(onPickChange(assetId, "fill", () => this.#follow()));
    this.#offs.push(
      useSettings.subscribe((state) => {
        if (state.inferredStyle !== this.#style) this.#restyle(state.inferredStyle);
      }),
    );
    this.#follow();
  }

  /** Whether the layers are on screen (Show or Highlight). */
  get visible(): boolean {
    return this.#style !== "hide";
  }

  status(): ScanLayersStatus {
    let tiles = 0;
    let gaussians = 0;
    let loading = this.#pending;
    for (const layer of this.#layers) {
      loading += layer.streamer?.loading ?? 0;
      if (!this.visible) continue;
      tiles += layer.onScreen.size;
      gaussians += layer.streamer?.drawnGaussians ?? 0;
    }
    return {
      wanted: this.visible ? this.#refs.length : 0,
      loaded: this.#layers.filter((l) => l.streamer !== null).length,
      drawn: this.visible ? this.#layers.filter((l) => l.onScreen.size > 0).length : 0,
      tiles,
      gaussians,
      loading,
      failed: this.#layers.filter((l) => l.failed).length,
      style: this.#style,
    };
  }

  /** Plans each layer's tiles for `view` (the scan's frame), as the scan's are planned. */
  update(view: View): void {
    if (this.#stopped || !this.visible) return;
    const budget = Math.max(1, this.options.budget());
    for (const layer of this.#layers) {
      if (!layer.streamer) continue;
      layer.streamer.setBudget(budget, budget * 1.5);
      layer.streamer.update(layerView(view, layer.placement));
    }
  }

  stop(): void {
    this.#stopped = true;
    for (const off of this.#offs.splice(0)) off();
    this.#unload();
  }

  /** Loads what the pick wants, once a style shows it; drops what it no longer wants. */
  #follow(): void {
    if (this.#stopped) return;
    const refs = pickedLayers(this.assetId, this.extras).refs;
    const key = refs.map((ref) => ref.uri).join("\n");
    if (key !== this.#key) {
      this.#unload();
      this.#key = key;
      this.#refs = refs;
      this.#requested = false;
      this.options.changed();
    }
    if (this.visible && !this.#requested) {
      this.#requested = true;
      const serial = ++this.#serial;
      for (const ref of refs) void this.#load(ref, serial);
    }
  }

  #restyle(style: InferredStyle): void {
    const was = this.visible;
    this.#style = style;
    const highlight = style === "highlight";
    for (const layer of this.#layers) {
      if (layer.look.highlight !== highlight) {
        layer.look = { ...layer.look, highlight };
        for (const mesh of layer.onScreen) this.backend.setLayerLook?.(mesh, layer.look);
      }
      if (was !== this.visible) {
        for (const mesh of layer.onScreen) {
          if (this.visible) this.backend.add(mesh);
          else this.backend.remove(mesh);
        }
      }
    }
    this.#follow();
    this.options.changed();
    if (!was && this.visible) this.options.arrived();
  }

  async #load(ref: InferredLayerRef, serial: number): Promise<void> {
    const url = resolveLayerUrl(this.tilesetUrl, ref.uri);
    const layer: DrawnLayer<M> = {
      ref,
      url,
      streamer: null,
      placement: null,
      look: { highlight: this.#style === "highlight", cones: null, fromScan: IDENTITY },
      onScreen: new Set(),
      failed: false,
    };
    this.#layers.push(layer);
    this.#pending += 1;
    try {
      const fetchJson =
        this.options.fetchJson ??
        (async (u: string): Promise<unknown> => {
          const response = await fetch(u);
          if (!response.ok) throw new Error(`answered ${String(response.status)}`);
          return (await response.json()) as unknown;
        });
      const json = await fetchJson(url);
      const tree = parseTileset(json);
      const meta = viewConesEnabled()
        ? viewConesMetaOf((json as { root?: { extras?: unknown } } | null)?.root?.extras)
        : null;
      const texels = meta
        ? await (this.options.loadCones ?? loadViewCones)(url, meta).catch((error: unknown) => {
            log.warn("a layer's view cones did not load; drawn from every side", {
              layer: ref.uri,
              error: error instanceof Error ? error.message : String(error),
            });
            return null;
          })
        : null;
      if (this.#stopped || serial !== this.#serial) return;
      const placement = layerPlacement(this.scanRoot, rootTransformOf(json));
      layer.placement = placement;
      layer.look = {
        ...layer.look,
        highlight: this.#style === "highlight",
        cones: meta && texels ? layerCones(meta, texels) : null,
        fromScan: placement ? (invertAffine(placement) ?? IDENTITY) : IDENTITY,
      };
      const backend = this.backend;
      const loadLayer = backend.loadLayer?.bind(backend);
      if (!loadLayer) return;
      const budget = Math.max(1, this.options.budget());
      layer.streamer = new TileStreamer<M>(
        tree,
        {
          load: (tile, signal) => loadLayer(url, tile, signal),
          show: (_tile, mesh) => {
            layer.onScreen.add(mesh);
            if (layer.placement) backend.place?.(mesh, layer.placement);
            backend.setLayerLook?.(mesh, layer.look);
            if (this.visible) backend.add(mesh);
            this.options.changed();
          },
          hide: (_tile, mesh) => {
            layer.onScreen.delete(mesh);
            if (this.visible) backend.remove(mesh);
            this.options.changed();
          },
          dispose: (mesh) => backend.dispose(mesh),
          failed: (tile, error) =>
            log.warn("an inferred layer's tile did not load", {
              layer: ref.uri,
              tile: tile.uri,
              error: error instanceof Error ? error.message : String(error),
            }),
        },
        { budget, cacheBudget: budget * 1.5, concurrency: LAYER_FETCHES_AT_ONCE },
      );
      layer.streamer.onArrival = () => this.options.arrived();
      this.options.arrived();
    } catch (error) {
      layer.failed = true;
      log.warn("an inferred layer did not load", {
        asset: this.assetId,
        layer: ref.uri,
        error: error instanceof Error ? error.message : String(error),
      });
    } finally {
      this.#pending -= 1;
    }
  }

  #unload(): void {
    this.#serial += 1;
    for (const layer of this.#layers.splice(0)) {
      const streamer = layer.streamer;
      layer.streamer = null;
      streamer?.stop();
      layer.onScreen.clear();
    }
  }
}
