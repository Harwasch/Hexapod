/**
 * Draws a measured splat's inferred layers (lib/inferred.ts) under CesiumJS: each its own
 * tileset, placed where the measured one is, shown only while CesiumJS draws the scan and as
 * the viewer's `inferredStyle` says (Show, Highlight or Hide), and faded by its own view cones
 * -- so a fill made for the outside of a scan is never drawn from the inside.
 *
 * **One sort.** Splats write no depth in any renderer: which of two splats is in front is the
 * order they are blended in, so a layer and the scan it sits on must be sorted as one. A layer
 * drawn as a primitive of its own was sorted apart from the scan, and CesiumJS draws splat
 * primitives back to front by the centres of their bounding volumes: from some sides the scan
 * was painted over the layer (a rebuilt top see-through to the drum under it), from the others
 * the layer over the scan (the top over the drum in front of it). Here the scan's own
 * primitive draws the layer's tiles (the engine patch's `companions`): they take slots in its
 * one texture and are sorted with its splats, and the layer's primitive draws nothing
 * (`drawnBy`). That needs the scan's primitive in incremental mode, as every scan's is; one
 * that is not (a CPU deformer's) has its layers draw themselves, as before.
 *
 * **Other renderers.** While PlayCanvas or Spark draws the scan (its tileset hidden, cesium/
 * scanView), they draw its layers too, in their own sort (scanView/scanLayers.ts), and these
 * stay hidden: `drawn` is whether CesiumJS draws the scan. They are still loaded -- their
 * tileset.json, not their tiles -- for what they say they are (`useInferred`, the legend) and
 * how the pick is doing (`useVariants`).
 *
 * **Highlight** is a part of a colour chain (`splatColor.ts`) switched by a uniform: pulled
 * toward purple, hatched in bands, a little see-through (`INFERRED_HIGHLIGHT`). On the scan's
 * primitive (`CompanionHighlight`) it acts only on the layers' slots -- never on a measured
 * splat; on a layer's own primitive (`InferredHighlight`), on all of it. The overlay draws the
 * same rule (scanView/layerLook.ts), so it reads the same under every renderer. The view cones
 * likewise: on the scan's primitive within the layer's slots (`CompanionViewCones`), on the
 * layer's own otherwise.
 *
 * **Which layers** are the pick's (`state/variants.ts`): Today's `extras.inferredLayers`, or a
 * fill variant's. Picking another unloads the drawn layers and loads its own. A variant that
 * names `supersedes` also hides the measured splats it replaces while it is shown
 * (state/supersedes.ts), whichever renderer draws them.
 */
import { Cartesian4, Cesium3DTileset, Matrix4, type Scene } from "cesium";

import { createLogger } from "@/lib/log";
import {
  evidenceOf,
  INFERRED_HIGHLIGHT,
  inferredHighlightColor,
  resolveLayerUrl,
  type InferredEvidence,
  type InferredLayerRef,
  type InferredStyle,
} from "@/lib/inferred";
import { inferredLayersFor, variantsOf } from "@/lib/variants";
import { useInferred } from "@/state/inferred";
import { useSettings } from "@/state/settings";
import { followSupersedes } from "@/state/supersedes";
import { onPickChange, pickedVariant, useVariants, type VariantStatus } from "@/state/variants";

import {
  addColorPart,
  hasColorPart,
  removeColorPart,
  type ColorPrimitive,
  type SplatColorPart,
} from "./splatColor";
import {
  keepOffscreenSplats,
  splatTilesetOf,
  type SplatPrimitive,
  type SplatShaderBuilder,
} from "./splatInternals";
import { attachLayerViewCones, companionSlots, companionSlotsGlsl } from "./splatViewCones";
import type { VisibilityPrimitive } from "./splatVisibility";

const log = createLogger("inferred");

export type LoadLayer = (url: string, parent: Cesium3DTileset) => Promise<Cesium3DTileset>;

async function loadLayer(url: string, parent: Cesium3DTileset): Promise<Cesium3DTileset> {
  const tileset = await Cesium3DTileset.fromUrl(url, {
    maximumScreenSpaceError: parent.maximumScreenSpaceError,
    show: false,
    enableCollision: false,
  });
  keepOffscreenSplats(tileset);
  tileset.cullRequestsWhileMoving = false;
  return tileset;
}

/**
 * Highlight on a layer's own primitive, as a part of its colour chain: as painted while
 * `u_inferredPattern.w` is 0 (Show), else Highlight. `position` is the splat's, in the draw
 * command's frame (metres): the bands run diagonally across it, so every face of a fill is
 * hatched.
 */
export const INFERRED_COLOR_GLSL = `
uniform vec4 u_inferredTint;
// x: band width (m); y: the darker bands' brightness; z: opacity factor; w: highlight on.
uniform vec4 u_inferredPattern;

vec4 splatInferredColor(uint splatIndex, vec3 position, vec4 color) {
    if (u_inferredPattern.w < 0.5) {
        return color;
    }
    vec3 rgb = mix(color.rgb, u_inferredTint.rgb, u_inferredTint.a);
    float band = fract(dot(position, vec3(0.57735027)) / u_inferredPattern.x);
    rgb *= band < 0.5 ? 1.0 : u_inferredPattern.y;
    return vec4(rgb, color.a * u_inferredPattern.z);
}
`;

/**
 * Highlight on the scan's primitive, for the layers it draws in its sort: the same rule as
 * `INFERRED_COLOR_GLSL`, on the splats in the layers' slots (`u_companionSlots`) only.
 */
export const COMPANION_COLOR_GLSL = `
uniform vec4 u_companionTint;
uniform vec4 u_companionPattern;
uniform ivec4 u_companionSlots[4];
${companionSlotsGlsl("splatCompanionSlot", "u_companionSlots")}
vec4 splatCompanionColor(uint splatIndex, vec3 position, vec4 color) {
    if (u_companionPattern.w < 0.5 || !splatCompanionSlot(splatIndex)) {
        return color;
    }
    vec3 rgb = mix(color.rgb, u_companionTint.rgb, u_companionTint.a);
    float band = fract(dot(position, vec3(0.57735027)) / u_companionPattern.x);
    rgb *= band < 0.5 ? 1.0 : u_companionPattern.y;
    return vec4(rgb, color.a * u_companionPattern.z);
}
`;

/** What `INFERRED_COLOR_GLSL` does to one colour (straight alpha), for tests. */
export const evaluateInferredColor = inferredHighlightColor;

/** The colour part of one layer's own primitive: installed with it, Highlight by a uniform. */
export class InferredHighlight implements SplatColorPart {
  readonly colorFunction = "splatInferredColor";
  readonly colorOrder = 10;
  /** Whether Highlight is on (the uniform's `w`). */
  on = false;
  readonly #tint = new Cartesian4(...INFERRED_HIGHLIGHT.tint);
  readonly #pattern = new Cartesian4(
    INFERRED_HIGHLIGHT.stripeM,
    INFERRED_HIGHLIGHT.stripeDark,
    INFERRED_HIGHLIGHT.opacity,
    0,
  );

  addToShader(shaderBuilder: SplatShaderBuilder, uniformMap: Record<string, () => unknown>): void {
    shaderBuilder.addVertexLines(INFERRED_COLOR_GLSL);
    uniformMap.u_inferredTint = () => this.#tint;
    uniformMap.u_inferredPattern = () => {
      this.#pattern.w = this.on ? 1 : 0;
      return this.#pattern;
    };
  }
}

/**
 * Highlight on the scan's primitive for the layers it draws as companions: a part of its colour
 * chain, after the objects' highlight, acting on the slots of `owners`' tiles alone.
 */
export class CompanionHighlight implements SplatColorPart {
  readonly colorFunction = "splatCompanionColor";
  readonly colorOrder = 10;
  /** Whether Highlight is on (the uniform's `w`). */
  on = false;
  /** The layers' tilesets: whose slots are highlighted. */
  #owners: ReadonlySet<unknown> = new Set();
  #primitive: SplatPrimitive | undefined;
  #slotsOf: unknown;
  #stale = true;
  #slots: Cartesian4[] = [];
  readonly #tint = new Cartesian4(...INFERRED_HIGHLIGHT.tint);
  readonly #pattern = new Cartesian4(
    INFERRED_HIGHLIGHT.stripeM,
    INFERRED_HIGHLIGHT.stripeDark,
    INFERRED_HIGHLIGHT.opacity,
    0,
  );

  addToShader(shaderBuilder: SplatShaderBuilder, uniformMap: Record<string, () => unknown>): void {
    shaderBuilder.addVertexLines(COMPANION_COLOR_GLSL);
    uniformMap.u_companionTint = () => this.#tint;
    uniformMap.u_companionPattern = () => {
      this.#pattern.w = this.on ? 1 : 0;
      return this.#pattern;
    };
    uniformMap.u_companionSlots = () => this.slots();
  }

  /** Whose slots are highlighted: the layers' tilesets. */
  setOwners(owners: readonly unknown[]): void {
    const same = owners.length === this.#owners.size && owners.every((o) => this.#owners.has(o));
    if (same) return;
    this.#owners = new Set(owners);
    this.#stale = true;
  }

  /** The owners' slot ranges, as the uniform carries them, read again as the slots change. */
  slots(): Cartesian4[] {
    const primitive = this.#primitive;
    if (this.#stale || primitive?._tileSlots !== this.#slotsOf) {
      this.#stale = false;
      this.#slotsOf = primitive?._tileSlots;
      this.#slots = companionSlots(primitive, this.#owners).vecs.map(
        ([a, b, c, d]) => new Cartesian4(a, b, c, d),
      );
    }
    return this.#slots;
  }

  install(primitive: SplatPrimitive & ColorPrimitive): boolean {
    if (this.#primitive !== primitive) {
      if (this.#primitive) this.uninstall();
      this.#primitive = primitive;
      this.#stale = true;
    }
    return hasColorPart(primitive, this) || addColorPart(primitive, this);
  }

  uninstall(): void {
    const primitive = this.#primitive;
    this.#primitive = undefined;
    if (primitive) removeColorPart(primitive, this);
  }
}

/**
 * The layers asset `assetId` draws now, whichever renderer draws them: the picked fill method's
 * (state/variants.ts), or Today's `extras.inferredLayers`; with the method's name.
 */
export function pickedLayers(
  assetId: string,
  extras: unknown,
): { refs: InferredLayerRef[]; variant: string | null } {
  const variants = variantsOf(extras);
  const picked = variants.fill.length > 0 ? pickedVariant(assetId, "fill", variants) : null;
  return { refs: inferredLayersFor(extras, picked), variant: picked?.name ?? null };
}

/** The patched engine's primitive, as far as drawing companions goes. */
type CompanionPrimitive = SplatPrimitive & ColorPrimitive & VisibilityPrimitive;

/** Whether `primitive` can draw other tilesets' tiles in its sort now (incremental, patched). */
export function drawsCompanions(primitive: CompanionPrimitive | undefined): boolean {
  return primitive?.incremental === true && primitive.companions !== undefined;
}

interface Layer {
  tileset: Cesium3DTileset;
  cones: ReturnType<typeof attachLayerViewCones>;
  highlight: InferredHighlight;
  offLoad: () => void;
}

/**
 * Loads `parent`'s inferred layers into `scene`; returns the disposer. `drawn` says whether
 * CesiumJS draws the measured scan (`parent.show`, the default): while another renderer draws
 * it, that renderer draws the layers too (scanView/scanLayers.ts).
 */
export function attachInferredLayers(
  parent: Cesium3DTileset,
  scene: Pick<Scene, "primitives" | "preUpdate" | "requestRender">,
  assetId: string,
  load: LoadLayer = loadLayer,
  drawn: () => boolean = () => parent.show,
): () => void {
  const extras = (parent.root as { extras?: unknown } | undefined)?.extras;
  const url = (parent as unknown as { resource?: { url?: string } }).resource?.url;
  const variants = variantsOf(extras);
  const offersVariants = variants.fill.length > 0;
  const current = () => pickedLayers(assetId, extras);
  if (!url || (current().refs.length === 0 && !offersVariants)) return () => undefined;
  const layers: Layer[] = [];
  let disposed = false;
  let style: InferredStyle = useSettings.getState().inferredStyle;
  /** Highlight on the scan's primitive, for the layers it draws (one for them all). */
  const companionHighlight = new CompanionHighlight();
  /** The scan's primitive while it draws the layers, or undefined. */
  let host: CompanionPrimitive | undefined;
  const sync = (): void => {
    const wanted = style !== "hide" && drawn();
    const primitive: CompanionPrimitive | undefined = splatTilesetOf(parent).gaussianSplatPrimitive;
    // One sort with the scan's splats, when its primitive can (see the file comment).
    const together = wanted && layers.length > 0 && drawsCompanions(primitive);
    const next = together ? primitive : undefined;
    if (host && host !== next) {
      host.companions = [];
      companionHighlight.uninstall();
    }
    host = next;
    companionHighlight.on = style === "highlight";
    companionHighlight.setOwners(layers.map((l) => l.tileset));
    for (const layer of layers) {
      const { tileset, highlight } = layer;
      if (!Matrix4.equals(tileset.modelMatrix, parent.modelMatrix))
        tileset.modelMatrix = Matrix4.clone(parent.modelMatrix);
      if (tileset.show !== wanted) tileset.show = wanted;
      const own: CompanionPrimitive | undefined = splatTilesetOf(tileset).gaussianSplatPrimitive;
      if (own) {
        // Drawn by the scan's primitive, or by its own; its own hooks are there for the latter.
        if (own.drawnBy !== host) own.drawnBy = host;
        if (!hasColorPart(own, highlight)) addColorPart(own, highlight);
      }
      highlight.on = style === "highlight";
      layer.cones.sync();
    }
    if (host) {
      const companions = layers.map((l) => l.tileset);
      const same =
        host.companions?.length === companions.length &&
        companions.every((t, i) => host?.companions?.[i] === t);
      if (!same) host.companions = companions;
      companionHighlight.install(host);
    }
  };
  const offUpdate = scene.preUpdate.addEventListener(sync);
  const offStyle = useSettings.subscribe((state) => {
    if (state.inferredStyle === style) return;
    style = state.inferredStyle;
    scene.requestRender();
  });
  const report = (status: VariantStatus | null): void => {
    if (offersVariants) useVariants.getState().setStatus(assetId, "fill", status);
  };
  const unload = (): void => {
    if (host) {
      host.companions = [];
      companionHighlight.uninstall();
      host = undefined;
    }
    for (const { tileset, cones, highlight, offLoad } of layers.splice(0)) {
      offLoad();
      cones.dispose();
      const own: CompanionPrimitive | undefined = splatTilesetOf(tileset).gaussianSplatPrimitive;
      if (own) {
        own.drawnBy = undefined;
        removeColorPart(own, highlight);
      }
      scene.primitives.remove(tileset);
    }
    useInferred.getState().setLayers(assetId, []);
  };
  /** The layers drawn now (their uris), and which load is the latest. */
  let shown: string | null = null;
  let serial = 0;
  const follow = (): void => {
    const { refs, variant } = current();
    const key = refs.map((ref) => ref.uri).join("\n");
    if (key === shown) return;
    shown = key;
    const mine = ++serial;
    unload();
    scene.requestRender();
    if (refs.length === 0) {
      report({ state: "ready" });
      return;
    }
    report({ state: "loading" });
    const evidence: InferredEvidence[] = [];
    void Promise.allSettled(
      refs.map(async (ref) => {
        const tileset = await load(resolveLayerUrl(url, ref.uri), parent);
        if (disposed || mine !== serial) {
          tileset.destroy();
          return;
        }
        // The layer's own root has the final say on what it is.
        const said = evidenceOf((tileset.root as { extras?: unknown } | undefined)?.extras);
        scene.primitives.add(tileset);
        layers.push({
          tileset,
          cones: attachLayerViewCones(tileset, () => host),
          highlight: new InferredHighlight(),
          // A tile that loads is drawn by the scan's primitive from its first frame: its own
          // primitive (made with its first tile) is told before it would upload anything.
          offLoad: tileset.tileLoad.addEventListener(sync),
        });
        evidence.push(said ?? ref.evidence);
        useInferred.getState().setLayers(assetId, [...evidence]);
        sync();
        scene.requestRender();
      }),
    ).then((results) => {
      if (disposed || mine !== serial) return;
      const failed = results.filter((r): r is PromiseRejectedResult => r.status === "rejected");
      for (const result of failed)
        log.warn("inferred layer did not load", {
          asset: assetId,
          variant,
          error: String(result.reason),
        });
      report(
        failed.length > 0
          ? {
              state: "error",
              message:
                failed.length === refs.length
                  ? String(failed[0]?.reason)
                  : `${String(failed.length)} of ${String(refs.length)} layers did not load`,
            }
          : { state: "ready" },
      );
    });
  };
  const offVariant = offersVariants ? onPickChange(assetId, "fill", follow) : () => undefined;
  // The measured splats the picked fill replaces, hidden while it is shown (lib/supersedes.ts).
  const offSupersedes = offersVariants ? followSupersedes(assetId, variants, url) : () => undefined;
  follow();
  return () => {
    disposed = true;
    offVariant();
    offSupersedes();
    offUpdate();
    offStyle();
    unload();
    report(null);
  };
}
