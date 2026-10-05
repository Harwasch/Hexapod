/**
 * Draws a measured splat's inferred layers (lib/inferred.ts) beside it: each its own
 * tileset, placed where the measured one is, shown only while the scan is drawn and as the
 * viewer's `inferredStyle` says (Show, Highlight or Hide), and faded by its own view cones --
 * so a fill made for the outside of a scan is never drawn from the inside.
 *
 * **Every renderer.** CesiumJS draws the layers whichever renderer draws the measured splats:
 * under PlayCanvas or Spark the scan's own tileset is hidden (it is drawn on the overlay,
 * cesium/scanView), but its layers stay CesiumJS's, on the globe's canvas under the overlay.
 * Splats write no depth in any renderer, so a layer there sits under the measured splats where
 * both cover a pixel -- which, for what was generated where no camera saw, is the right way
 * round. `drawn` says whether the scan is on screen at all, by either.
 *
 * **Highlight** is the engine patch's colour hook (`vertexColor`) on each layer's own splat
 * primitive -- never the measured scan's -- installed once and switched by a uniform:
 * pulled toward purple, hatched in bands across the layer, a little see-through
 * (`INFERRED_HIGHLIGHT`). One place draws it, so it reads the same under every renderer.
 *
 * **Which layers** are the pick's (`state/variants.ts`): Today's `extras.inferredLayers`, or a
 * fill variant's. Picking another unloads the drawn layers and loads its own.
 */
import { Cartesian4, Cesium3DTileset, Matrix4, type Scene } from "cesium";

import { createLogger } from "@/lib/log";
import {
  evidenceOf,
  INFERRED_HIGHLIGHT,
  resolveLayerUrl,
  type InferredEvidence,
  type InferredStyle,
} from "@/lib/inferred";
import { inferredLayersFor, variantsOf } from "@/lib/variants";
import { useInferred } from "@/state/inferred";
import { useSettings } from "@/state/settings";
import { onPickChange, pickedVariant, useVariants, type VariantStatus } from "@/state/variants";

import type { SplatVertexColor } from "./splatInstances";
import { keepOffscreenSplats, splatTilesetOf, type SplatShaderBuilder } from "./splatInternals";
import { attachViewCones } from "./splatViewCones";

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
 * The patched engine's `splatVertexColor` for an inferred layer: as painted while
 * `u_inferredPattern.w` is 0 (Show), else Highlight. `position` is the splat's, in its
 * tileset's root frame (metres): the bands run diagonally across it, so every face of a fill
 * is hatched.
 */
export const INFERRED_COLOR_GLSL = `
uniform vec4 u_inferredTint;
// x: band width (m); y: the darker bands' brightness; z: opacity factor; w: highlight on.
uniform vec4 u_inferredPattern;

vec4 splatVertexColor(uint splatIndex, vec3 position, vec4 color) {
    if (u_inferredPattern.w < 0.5) {
        return color;
    }
    vec3 rgb = mix(color.rgb, u_inferredTint.rgb, u_inferredTint.a);
    float band = fract(dot(position, vec3(0.57735027)) / u_inferredPattern.x);
    rgb *= band < 0.5 ? 1.0 : u_inferredPattern.y;
    return vec4(rgb, color.a * u_inferredPattern.z);
}
`;

/** What `INFERRED_COLOR_GLSL` does to one colour (straight alpha), for tests. */
export function evaluateInferredColor(
  color: readonly [number, number, number, number],
  position: readonly [number, number, number],
  highlight: boolean,
): [number, number, number, number] {
  if (!highlight) return [color[0], color[1], color[2], color[3]];
  const { tint, stripeM, stripeDark, opacity } = INFERRED_HIGHLIGHT;
  const along = (position[0] + position[1] + position[2]) * 0.57735027;
  const band = along / stripeM - Math.floor(along / stripeM);
  const shade = band < 0.5 ? 1 : stripeDark;
  const mix = (c: number, t: number): number => (c + (t - c) * tint[3]) * shade;
  return [
    mix(color[0], tint[0]),
    mix(color[1], tint[1]),
    mix(color[2], tint[2]),
    color[3] * opacity,
  ];
}

/** The colour hook of one layer: installed with its primitive, Highlight by a uniform. */
export class InferredHighlight implements SplatVertexColor {
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

/** A layer's primitive, as far as the colour hook goes (the patched engine's accessor). */
interface ColorPrimitive {
  vertexColor?: SplatVertexColor;
}

interface Layer {
  tileset: Cesium3DTileset;
  off: () => void;
  highlight: InferredHighlight;
}

/**
 * Loads `parent`'s inferred layers into `scene`; returns the disposer. `drawn` says whether
 * the measured scan is on screen, by CesiumJS (`parent.show`, the default) or by another
 * renderer (SiteManager knows which).
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
  const current = () => {
    const picked = offersVariants ? pickedVariant(assetId, "fill", variants) : null;
    return { refs: inferredLayersFor(extras, picked), variant: picked?.name ?? null };
  };
  if (!url || (current().refs.length === 0 && !offersVariants)) return () => undefined;
  const layers: Layer[] = [];
  let disposed = false;
  let style: InferredStyle = useSettings.getState().inferredStyle;
  const sync = (): void => {
    const wanted = style !== "hide" && drawn();
    for (const { tileset, highlight } of layers) {
      if (!Matrix4.equals(tileset.modelMatrix, parent.modelMatrix))
        tileset.modelMatrix = Matrix4.clone(parent.modelMatrix);
      if (tileset.show !== wanted) tileset.show = wanted;
      const primitive = splatTilesetOf(tileset).gaussianSplatPrimitive as
        (ColorPrimitive & object) | undefined;
      // The hook is the layer's own: a primitive whose slot something else holds is left be.
      if (primitive && "vertexColor" in primitive && primitive.vertexColor === undefined)
        primitive.vertexColor = highlight;
      highlight.on = style === "highlight";
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
    for (const { tileset, off, highlight } of layers.splice(0)) {
      off();
      const primitive = splatTilesetOf(tileset).gaussianSplatPrimitive as
        ColorPrimitive | undefined;
      if (primitive?.vertexColor === highlight) primitive.vertexColor = undefined;
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
        layers.push({ tileset, off: attachViewCones(tileset), highlight: new InferredHighlight() });
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
  follow();
  return () => {
    disposed = true;
    offVariant();
    offUpdate();
    offStyle();
    unload();
    report(null);
  };
}
