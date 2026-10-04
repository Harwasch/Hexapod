/**
 * A scan's objects (scene instances, `lib/instances.ts`) hidden and highlighted under a
 * dedicated splat renderer, as `splatInstances.ts` does under CesiumJS's own.
 *
 * The data is the same: the file's per-tile runs, keyed by the checksum of each tile's own
 * positions, give each gaussian of a tile its instance id; a small state table (RGBA8, one
 * texel an id, `writeStateTexels`) says which ids are hidden (`r`) and highlighted (`g`). A
 * back-end that streams the tiles itself digests each tile as it decodes it, writes the ids
 * beside its splats and reads the table in its own shader (`ScanBackend.setInstances`): a
 * hidden splat gets no opacity, a highlighted one is pulled toward the tint, and while
 * anything is highlighted the rest is dimmed -- the same rule and numbers as CesiumJS's
 * (`evaluateInstanceShading`).
 *
 * `linkScanInstances` follows the store (`state/instances.ts`) for one asset and hands the
 * back-end a fresh style whenever what is hidden or highlighted changes. A back-end that
 * cannot apply it (no `setInstances`, or a scan streamed in a format without object ids) is
 * reported to the store as a gap, which the objects panel shows with a switch to CesiumJS.
 */

import type { InstancesDoc } from "@/lib/instances";
import { useInstances } from "@/state/instances";
import { effectiveDoc, onCustomSetsChange } from "@/state/sceneSelect";

import {
  HIGHLIGHT_STYLE,
  INSTANCE_TEXTURE_WIDTH,
  instancesDocOf,
  stateTextureRows,
  writeStateTexels,
  type HighlightStyle,
} from "../splatInstances";
import type { ScanBackend } from "./types";

/** Everything a back-end needs to draw a scan's objects hidden or highlighted. */
export interface InstanceStyle {
  readonly doc: InstancesDoc;
  /** RGBA8 texels, `INSTANCE_TEXTURE_WIDTH` a row, `rows` rows: `r` hidden, `g` highlighted. */
  readonly state: Uint8Array;
  readonly rows: number;
  /**
   * x: anything hidden or highlighted (otherwise the shader returns at once); y: the largest
   * id; z: anything highlighted; w: unused.
   */
  readonly params: readonly [number, number, number, number];
  /** Linear RGB the highlight is pulled toward, and how far (`a`). */
  readonly tint: readonly [number, number, number, number];
  /** While a highlight is active, everything else: colour times `x`, opacity times `y`. */
  readonly dim: readonly [number, number, number, number];
}

/**
 * The style for `doc` with `hidden` and `highlighted`: the store's exact sets (a category's or
 * an object's members, `state/instances.ts`), applied id for id.
 */
export function instanceStyle(
  doc: InstancesDoc,
  hidden: ReadonlySet<number>,
  highlighted: ReadonlySet<number>,
  dimOthers: boolean,
  style: HighlightStyle = HIGHLIGHT_STYLE,
): InstanceStyle {
  const rows = stateTextureRows(doc.maxId);
  const state = new Uint8Array(INSTANCE_TEXTURE_WIDTH * rows * 4);
  writeStateTexels(state, doc.maxId, hidden, highlighted);
  const anyHidden = hidden.size > 0;
  const anyLit = highlighted.size > 0;
  return {
    doc,
    state,
    rows,
    params: [anyHidden || anyLit ? 1 : 0, doc.maxId, anyLit ? 1 : 0, 0],
    tint: style.tint,
    dim: dimOthers ? [style.dim[0], style.dim[1], 0, 0] : [1, 1, 0, 0],
  };
}

/**
 * The GLSL both WebGL back-ends share: the id's state texel, and the colour rule. `id` is the
 * splat's instance id (0 for none); the caller declares the uniforms named here.
 */
export const SCAN_INSTANCE_RULE_GLSL = `
vec4 hexapodInstanceState(uint id) {
    if (id == 0u || float(id) > uInstanceParams.y) {
        return vec4(0.0);
    }
    int i = int(id);
    return texelFetch(uInstanceState, ivec2(i & ${String(INSTANCE_TEXTURE_WIDTH - 1)}, i >> ${String(Math.log2(INSTANCE_TEXTURE_WIDTH))}), 0);
}

vec4 hexapodInstanceColor(uint id, vec4 color) {
    if (uInstanceParams.x < 0.5) {
        return color;
    }
    vec4 state = hexapodInstanceState(id);
    if (state.r > 0.5) {
        return vec4(color.rgb, 0.0);
    }
    if (uInstanceParams.z < 0.5) {
        return color;
    }
    if (state.g > 0.5) {
        return vec4(mix(color.rgb, uInstanceTint.rgb, uInstanceTint.a) + 0.06, color.a);
    }
    return vec4(color.rgb * uInstanceDim.x, color.a * uInstanceDim.y);
}
`;

/**
 * The same rule in WGSL, for PlayCanvas on WebGPU (Spark has no WebGPU path). A line-for-line
 * port of `SCAN_INSTANCE_RULE_GLSL` -- same names, same constants, same order of tests -- so
 * that hiding and highlighting look the same on either API; `scanInstances.test.ts` holds the
 * two to that. PlayCanvas's WGSL reads uniforms through its `uniform.` block and declares a
 * texture without a sampler as `var name: texture_2d<f32>`, read with `textureLoad` (GLSL's
 * `texelFetch`). The caller declares the uniforms and the texture.
 */
export const SCAN_INSTANCE_RULE_WGSL = `
fn hexapodInstanceState(id: u32) -> vec4f {
    if (id == 0u || f32(id) > uniform.uInstanceParams.y) {
        return vec4f(0.0);
    }
    let i = i32(id);
    return textureLoad(uInstanceState, vec2i(i & ${String(INSTANCE_TEXTURE_WIDTH - 1)}, i >> ${String(Math.log2(INSTANCE_TEXTURE_WIDTH))}u), 0);
}

fn hexapodInstanceColor(id: u32, color: vec4f) -> vec4f {
    if (uniform.uInstanceParams.x < 0.5) {
        return color;
    }
    let state = hexapodInstanceState(id);
    if (state.r > 0.5) {
        return vec4f(color.rgb, 0.0);
    }
    if (uniform.uInstanceParams.z < 0.5) {
        return color;
    }
    if (state.g > 0.5) {
        return vec4f(mix(color.rgb, uniform.uInstanceTint.rgb, uniform.uInstanceTint.a) + 0.06, color.a);
    }
    return vec4f(color.rgb * uniform.uInstanceDim.x, color.a * uniform.uInstanceDim.y);
}
`;

/**
 * The ids of a tile whose splats a renderer reordered -- PlayCanvas sorts each tile along a
 * Morton curve, and `order[i]` is the tile's own splat now at `i` -- written into `out`:
 * the file's ids permuted the same way, or zeros for a tile the file does not list.
 */
export function idsInResourceOrder(
  ids: Uint32Array | undefined,
  order: Uint32Array,
  out: Uint32Array,
): Uint32Array {
  out.fill(0);
  if (!ids) return out;
  for (let i = 0; i < order.length; i++) out[i] = ids[order[i] ?? 0] ?? 0;
  return out;
}

/**
 * The scan's `instances.json` with the objects painted in this browser drawn as ids of their
 * own (lib/customSets.ts): what the back-ends draw from.
 */
export function paintedDocOf(assetId: string): InstancesDoc | undefined {
  const base = instancesDocOf(assetId);
  return base ? effectiveDoc(assetId, base) : undefined;
}

/** Why a back-end cannot apply the style, or null when it can. */
export function instanceGap(
  backend: Pick<ScanBackend<unknown>, "setInstances" | "name">,
  native: boolean,
): string | null {
  if (!backend.setInstances) {
    return `The ${backend.name} renderer cannot draw them.`;
  }
  if (native) {
    return "This scan streams in PlayCanvas's own format, which carries no object ids.";
  }
  return null;
}

/**
 * Keeps `backend` drawing asset `assetId`'s objects as the store has them, from the moment
 * its instances load. `native` is a session that streams a package without object ids.
 * `restyled` is called after each style the back-end is handed: the overlay draws only when
 * something changes (overlayFrames.ts), and a hide or highlight is a change no camera makes.
 * Returns the disposer, which also clears any gap it reported.
 */
export function linkScanInstances(
  assetId: string,
  backend: ScanBackend<unknown>,
  native: boolean,
  docOf: (assetId: string) => InstancesDoc | undefined = paintedDocOf,
  restyled: () => void = () => undefined,
): () => void {
  const gap = instanceGap(backend, native);
  const store = useInstances;
  if (gap !== null) {
    store.getState().setGap(assetId, { renderer: backend.name, reason: gap });
    return () => store.getState().setGap(assetId, null);
  }
  let last: { doc: InstancesDoc; hidden: unknown; highlighted: unknown; dim: boolean } | null =
    null;
  const push = (): void => {
    const state = store.getState();
    const entry = state.assets[assetId];
    const doc = entry ? docOf(assetId) : undefined;
    if (!entry || !doc) {
      if (last !== null) {
        backend.setInstances?.(null);
        restyled();
      }
      last = null;
      return;
    }
    if (
      last?.doc === doc &&
      last.hidden === entry.hidden &&
      last.highlighted === entry.highlighted &&
      last.dim === state.dimOthers
    ) {
      return;
    }
    last = { doc, hidden: entry.hidden, highlighted: entry.highlighted, dim: state.dimOthers };
    backend.setInstances?.(instanceStyle(doc, entry.hidden, entry.highlighted, state.dimOthers));
    restyled();
  };
  const off = store.subscribe(push);
  const offCustom = onCustomSetsChange(assetId, push);
  push();
  return () => {
    off();
    offCustom();
    backend.setInstances?.(null);
  };
}
