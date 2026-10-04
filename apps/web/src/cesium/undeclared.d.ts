/**
 * Exports of the CesiumJS barrel that `Cesium.d.ts` does not declare.
 *
 * All six are real runtime exports of `@cesium/engine` (`index.js`) re-exported by
 * `cesium/Source/Cesium.js`: the Gaussian splat subsystem is missing from the typings
 * altogether (docs/CESIUM.md, "API changes noted while building"), and the renderer's
 * `Texture`, `Sampler` and `ShaderDestination` are private API. The splat hooks used to reach
 * them as properties of `import * as CesiumBarrel from "cesium"`, which compiled without
 * these declarations -- and also made the bundler keep every one of the engine's exports,
 * because a namespace read by a computed key could be read for any of them. Importing them by
 * name lets the build drop what nothing uses (655 kB of the engine chunk, measured).
 *
 * Each is declared `unknown` on purpose. The engine patch and the internals the hooks rely on
 * are version-pinned, not typed, and every consumer already narrows what it reads
 * (`typeof x === "function"`, a field present) and degrades when a CesiumJS upgrade moves
 * something, so a precise type here would only be a second, unchecked claim about it. What
 * the named import adds is a build-time check: a renamed export is a bundler error, not a
 * feature that silently switched itself off.
 */
import "cesium";

declare module "cesium" {
  /** `Scene/GaussianSplatPrimitive.js`; the patch adds the static `sortHook`. */
  export const GaussianSplatPrimitive: unknown;
  /** `Scene/GaussianSplatTextureGenerator.js`, a module object with `generateFromAttributes`. */
  export const GaussianSplatTextureGenerator: unknown;
  /** `Scene/GltfSpzLoader.js`; the patch adds the static `decodeHook`. */
  export const GltfSpzLoader: unknown;
  /** `Renderer/Texture.js`. */
  export const Texture: unknown;
  /** `Renderer/Sampler.js`, for `Sampler.NEAREST`. */
  export const Sampler: unknown;
  /** `Renderer/ShaderDestination.js`. */
  export const ShaderDestination: unknown;
}
