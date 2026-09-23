/**
 * Capturing the packed splat buffer the engine uploads, without patching the engine.
 *
 * `GaussianSplatPrimitive.generateSplatTexture` resolves `generateFromAttributes` as a *property
 * lookup* on the `GaussianSplatTextureGenerator` module object at call time, and that module is
 * exported from the public barrel (`cesium/Source/Cesium.js:638`). Replacing the property
 * therefore hands us the exact `Uint32Array` the engine goes on to turn into a texture — correct
 * by construction, and free.
 *
 * Why we need it at all: `copyFrom` takes a contiguous rectangle, and each splat's position
 * texel sits next to its covariance/colour texel, so any rectangle covering the positions covers
 * the covariance too. Either we reimplement the engine's packer on the CPU (a lane-order mistake
 * there corrupts geometry and colour silently), read the texture back with `gl.readPixels` (a
 * correct but 760–900 ms stall, measured), or keep the buffer the engine already packed. This is
 * the third.
 *
 * `gl.readPixels` remains the recovery path for a snapshot whose creation this missed — it is
 * not implemented here, and the deformer reports `no-capture` instead, because a readback must
 * never land on the steady-state path.
 *
 * **Install before the first splat tile loads.** The interception only sees later calls; a
 * tileset whose texture was generated first will show `no-capture` until its next rebuild.
 */

import * as CesiumBarrel from "cesium";

import { createLogger } from "@/lib/log";

import { digestSplatPositions, recordSplatCapture } from "./splatCaptureRegistry";

const log = createLogger("splat-capture");

/** The parameters `generateSplatTexture` passes. Only the fields we read are declared. */
interface GenerateParameters {
  readonly attributes?: { readonly positions?: Float32Array };
  readonly count?: number;
}

/** What the WASM task resolves to. `data` is the packed RGBA32UI buffer. */
interface GeneratedTextureData {
  readonly data?: Uint32Array;
}

type GenerateFromAttributes = (
  this: unknown,
  parameters: GenerateParameters,
) => Promise<GeneratedTextureData | undefined> | undefined;

interface TextureGeneratorModule {
  generateFromAttributes: GenerateFromAttributes;
}

/** Reads the undeclared `GaussianSplatTextureGenerator` export off the barrel. */
function textureGeneratorModule(): TextureGeneratorModule | undefined {
  const barrel = CesiumBarrel as unknown as Record<string, unknown>;
  const candidate = barrel.GaussianSplatTextureGenerator;
  if (typeof candidate !== "function" && typeof candidate !== "object") return undefined;
  if (candidate === null) return undefined;
  const module = candidate as TextureGeneratorModule;
  return typeof module.generateFromAttributes === "function" ? module : undefined;
}

let installed: { module: TextureGeneratorModule; original: GenerateFromAttributes } | undefined;

/** Restores the engine's own `generateFromAttributes`. Safe to call when nothing is installed. */
export function uninstallSplatTextureInterception(): void {
  if (installed === undefined) return;
  installed.module.generateFromAttributes = installed.original;
  installed = undefined;
}

/** Whether the interception is currently in place. */
export function splatTextureInterceptionInstalled(): boolean {
  return installed !== undefined;
}

/**
 * Installs the interception. Idempotent; returns the uninstaller.
 *
 * Nothing here may throw into the engine's update: a capture is an optimisation for a feature
 * that is allowed to be absent, and a scene that renders splats must keep rendering them
 * whatever this does.
 */
export function installSplatTextureInterception(): () => void {
  if (installed !== undefined) return uninstallSplatTextureInterception;
  const module = textureGeneratorModule();
  if (module === undefined) {
    log.warn("GaussianSplatTextureGenerator is not on the cesium barrel; deformation unavailable");
    return () => undefined;
  }
  const original = module.generateFromAttributes;

  module.generateFromAttributes = function intercepted(
    this: unknown,
    parameters: GenerateParameters,
  ) {
    let key: { count: number; digest: string } | undefined;
    try {
      const positions = parameters.attributes?.positions;
      const count = parameters.count;
      if (positions instanceof Float32Array && typeof count === "number" && count > 0) {
        // Before the call, not after: `scheduleTask` transfers these buffers to the worker and
        // leaves the views detached.
        key = { count, digest: digestSplatPositions(positions, count) };
      }
    } catch (error) {
      log.warn("could not digest splat positions", { error: String(error) });
    }

    const promise = original.call(this, parameters);
    if (promise !== undefined && key !== undefined) {
      const matched = key;
      void promise
        .then((result) => {
          const data = result?.data;
          if (data instanceof Uint32Array) {
            recordSplatCapture({
              count: matched.count,
              digest: matched.digest,
              data,
              at: Date.now(),
            });
          }
          return undefined;
        })
        .catch(() => undefined);
    }
    return promise;
  };

  installed = { module, original };
  log.info("splat texture interception installed");
  return uninstallSplatTextureInterception;
}
