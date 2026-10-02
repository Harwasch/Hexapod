/**
 * Several motions on one splat primitive.
 *
 * The engine patch has one `vertexMotion` slot (`patches/@cesium__engine@26.3.0.patch`): an
 * object whose shader lines define `vec3 splatVertexMotion(uint, vec3)`. More than one thing
 * moves splats -- the Living Survey's rig (`splatGpuMotion.ts`), a scene object's skin
 * (`splatSkin.ts`) -- so the slot holds a chain, as the visibility slot does
 * (`splatVisibility.ts`): each part defines its own function returning a **displacement** of
 * the rest (fetched) position, and the chain's `splatVertexMotion` adds them up:
 *
 *     x' = x + Σ_parts δ_part(x)
 *
 * Every part sees the rest position, so the order does not matter and the parts do not
 * compound; a part at rest returns exact zeros and the rest position comes back untouched.
 *
 * A part may also give the linear part of its displacement's gradient at the splat
 * (`jacobianFunction`, `mat3 <name>(uint, vec3)`, called after its motion function for the
 * same splat, so it can hand back what that computed). The chain then declares the patch's
 * optional `splatVertexJacobian` as `I + Σ_parts A_part`, and the shader draws the splat's
 * covariance through it (`J · Σ · Jᵀ`). Parts without one leave covariances as measured.
 */

import type { SplatPrimitive, SplatShaderBuilder, SplatVertexMotion } from "./splatInternals";

/** One motion in the chain. */
export interface SplatMotionPart {
  /** The GLSL function its lines define, `vec3 <name>(uint splatIndex, vec3 position)`. */
  readonly motionFunction: string;
  /** Optionally, `mat3 <name>(uint splatIndex, vec3 position)`: the displacement's gradient. */
  readonly jacobianFunction?: string;
  /** Lower comes first in the shader (and declares its uniforms first). */
  readonly motionOrder: number;
  addToShader(
    shaderBuilder: SplatShaderBuilder,
    uniformMap: Record<string, () => unknown>,
    context: unknown,
  ): void;
}

/** The composite installed in the slot. */
export class SplatMotionChain implements SplatVertexMotion {
  readonly #parts: SplatMotionPart[] = [];

  get parts(): readonly SplatMotionPart[] {
    return this.#parts;
  }

  add(part: SplatMotionPart): boolean {
    if (this.#parts.includes(part)) return false;
    this.#parts.push(part);
    this.#parts.sort((a, b) => a.motionOrder - b.motionOrder);
    return true;
  }

  remove(part: SplatMotionPart): boolean {
    const at = this.#parts.indexOf(part);
    if (at < 0) return false;
    this.#parts.splice(at, 1);
    return true;
  }

  addToShader(
    shaderBuilder: SplatShaderBuilder,
    uniformMap: Record<string, () => unknown>,
    context: unknown,
  ): void {
    for (const part of this.#parts) part.addToShader(shaderBuilder, uniformMap, context);
    shaderBuilder.addVertexLines(
      composeMotionGlsl(
        this.#parts.map((p) => p.motionFunction),
        this.#parts.flatMap((p) => (p.jacobianFunction ? [p.jacobianFunction] : [])),
      ),
    );
  }
}

/**
 * `splatVertexMotion` as the rest position plus every part's displacement, and -- when any
 * part has one -- `splatVertexJacobian` as the identity plus every part's linear part.
 */
export function composeMotionGlsl(
  functions: readonly string[],
  jacobians: readonly string[] = [],
): string {
  const steps = functions
    .map((name) => `    displacement += ${name}(splatIndex, position);`)
    .join("\n");
  const motion = `
vec3 splatVertexMotion(uint splatIndex, vec3 position) {
    vec3 displacement = vec3(0.0);
${steps}
    return position + displacement;
}
`;
  if (jacobians.length === 0) return motion;
  const linear = jacobians
    .map((name) => `    jacobian += ${name}(splatIndex, position);`)
    .join("\n");
  return `${motion}
#define HAS_SPLAT_VERTEX_JACOBIAN
mat3 splatVertexJacobian(uint splatIndex, vec3 position) {
    mat3 jacobian = mat3(1.0);
${linear}
    return jacobian;
}
`;
}

/** A primitive of the patched engine, with the motion accessor. */
export type MotionPrimitive = SplatPrimitive;

/** The chain in `primitive`'s slot, if it is ours. */
export function motionChainOf(primitive: MotionPrimitive): SplatMotionChain | undefined {
  const current = primitive.vertexMotion;
  return current instanceof SplatMotionChain ? current : undefined;
}

/** Re-assigns the slot so the patched engine rebuilds the draw command. */
function refresh(primitive: MotionPrimitive, chain: SplatMotionChain): void {
  primitive.vertexMotion = undefined;
  if (chain.parts.length > 0) primitive.vertexMotion = chain;
}

/**
 * Adds `part` to `primitive`'s motion chain, starting one if the slot is empty. False on an
 * engine without the slot, or when something other than a chain holds it.
 */
export function addMotionPart(primitive: MotionPrimitive, part: SplatMotionPart): boolean {
  if (!("vertexMotion" in primitive)) return false;
  if (primitive.vertexMotion !== undefined && !motionChainOf(primitive)) return false;
  const chain = motionChainOf(primitive) ?? new SplatMotionChain();
  if (chain.add(part) || primitive.vertexMotion !== chain) refresh(primitive, chain);
  return true;
}

/** Takes `part` out of `primitive`'s chain; the slot is emptied with its last part. */
export function removeMotionPart(primitive: MotionPrimitive, part: SplatMotionPart): void {
  if (primitive.isDestroyed?.() === true) return;
  const chain = motionChainOf(primitive);
  if (chain?.remove(part)) refresh(primitive, chain);
}

/** Whether `part` is in `primitive`'s chain. */
export function hasMotionPart(
  primitive: MotionPrimitive | undefined,
  part: SplatMotionPart | undefined,
): boolean {
  return (
    primitive !== undefined &&
    part !== undefined &&
    (motionChainOf(primitive)?.parts.includes(part) ?? false)
  );
}
