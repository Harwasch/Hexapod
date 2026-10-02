/**
 * Several visibility weights on one splat primitive.
 *
 * The engine patch has one `vertexVisibility` slot (`patches/@cesium__engine@26.3.0.patch`):
 * an object whose shader lines define `float splatVertexVisibility(uint, vec3)`. More than one
 * thing wants a say -- the view cones fade what a capture never saw (`splatViewCones.ts`), an
 * instance can be hidden (`splatInstances.ts`) -- so the slot holds a chain: each part defines
 * its own function and the chain's `splatVertexVisibility` multiplies them, stopping at the
 * first 0. A part joins or leaves without the others noticing; the draw command is rebuilt
 * on the next render either way.
 */

import type { SplatPrimitive, SplatShaderBuilder } from "./splatInternals";

/** What the patched engine calls on each draw-command build (`vertexVisibility`). */
export interface SplatVertexVisibility {
  addToShader(
    shaderBuilder: SplatShaderBuilder,
    uniformMap: Record<string, () => unknown>,
    context: unknown,
  ): void;
}

/** A primitive of the patched engine, with the visibility accessor. */
export type VisibilityPrimitive = SplatPrimitive & { vertexVisibility?: SplatVertexVisibility };

/** One weight in the chain. */
export interface SplatVisibilityPart {
  /**
   * The GLSL function its lines define, `float <name>(uint splatIndex, vec3 position)`:
   * 0 draws nothing, anything else multiplies the splat's opacity.
   */
  readonly visibilityFunction: string;
  /** Lower runs first; the cheapest test that can return 0 should go first. */
  readonly visibilityOrder: number;
  addToShader(
    shaderBuilder: SplatShaderBuilder,
    uniformMap: Record<string, () => unknown>,
    context: unknown,
  ): void;
}

/** The composite installed in the slot. */
export class SplatVisibilityChain implements SplatVertexVisibility {
  readonly #parts: SplatVisibilityPart[] = [];

  get parts(): readonly SplatVisibilityPart[] {
    return this.#parts;
  }

  add(part: SplatVisibilityPart): boolean {
    if (this.#parts.includes(part)) return false;
    this.#parts.push(part);
    this.#parts.sort((a, b) => a.visibilityOrder - b.visibilityOrder);
    return true;
  }

  remove(part: SplatVisibilityPart): boolean {
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
      composeVisibilityGlsl(this.#parts.map((p) => p.visibilityFunction)),
    );
  }
}

/** `splatVertexVisibility` as the product of `functions`, in order, stopping at 0. */
export function composeVisibilityGlsl(functions: readonly string[]): string {
  const steps = functions
    .map(
      (name) => `    weight *= ${name}(splatIndex, position);
    if (weight <= 0.0) {
        return 0.0;
    }`,
    )
    .join("\n");
  return `
float splatVertexVisibility(uint splatIndex, vec3 position) {
    float weight = 1.0;
${steps}
    return weight;
}
`;
}

/** The chain in `primitive`'s slot, if it is ours. */
export function visibilityChainOf(
  primitive: VisibilityPrimitive,
): SplatVisibilityChain | undefined {
  const current = primitive.vertexVisibility;
  return current instanceof SplatVisibilityChain ? current : undefined;
}

/** Re-assigns the slot so the patched engine rebuilds the draw command. */
function refresh(primitive: VisibilityPrimitive, chain: SplatVisibilityChain): void {
  primitive.vertexVisibility = undefined;
  if (chain.parts.length > 0) primitive.vertexVisibility = chain;
}

/**
 * Adds `part` to `primitive`'s visibility chain, starting one if the slot is empty. False on
 * an engine without the slot, or when something other than a chain holds it.
 */
export function addVisibilityPart(
  primitive: VisibilityPrimitive,
  part: SplatVisibilityPart,
): boolean {
  if (!("vertexVisibility" in primitive)) return false;
  if (primitive.vertexVisibility !== undefined && !visibilityChainOf(primitive)) return false;
  const chain = visibilityChainOf(primitive) ?? new SplatVisibilityChain();
  if (chain.add(part) || primitive.vertexVisibility !== chain) refresh(primitive, chain);
  return true;
}

/** Takes `part` out of `primitive`'s chain; the slot is emptied with its last part. */
export function removeVisibilityPart(
  primitive: VisibilityPrimitive,
  part: SplatVisibilityPart,
): void {
  if (primitive.isDestroyed?.() === true) return;
  const chain = visibilityChainOf(primitive);
  if (chain?.remove(part)) refresh(primitive, chain);
}

/** Whether `part` is in `primitive`'s chain. */
export function hasVisibilityPart(
  primitive: VisibilityPrimitive | undefined,
  part: SplatVisibilityPart,
): boolean {
  return primitive !== undefined && (visibilityChainOf(primitive)?.parts.includes(part) ?? false);
}
