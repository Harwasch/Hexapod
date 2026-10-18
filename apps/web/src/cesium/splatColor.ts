/**
 * Several colours on one splat primitive.
 *
 * The engine patch has one `vertexColor` slot (`patches/@cesium__engine@26.3.0.patch`): an
 * object whose shader lines define `vec4 splatVertexColor(uint, vec3, vec4)`, called last with
 * the splat's final colour (straight alpha). More than one thing wants a say -- a scan's
 * highlighted objects (`splatInstances.ts`), the inferred layer it draws in its own sort and
 * Highlight's purple on it (`inferredLayers.ts`) -- so the slot holds a chain, as the
 * visibility slot does (`splatVisibility.ts`): each part defines its own function, and the
 * chain's `splatVertexColor` passes the colour through them in order. A part joins or leaves
 * without the others noticing; the draw command is rebuilt on the next render either way.
 */

import type { SplatPrimitive, SplatShaderBuilder } from "./splatInternals";

/** What the patched engine calls on each draw-command build (`vertexColor`). */
export interface SplatVertexColor {
  addToShader(
    shaderBuilder: SplatShaderBuilder,
    uniformMap: Record<string, () => unknown>,
    context: unknown,
  ): void;
}

/** A primitive of the patched engine, with the colour accessor. */
export type ColorPrimitive = SplatPrimitive & { vertexColor?: SplatVertexColor };

/** One colour in the chain. */
export interface SplatColorPart {
  /** The GLSL function its lines define, `vec4 <name>(uint splatIndex, vec3 position, vec4 c)`. */
  readonly colorFunction: string;
  /** Lower runs first. */
  readonly colorOrder: number;
  addToShader(
    shaderBuilder: SplatShaderBuilder,
    uniformMap: Record<string, () => unknown>,
    context: unknown,
  ): void;
}

/** The composite installed in the slot. */
export class SplatColorChain implements SplatVertexColor {
  readonly #parts: SplatColorPart[] = [];

  get parts(): readonly SplatColorPart[] {
    return this.#parts;
  }

  add(part: SplatColorPart): boolean {
    if (this.#parts.includes(part)) return false;
    this.#parts.push(part);
    this.#parts.sort((a, b) => a.colorOrder - b.colorOrder);
    return true;
  }

  remove(part: SplatColorPart): boolean {
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
    shaderBuilder.addVertexLines(composeColorGlsl(this.#parts.map((p) => p.colorFunction)));
  }
}

/** `splatVertexColor` as `functions` applied in turn. */
export function composeColorGlsl(functions: readonly string[]): string {
  const steps = functions
    .map((name) => `    color = ${name}(splatIndex, position, color);`)
    .join("\n");
  return `
vec4 splatVertexColor(uint splatIndex, vec3 position, vec4 color) {
${steps}
    return color;
}
`;
}

/** The chain in `primitive`'s slot, if it is ours. */
export function colorChainOf(primitive: ColorPrimitive): SplatColorChain | undefined {
  const current = primitive.vertexColor;
  return current instanceof SplatColorChain ? current : undefined;
}

/** Re-assigns the slot so the patched engine rebuilds the draw command. */
function refresh(primitive: ColorPrimitive, chain: SplatColorChain): void {
  primitive.vertexColor = undefined;
  if (chain.parts.length > 0) primitive.vertexColor = chain;
}

/**
 * Adds `part` to `primitive`'s colour chain, starting one if the slot is empty. False on an
 * engine without the slot, or when something other than a chain holds it.
 */
export function addColorPart(primitive: ColorPrimitive, part: SplatColorPart): boolean {
  if (!("vertexColor" in primitive)) return false;
  if (primitive.vertexColor !== undefined && !colorChainOf(primitive)) return false;
  const chain = colorChainOf(primitive) ?? new SplatColorChain();
  if (chain.add(part) || primitive.vertexColor !== chain) refresh(primitive, chain);
  return true;
}

/** Takes `part` out of `primitive`'s chain; the slot is emptied with its last part. */
export function removeColorPart(primitive: ColorPrimitive, part: SplatColorPart): void {
  if (primitive.isDestroyed?.() === true) return;
  const chain = colorChainOf(primitive);
  if (chain?.remove(part)) refresh(primitive, chain);
}

/** Whether `part` is in `primitive`'s chain. */
export function hasColorPart(primitive: ColorPrimitive | undefined, part: SplatColorPart): boolean {
  return primitive !== undefined && (colorChainOf(primitive)?.parts.includes(part) ?? false);
}
