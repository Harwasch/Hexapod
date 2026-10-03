/**
 * The objects' colour rule in both shading languages (cesium/scanView/scanInstances.ts) and
 * PlayCanvas's work-buffer modifier built on it (playcanvasBackend.ts): the WGSL port must say
 * what the GLSL says, line for line, so hide and highlight look the same on WebGPU as on WebGL2;
 * and the WGSL modifier must define PlayCanvas's three functions with PlayCanvas's own WGSL
 * signatures, or WebGPU fails to build the work-buffer pipeline and draws no tile at all.
 * (Compiling it is e2e/instances.spec.ts's `@webgpu` test, on a real WebGPU device.)
 */

import { describe, expect, it } from "vitest";

import {
  PLAYCANVAS_INSTANCE_WGSL,
  playcanvasModifier,
  playcanvasModifierGlsl,
} from "@/cesium/scanView/playcanvasBackend";
import { SCAN_INSTANCE_RULE_GLSL, SCAN_INSTANCE_RULE_WGSL } from "@/cesium/scanView/scanInstances";
import { INSTANCE_TEXTURE_WIDTH } from "@/cesium/splatInstances";

/**
 * A rule's statements with each language's spelling of the same thing made one: WGSL's
 * `uniform.` block, its `vec4f`/`vec2i`/`f32`/`i32` and `let`, `textureLoad` for GLSL's
 * `texelFetch`, and the `u` an unsigned literal carries. Signatures are left out (the types are
 * written differently by nature); what remains must be identical.
 */
function statements(source: string): string[] {
  return source
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line !== "" && !/^(fn |vec4 hexapod)/.test(line))
    .map((line) =>
      line
        .replace(/\buniform\./g, "")
        .replace(/\bvec4f\(/g, "vec4(")
        .replace(/\bvec2i\(/g, "ivec2(")
        .replace(/\bf32\(/g, "float(")
        .replace(/\bi32\(/g, "int(")
        .replace(/\btextureLoad\(/g, "texelFetch(")
        .replace(/^(?:let|vec4|int)\s+(\w+)\s*=/, "$1 =")
        .replace(/\b(\d+)u\b/g, "$1"),
    );
}

/** PlayCanvas 2.22's default WGSL modifier (shader-lib/wgsl/chunks/gsplat/vert/gsplatModify.js). */
const PLAYCANVAS_WGSL_SIGNATURES = [
  "fn modifySplatCenter(center: ptr<function, vec3f>) {",
  "fn modifySplatRotationScale(originalCenter: vec3f, modifiedCenter: vec3f, rotation: ptr<function, vec4f>, scale: ptr<function, vec3f>) {",
  "fn modifySplatColor(center: vec3f, color: ptr<function, vec4f>) {",
];

describe("the objects' colour rule on WebGPU", () => {
  it("says in WGSL what the GLSL says, statement for statement", () => {
    const wgsl = statements(SCAN_INSTANCE_RULE_WGSL);
    expect(wgsl).toEqual(statements(SCAN_INSTANCE_RULE_GLSL));
    // Something was compared: the state lookup, the three outcomes.
    expect(wgsl).toContain("return vec4(color.rgb, 0.0);");
    expect(wgsl).toContain(
      "return vec4(mix(color.rgb, uInstanceTint.rgb, uInstanceTint.a) + 0.06, color.a);",
    );
    expect(wgsl).toContain("return vec4(color.rgb * uInstanceDim.x, color.a * uInstanceDim.y);");
  });

  it("reads the state table with the GLSL's row and column", () => {
    const shift = Math.log2(INSTANCE_TEXTURE_WIDTH);
    expect(SCAN_INSTANCE_RULE_WGSL).toContain(
      `vec2i(i & ${String(INSTANCE_TEXTURE_WIDTH - 1)}, i >> ${String(shift)}u)`,
    );
  });

  it("defines PlayCanvas's three modifier functions with its WGSL signatures", () => {
    const lines = PLAYCANVAS_INSTANCE_WGSL.split("\n").map((line) => line.trim());
    for (const signature of PLAYCANVAS_WGSL_SIGNATURES) expect(lines).toContain(signature);
    // The id stream as PlayCanvas declares an R32U stream (a `vec4u`), the colour by pointer.
    expect(PLAYCANVAS_INSTANCE_WGSL).toContain(
      "*color = hexapodInstanceColor(loadSplatInstance().r, *color);",
    );
    // Uniforms and the table as PlayCanvas's WGSL preprocessor takes them.
    for (const name of ["uInstanceParams", "uInstanceTint", "uInstanceDim"]) {
      expect(lines).toContain(`uniform ${name}: vec4f;`);
    }
    expect(lines).toContain("var uInstanceState: texture_2d<f32>;");
    // No GLSL left in it.
    expect(PLAYCANVAS_INSTANCE_WGSL).not.toMatch(/\b(?:void|texelFetch|highp|sampler2D)\b/);
  });

  it("hands PlayCanvas both languages for hide and highlight, for it to take its device's", () => {
    expect(playcanvasModifier(true, false)).toEqual({
      glsl: playcanvasModifierGlsl(true, false),
      wgsl: PLAYCANVAS_INSTANCE_WGSL,
    });
  });

  it("has no WGSL for a skin: the motion is GLSL only, and not applied on WebGPU", () => {
    expect(playcanvasModifier(true, true).wgsl).toBeUndefined();
    expect(playcanvasModifier(false, true).wgsl).toBeUndefined();
    expect(playcanvasModifier(false, true).glsl).toContain("loadSplatWeights()");
    // One object per kind: a tile's modifier is compared by identity.
    expect(playcanvasModifier(true, true)).toBe(playcanvasModifier(true, true));
  });
});
