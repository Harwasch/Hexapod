/**
 * The per-cell view rules' two shader parts (cesium/splatCompositor.ts) as they compose into
 * a splat primitive's chains: the visibility part after a hidden instance and before the view
 * cones, the colour part after the inferred highlight; their shared helpers declared once
 * whichever chain the build reaches first; the uniforms saying what lib/compositor.ts says
 * (cell, feather, plane, slots covered, the tinted sources), and nothing acting until the
 * frame is known. Compiling and drawing is e2e/compositorScene.spec.ts's, in a real engine.
 */
import { describe, expect, it } from "vitest";

import { composeColorGlsl, SplatColorChain, type SplatColorPart } from "@/cesium/splatColor";
import {
  COMPOSITOR_COLOR_GLSL,
  COMPOSITOR_HELPERS_GLSL,
  COMPOSITOR_VISIBILITY_GLSL,
  ownerTexels,
  OWNER_TEXTURE_WIDTH,
  SplatCompositor,
  type CompositorGpu,
} from "@/cesium/splatCompositor";
import { NO_SOURCE, type SourceTexture } from "@/cesium/splatSources";
import {
  composeVisibilityGlsl,
  SplatVisibilityChain,
  type SplatVisibilityPart,
} from "@/cesium/splatVisibility";
import { featherWidth, NO_OWNER, type OwnerPlane } from "@/lib/compositor";
import { INFERRED_HIGHLIGHT } from "@/lib/inferred";

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

function gpu() {
  const made: { height: number; data: Uint8Array }[] = [];
  const g: CompositorGpu = {
    createR8: (_c, _w, height, data): SourceTexture => {
      made.push({ height, data: data.slice() });
      return { copyFrom: () => undefined, isDestroyed: () => false, destroy: () => undefined };
    },
    mat4: (values) => [...values],
    vec4: (x, y, z, w) => [x, y, z, w],
  };
  return { g, made };
}

function primitive() {
  return {
    vertexVisibility: undefined as unknown,
    vertexColor: undefined as unknown,
    _rootTransform: undefined as number[] | undefined,
    _tileSlots: undefined as
      Map<{ tileset: unknown }, { start: number; count: number }> | undefined,
    isDestroyed: () => false,
  };
}

function plane(): OwnerPlane {
  return {
    rule: "latest",
    lattice: { baseM: 0.25, level: 2 },
    cellM: 1,
    i0: -2,
    j0: -3,
    ni: 3,
    nj: 2,
    owners: Uint8Array.from([0, 0, 1, 1, NO_OWNER, 2]),
    sources: [],
  };
}

const part = (name: string, order: number): SplatVisibilityPart => ({
  visibilityFunction: name,
  visibilityOrder: order,
  addToShader: () => undefined,
});

function build(slot: { addToShader: SplatVisibilityChain["addToShader"] } | undefined): {
  lines: string;
  uniforms: Record<string, () => unknown>;
} {
  const lines: string[] = [];
  const uniforms: Record<string, () => unknown> = {};
  slot?.addToShader(
    { addUniform: () => undefined, addVertexLines: (l) => void lines.push(String(l)) },
    uniforms,
    {},
  );
  return { lines: lines.join("\n"), uniforms };
}

describe("the view rules' shader parts", () => {
  it("weigh after a hidden instance and before the view cones, and colour after Highlight", () => {
    const { g } = gpu();
    const compositor = new SplatCompositor(g, () => IDENTITY);
    const visibility = new SplatVisibilityChain();
    visibility.add(part("splatViewConeVisibility", 10));
    visibility.add(compositor);
    visibility.add(part("splatInstanceVisibility", 0));
    expect(composeVisibilityGlsl(visibility.parts.map((p) => p.visibilityFunction))).toContain(
      [
        "    weight *= splatInstanceVisibility(splatIndex, position);",
        "    if (weight <= 0.0) {",
        "        return 0.0;",
        "    }",
        "    weight *= splatCompositorVisibility(splatIndex, position);",
      ].join("\n"),
    );
    const colour = new SplatColorChain();
    const highlight: SplatColorPart = {
      colorFunction: "splatCompanionColor",
      colorOrder: 10,
      addToShader: () => undefined,
    };
    colour.add(compositor.colorHook);
    colour.add(highlight);
    expect(composeColorGlsl(colour.parts.map((p) => p.colorFunction))).toContain(
      "    color = splatCompanionColor(splatIndex, position, color);\n    color = splatCompositorColor(splatIndex, position, color);",
    );
  });

  it("declare their helpers once, whichever chain is built first", () => {
    const { g } = gpu();
    const compositor = new SplatCompositor(g, () => IDENTITY);
    const p = primitive();
    expect(compositor.install(p as never)).toBe(true);
    const visibility = build(p.vertexVisibility as SplatVisibilityChain);
    const colour = build(p.vertexColor as SplatColorChain);
    // Both chains carry the guarded helpers; the preprocessor keeps the first.
    expect(COMPOSITOR_HELPERS_GLSL.trim().startsWith("#ifndef HEXAPOD_SPLAT_COMPOSITOR")).toBe(
      true,
    );
    expect(COMPOSITOR_HELPERS_GLSL.trim().endsWith("#endif")).toBe(true);
    for (const fn of ["int splatCompSource(uint splatIndex)", "int splatCompOwner(ivec2 c)"]) {
      expect(COMPOSITOR_HELPERS_GLSL.split(fn)).toHaveLength(2);
      expect(COMPOSITOR_VISIBILITY_GLSL).not.toContain(fn);
      expect(COMPOSITOR_COLOR_GLSL).not.toContain(fn);
    }
    expect(visibility.lines).toContain(
      "float splatCompositorVisibility(uint splatIndex, vec3 position)",
    );
    expect(visibility.lines).not.toContain("vec4 splatCompositorColor(");
    expect(colour.lines).toContain(
      "vec4 splatCompositorColor(uint splatIndex, vec3 position, vec4 color)",
    );
    expect(colour.lines).not.toContain("float splatCompositorVisibility(");
    expect(Object.keys(visibility.uniforms).sort()).toEqual(
      ["u_compFromModel", "u_compOwners", "u_compParams", "u_compPlane", "u_compSources"].sort(),
    );
    expect(Object.keys(colour.uniforms)).toEqual(
      expect.arrayContaining(["u_compTint", "u_compPattern", "u_compParams", "u_compSources"]),
    );
    compositor.uninstall();
    expect(p.vertexVisibility).toBeUndefined();
    expect(p.vertexColor).toBeUndefined();
  });

  it("read the owner plane's cell and feather, and draw nothing until the frame is known", () => {
    const { g, made } = gpu();
    const frame: { root?: number[] } = {};
    const compositor = new SplatCompositor(g, () => frame.root);
    const p = primitive();
    compositor.install(p as never);
    compositor.setPlane(plane());
    compositor.sources.setSources(["host", "later", "layer"]);
    p._tileSlots = new Map([[{ tileset: "host" }, { start: 0, count: 40 }]]);
    const { uniforms } = build(p.vertexVisibility as SplatVisibilityChain);
    // No frame yet: the rule is on and owns no cell, so nothing is drawn -- fail closed,
    // never two dates blended where the cells cannot be read.
    expect((uniforms.u_compParams?.() as number[])[0]).toBe(1);
    expect(uniforms.u_compPlane?.()).toEqual([0, 0, 0, 0]);
    // Off (a harness's comparison) leaves every splat alone.
    compositor.enabled = false;
    expect((uniforms.u_compParams?.() as number[])[0]).toBe(0);
    compositor.enabled = true;
    frame.root = IDENTITY;
    p._rootTransform = IDENTITY;
    expect(uniforms.u_compParams?.()).toEqual([1, 1, featherWidth(1), 40]);
    expect(uniforms.u_compPlane?.()).toEqual([-2, -3, 3, 2]);
    expect(uniforms.u_compFromModel?.()).toEqual(IDENTITY);
    // The owner texture is the plane's bytes, in whole rows, NO_OWNER after.
    uniforms.u_compOwners?.();
    expect(made.at(-1)?.height).toBe(1);
    expect(Array.from(made.at(-1)?.data.subarray(0, 7) ?? [])).toEqual([
      0,
      0,
      1,
      1,
      NO_OWNER,
      2,
      NO_OWNER,
    ]);
    expect(compositor.active).toBe(true);
  });

  it("tint only the flagged sources, as the inferred highlight looks", () => {
    const { g } = gpu();
    const compositor = new SplatCompositor(g, () => IDENTITY);
    const p = primitive();
    compositor.install(p as never);
    const { uniforms } = build(p.vertexColor as SplatColorChain);
    expect((uniforms.u_compPattern?.() as number[])[3]).toBe(0);
    compositor.tinted = (1 << 2) | (1 << 5);
    expect(uniforms.u_compPattern?.()).toEqual([
      INFERRED_HIGHLIGHT.stripeM,
      INFERRED_HIGHLIGHT.stripeDark,
      INFERRED_HIGHLIGHT.opacity,
      36,
    ]);
    expect(uniforms.u_compTint?.()).toEqual([...INFERRED_HIGHLIGHT.tint]);
    expect(COMPOSITOR_COLOR_GLSL).toContain("((mask >> source) & 1) == 0");
  });

  it("fail closed: a slot of no source, or a cell of no owner, draws nothing", () => {
    expect(COMPOSITOR_VISIBILITY_GLSL).toContain(
      `if (source == ${String(NO_SOURCE)}) {\n        return 0.0;`,
    );
    expect(COMPOSITOR_VISIBILITY_GLSL).toContain("if (owner != source) {\n        return 0.0;");
    expect(COMPOSITOR_HELPERS_GLSL).toContain(`return ${String(NO_OWNER)};`);
    // The feather is lib/compositor.ts's: mix(0.5, 1, smoothstep(0, w, d)).
    expect(COMPOSITOR_VISIBILITY_GLSL).toContain(
      "return mix(0.5, 1.0, smoothstep(0.0, u_compParams.z, d));",
    );
    const big = { owners: new Uint8Array(OWNER_TEXTURE_WIDTH + 1).fill(3) };
    const texels = ownerTexels(big);
    expect(texels.height).toBe(2);
    expect(texels.data[OWNER_TEXTURE_WIDTH]).toBe(3);
    expect(texels.data[OWNER_TEXTURE_WIDTH + 1]).toBe(NO_OWNER);
  });
});
