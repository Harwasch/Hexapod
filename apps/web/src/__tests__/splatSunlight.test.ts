/**
 * The sunlight relight in the scene (cesium/splatSunlight.ts, sunlight.ts, sunlightSlot.ts):
 * what the colour part declares to the patched engine and where it sits in the chain; the sun
 * it works out, against the almanac; and that the relight joins the shader only while the sun
 * is on and As captured is off, reports what it drew, and leaves nothing behind. The GLSL
 * compiles and lights an east-facing slab in a real engine in e2e/sunlightScene.spec.ts.
 */
import { Cartesian3, Event, JulianDate, type Cesium3DTileset, type Scene } from "cesium";
import { afterEach, describe, expect, it, vi } from "vitest";

import { colorChainOf, composeColorGlsl, type SplatVertexColor } from "@/cesium/splatColor";
import type { SplatShaderBuilder } from "@/cesium/splatInternals";
import { SplatSunlight, sunSine } from "@/cesium/splatSunlight";
import { attachSunlight } from "@/cesium/sunlightSlot";
import type * as sunlightModule from "@/cesium/sunlight";
import { startSunlight } from "@/cesium/sunlight";
import { FLAT_RATIO, SUN_AMBIENT } from "@/lib/splatNormals";
import { useAsCaptured } from "@/state/asCaptured";
import { useLighting } from "@/state/lighting";
import { useSunReport } from "@/state/sunReport";

const vec4 = (x: number, y: number, z: number, w: number) => ({ vec4: [x, y, z, w] });

/** The yard's frame origin (synthetic_yard.py). */
const YARD = Cartesian3.fromDegrees(-123.88, 46.1335, 0);

function build(part: SplatSunlight) {
  const out: string[] = [];
  const uniforms: Record<string, () => unknown> = {};
  const builder: SplatShaderBuilder = {
    addUniform: (type, name) => out.push(`uniform ${type} ${name};`),
    addVertexLines: (text) => out.push(...(typeof text === "string" ? [text] : text)),
  };
  part.addToShader(builder, uniforms, {});
  return { lines: out.join("\n"), uniforms };
}

function scan() {
  const primitive = { vertexColor: undefined as SplatVertexColor | undefined };
  const tileLoad = new Event();
  const tileset = {
    gaussianSplatPrimitive: primitive as typeof primitive | undefined,
    tileLoad,
    boundingSphere: { center: YARD },
  } as unknown as Cesium3DTileset & { gaussianSplatPrimitive?: typeof primitive };
  const preRender = new Event();
  const renders = { count: 0 };
  const scene = {
    preRender,
    requestRender: () => {
      renders.count += 1;
    },
  } as unknown as Scene;
  const functions = () =>
    (colorChainOf(primitive as never)?.parts ?? []).map((p) => p.colorFunction);
  return { primitive, tileset, tileLoad, scene, preRender, renders, functions };
}

afterEach(() => {
  useLighting.setState({ sun: "off" });
  useAsCaptured.setState({ on: false });
  useSunReport.setState({ reports: {} });
  vi.useRealTimers();
});

describe("the colour part", () => {
  it("declares only its own uniform and reads the engine's texture where the engine does", () => {
    const { lines, uniforms } = build(new SplatSunlight(vec4));
    expect(lines).toContain("uniform vec4 u_sunlight;");
    expect(lines).toContain("vec4 splatSunlightColor(uint splatIndex, vec3 position, vec4 color)");
    // The engine declares these before any hook's lines; a second declaration would not link.
    expect(lines).not.toMatch(/uniform\s+\S+\s+u_splatAttributeTexture/);
    expect(lines).not.toMatch(/uniform\s+\S+\s+u_splatRow(Mask|Shift)/);
    // The covariance texel: the odd column beside the position (splatTexels.ts covarianceTexel).
    expect(lines).toContain("int(((splatIndex & rowMask) << 1) | 1u), int(splatIndex >> rowShift)");
    expect(lines).toContain("mat3(u1.x, u1.y, u2.x, u1.y, u2.y, u3.x, u2.x, u3.x, u3.y)");
    // The engine's sun and model-view: automatic uniforms, declared by the engine on use.
    expect(lines).toContain("czm_sunDirectionEC");
    expect(lines).toContain("mat3(czm_modelView)");
    // The motion's Jacobian only where the motion chain declares one.
    expect(lines).toMatch(
      /#if defined\(HAS_SPLAT_VERTEX_MOTION\) && defined\(HAS_SPLAT_VERTEX_JACOBIAN\)\s+mat3 jacobian = splatVertexJacobian\(splatIndex, position\);/,
    );
    // The constants are the mirror's.
    expect(lines).toContain(`SPLAT_SUN_FLAT = ${String(FLAT_RATIO)};`);
    expect(lines).toContain(`SPLAT_SUN_AMBIENT = ${String(SUN_AMBIENT)};`);
    expect(Object.keys(uniforms)).toEqual(["u_sunlight"]);
    expect(uniforms.u_sunlight?.()).toEqual({ vec4: [1, 1, 0, 0] });
  });

  it("sets the sun's height as a uniform only, without a rebuild", () => {
    const part = new SplatSunlight(vec4);
    const { uniforms } = build(part);
    part.sunUp = 0;
    expect(uniforms.u_sunlight?.()).toEqual({ vec4: [1, 0, 0, 0] });
    const before = uniforms.u_sunlight?.();
    part.sunUp = 0;
    expect(uniforms.u_sunlight?.()).toBe(before);
  });

  it("runs after the objects' highlight and before Confidence / Data age", () => {
    const { primitive, functions } = scan();
    const others = [0, 5, 10].map((order) => ({
      colorFunction: `part${String(order)}`,
      colorOrder: order,
      addToShader: () => undefined,
    }));
    const chain = () => colorChainOf(primitive);
    const sunlight = new SplatSunlight(vec4);
    // Joined to a chain that holds others, between them; the others untouched when it leaves.
    expect(sunlight.install(primitive as never)).toBe(true);
    for (const p of others) chain()?.add(p);
    expect(functions()).toEqual(["part0", "splatSunlightColor", "part5", "part10"]);
    const glsl = composeColorGlsl(functions());
    expect(glsl.indexOf("splatSunlightColor")).toBeLessThan(glsl.indexOf("part5"));
    sunlight.uninstall();
    expect(functions()).toEqual(["part0", "part5", "part10"]);
  });

  it("is refused by an engine without the colour slot", () => {
    expect(new SplatSunlight(vec4).install({})).toBe(false);
  });
});

describe("the sun", () => {
  const sine = (iso: string) => sunSine(JulianDate.fromIso8601(iso), YARD);
  const degrees = (s: number) => (Math.asin(s) * 180) / Math.PI;

  it("stands where the almanac puts it over the yard at the June solstice", () => {
    // Solar noon at 123.88° W is about 20:21 UTC (equation of time -2 min): 90 - 46.13 + 23.44.
    expect(degrees(sine("2026-06-21T20:21:00Z"))).toBeCloseTo(67.3, 0);
    // Local midnight: below the horizon by about the complement.
    expect(degrees(sine("2026-06-22T08:21:00Z"))).toBeLessThan(-15);
  });

  it("is NaN for a model that is not placed yet", () => {
    expect(sunSine(JulianDate.fromIso8601("2026-06-21T20:00:00Z"), Cartesian3.ZERO)).toBeNaN();
  });
});

describe("the relight", () => {
  it("joins the chain while on, sets the sun each frame, reports what it drew", () => {
    const s = scan();
    const handle = startSunlight(s.tileset, s.scene, "a", { vec4 });
    expect(s.functions()).toEqual([]);
    handle.sync(true);
    expect(s.functions()).toEqual(["splatSunlightColor"]);
    expect(s.renders.count).toBeGreaterThan(0);
    // A night frame: the sun below the horizon, reported to the minute.
    s.preRender.raiseEvent(s.scene, JulianDate.fromIso8601("2026-06-22T08:21:42Z"));
    const night = useSunReport.getState().reports.a;
    expect(night?.at).toBe("2026-06-22T08:21:00.000Z");
    expect(night?.sine).toBeLessThan(0);
    // By day.
    s.preRender.raiseEvent(s.scene, JulianDate.fromIso8601("2026-06-21T20:21:00Z"));
    expect(useSunReport.getState().reports.a?.sine).toBeGreaterThan(0.9);
    handle.sync(false);
    expect(s.functions()).toEqual([]);
    expect(useSunReport.getState().reports.a).toBeUndefined();
    handle.dispose();
  });

  it("waits for the primitive, which comes with the first tile", () => {
    const s = scan();
    s.tileset.gaussianSplatPrimitive = undefined;
    const handle = startSunlight(s.tileset, s.scene, "a", { vec4 });
    handle.sync(true);
    s.tileset.gaussianSplatPrimitive = s.primitive;
    expect(s.functions()).toEqual([]);
    s.tileLoad.raiseEvent();
    expect(s.functions()).toEqual(["splatSunlightColor"]);
    handle.dispose();
    expect(s.functions()).toEqual([]);
  });

  it("asks for a frame each minute while on, so the sun of now moves on", () => {
    vi.useFakeTimers();
    const s = scan();
    const handle = startSunlight(s.tileset, s.scene, "a", { vec4 });
    handle.sync(true);
    const before = s.renders.count;
    vi.advanceTimersByTime(60_000);
    expect(s.renders.count).toBe(before + 1);
    handle.sync(false);
    const after = s.renders.count;
    vi.advanceTimersByTime(180_000);
    expect(s.renders.count).toBe(after);
    handle.dispose();
  });
});

describe("the slot", () => {
  const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

  it("loads nothing while the sun is off, and draws nothing while As captured is on", async () => {
    const s = scan();
    const load = vi.fn(() => import("@/cesium/sunlight"));
    const detach = attachSunlight(s.tileset, s.scene, "a", load);
    await flush();
    expect(load).not.toHaveBeenCalled();
    useAsCaptured.getState().set(true);
    useLighting.getState().set("now");
    await flush();
    expect(load).not.toHaveBeenCalled();
    expect(s.functions()).toEqual([]);

    useAsCaptured.getState().set(false);
    await vi.waitFor(() => expect(s.functions()).toEqual(["splatSunlightColor"]));
    expect(load).toHaveBeenCalledTimes(1);
    useAsCaptured.getState().set(true);
    expect(s.functions()).toEqual([]);
    useAsCaptured.getState().set(false);
    expect(s.functions()).toEqual(["splatSunlightColor"]);
    useLighting.getState().set("capture");
    expect(s.functions()).toEqual(["splatSunlightColor"]);
    useLighting.getState().set("off");
    expect(s.functions()).toEqual([]);
    detach();
    useLighting.getState().set("now");
    await flush();
    expect(s.functions()).toEqual([]);
    expect(load).toHaveBeenCalledTimes(1);
  });

  it("claims nothing when the relight fails to load, and tries again on the next switch", async () => {
    const s = scan();
    const load = vi
      .fn<() => Promise<typeof sunlightModule>>()
      .mockRejectedValueOnce(new Error("offline"))
      .mockImplementation(() => import("@/cesium/sunlight"));
    const detach = attachSunlight(s.tileset, s.scene, "a", load);
    useLighting.getState().set("now");
    await flush();
    expect(s.functions()).toEqual([]);
    expect(useSunReport.getState().reports).toEqual({});
    useLighting.getState().set("capture");
    await vi.waitFor(() => expect(s.functions()).toEqual(["splatSunlightColor"]));
    detach();
    expect(s.functions()).toEqual([]);
  });
});
