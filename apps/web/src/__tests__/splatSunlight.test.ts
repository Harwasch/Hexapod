/**
 * The sunlight relight in the scene (cesium/splatSunlight.ts, sunlight.ts, sunlightSlot.ts):
 * what the colour part declares to the patched engine and where it sits in the chain; the sun
 * it works out, against the almanac; and that the relight joins the shader only while the sun
 * is on and As captured is off, reports what it drew, and leaves nothing behind. Under PlayCanvas
 * the same relight hands its sun, in the scan's frame, to the renderer drawing the scan
 * (scanView/scanSunlight.ts) and reports only while one does. The GLSL compiles and lights an
 * east-facing slab in a real engine, under both, in e2e/sunlightScene.spec.ts.
 */
import {
  Cartesian3,
  Ellipsoid,
  Event,
  JulianDate,
  Matrix3,
  Matrix4,
  Simon1994PlanetaryPositions,
  Transforms,
  type Cesium3DTileset,
  type Scene,
} from "cesium";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  publishScanSun,
  scanSunDrawn,
  watchScanSun,
  type ScanSun,
} from "@/cesium/scanView/scanSunlight";
import { colorChainOf, composeColorGlsl, type SplatVertexColor } from "@/cesium/splatColor";
import type { SplatShaderBuilder } from "@/cesium/splatInternals";
import { SplatSunlight, sunSine } from "@/cesium/splatSunlight";
import { attachSunlight } from "@/cesium/sunlightSlot";
import type * as sunlightModule from "@/cesium/sunlight";
import { startSunlight } from "@/cesium/sunlight";
import { FLAT_RATIO, SUN_AMBIENT } from "@/lib/splatNormals";
import { useAsCaptured } from "@/state/asCaptured";
import { useLighting } from "@/state/lighting";
import { useRendererOverride, useSettings } from "@/state/settings";
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
    // The yard's frame: east, north, up at its origin, as a splat tileset's root has it.
    root: { computedTransform: Transforms.eastNorthUpToFixedFrame(YARD, Ellipsoid.WGS84) },
    show: true,
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

beforeEach(() => {
  useSettings.getState().set({ splatRenderer: "cesium" });
});

afterEach(() => {
  useSettings.getState().reset();
  useRendererOverride.setState({ renderer: null });
  useLighting.setState({ sun: "off" });
  useAsCaptured.setState({ on: false });
  useSunReport.setState({ reports: {} });
  vi.useRealTimers();
  vi.restoreAllMocks();
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
    // No sun to light by: strength 0, so the shader leaves the colour as it is.
    part.sunUp = Number.NaN;
    expect(uniforms.u_sunlight?.()).toEqual({ vec4: [0, 1, 0, 0] });
    const unlit = uniforms.u_sunlight?.();
    part.sunUp = Number.NaN;
    expect(uniforms.u_sunlight?.()).toBe(unlit);
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

  it("turns by TEME before the engine's Earth-orientation data has loaded, as the engine does", () => {
    // No ICRF matrix yet (the XYS and EOP data still loading): the engine lights by TEME.
    const icrf = vi
      .spyOn(Transforms, "computeIcrfToCentralBodyFixedMatrix")
      .mockReturnValue(undefined);
    expect(degrees(sine("2026-06-21T20:21:00Z"))).toBeCloseTo(67.3, 0);
    icrf.mockRestore();
  });

  it("is NaN, never a made-up sun, when the engine's matrix is not a number", () => {
    const icrf = vi
      .spyOn(Transforms, "computeIcrfToCentralBodyFixedMatrix")
      .mockImplementation((_time, result) =>
        Matrix3.fromArray(new Array<number>(9).fill(Number.NaN), 0, result),
      );
    expect(sine("2026-06-21T20:21:00Z")).toBeNaN();
    icrf.mockRestore();
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

  it("draws nothing while another renderer draws splats, and comes back with CesiumJS", () => {
    const s = scan();
    const handle = startSunlight(s.tileset, s.scene, "a", { vec4 });
    handle.sync(true);
    s.preRender.raiseEvent(s.scene, JulianDate.fromIso8601("2026-06-21T20:21:00Z"));
    expect(useSunReport.getState().reports.a).toBeDefined();
    useSettings.getState().set({ splatRenderer: "playcanvas" });
    expect(s.functions()).toEqual([]);
    expect(useSunReport.getState().reports.a).toBeUndefined();
    useSettings.getState().set({ splatRenderer: "cesium" });
    expect(s.functions()).toEqual(["splatSunlightColor"]);
    // A page address's choice counts as the setting does.
    useRendererOverride.setState({ renderer: "spark" });
    expect(s.functions()).toEqual([]);
    useRendererOverride.setState({ renderer: null });
    expect(s.functions()).toEqual(["splatSunlightColor"]);
    handle.dispose();
    useSettings.getState().set({ splatRenderer: "playcanvas" });
    useSettings.getState().set({ splatRenderer: "cesium" });
    expect(s.functions()).toEqual([]);
  });

  it("draws no relight and claims none for a frame whose sun is not a number", () => {
    const s = scan();
    const part = { value: undefined as unknown };
    const handle = startSunlight(s.tileset, s.scene, "a", {
      vec4: (...v) => (part.value = vec4(...v)),
    });
    handle.sync(true);
    const noon = JulianDate.fromIso8601("2026-06-21T20:21:00Z");
    s.preRender.raiseEvent(s.scene, noon);
    expect(part.value).toEqual({ vec4: [1, 1, 0, 0] });
    expect(useSunReport.getState().reports.a).toBeDefined();
    // The engine's sun not yet a number: strength 0 (the colour untouched), and no report.
    const icrf = vi
      .spyOn(Transforms, "computeIcrfToCentralBodyFixedMatrix")
      .mockImplementation((_time, result) =>
        Matrix3.fromArray(new Array<number>(9).fill(Number.NaN), 0, result),
      );
    s.preRender.raiseEvent(s.scene, noon);
    expect(part.value).toEqual({ vec4: [0, 1, 0, 0] });
    expect(useSunReport.getState().reports.a).toBeUndefined();
    icrf.mockRestore();
    s.preRender.raiseEvent(s.scene, noon);
    expect(part.value).toEqual({ vec4: [1, 1, 0, 0] });
    expect(useSunReport.getState().reports.a?.sine).toBeGreaterThan(0.9);
    handle.dispose();
  });

  it("reports only while its tileset is shown, so a hidden date claims no relight", () => {
    const s = scan();
    const handle = startSunlight(s.tileset, s.scene, "a", { vec4 });
    handle.sync(true);
    const noon = JulianDate.fromIso8601("2026-06-21T20:21:00Z");
    s.preRender.raiseEvent(s.scene, noon);
    expect(useSunReport.getState().reports.a).toBeDefined();
    // Hidden (another date shown, the site left behind, a representation switched): its
    // report goes at the next frame, and none comes while it stays hidden.
    s.tileset.show = false;
    s.preRender.raiseEvent(s.scene, noon);
    expect(useSunReport.getState().reports.a).toBeUndefined();
    s.preRender.raiseEvent(s.scene, JulianDate.fromIso8601("2026-06-21T20:25:00Z"));
    expect(useSunReport.getState().reports.a).toBeUndefined();
    // A hidden tileset loaded on start reports nothing at all.
    const other = scan();
    other.tileset.show = false;
    const cached = startSunlight(other.tileset, other.scene, "b", { vec4 });
    cached.sync(true);
    other.preRender.raiseEvent(other.scene, noon);
    expect(useSunReport.getState().reports.b).toBeUndefined();
    // Shown again: reported again.
    s.tileset.show = true;
    s.preRender.raiseEvent(s.scene, noon);
    expect(useSunReport.getState().reports.a).toBeDefined();
    handle.dispose();
    cached.dispose();
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

/** The sun's unit direction in the yard's east/north/up frame at `iso`, by the almanac. */
function enuSun(iso: string): [number, number, number] {
  const time = JulianDate.fromIso8601(iso);
  const m = new Matrix3();
  Transforms.computeIcrfToCentralBodyFixedMatrix(time, m);
  const p = Simon1994PlanetaryPositions.computeSunPositionInEarthInertialFrame(time);
  Matrix3.multiplyByVector(m, p, p);
  const toLocal = Matrix4.inverseTransformation(
    Transforms.eastNorthUpToFixedFrame(YARD, Ellipsoid.WGS84),
    new Matrix4(),
  );
  const local = Matrix4.multiplyByPointAsVector(toLocal, p, new Cartesian3());
  Cartesian3.normalize(local, local);
  return [local.x, local.y, local.z];
}

describe("the relight under PlayCanvas", () => {
  const MORNING = "2026-06-21T15:30:00Z";
  beforeEach(() => {
    useSettings.getState().set({ splatRenderer: "playcanvas" });
  });
  afterEach(() => {
    publishScanSun("a", null);
  });

  it("hands the renderer drawing the scan the sun in the scan's frame, and reports only then", () => {
    const s = scan();
    const handle = startSunlight(s.tileset, s.scene, "a", { vec4 });
    handle.sync(true);
    // Not CesiumJS's part: PlayCanvas draws the scan.
    expect(s.functions()).toEqual([]);
    // No renderer draws it yet: nothing claimed.
    s.preRender.raiseEvent(s.scene, JulianDate.fromIso8601(MORNING));
    expect(useSunReport.getState().reports.a).toBeUndefined();
    // The renderer starts drawing it: a frame is asked for, the sun worked out in it.
    const seen: (ScanSun | null)[] = [];
    const renders = s.renders.count;
    const unwatch = watchScanSun("a", (sun) => seen.push(sun));
    expect(s.renders.count).toBeGreaterThan(renders);
    s.preRender.raiseEvent(s.scene, JulianDate.fromIso8601(MORNING));
    const sun = seen.at(-1);
    expect(sun?.strength).toBe(1);
    expect(sun?.sunUp).toBe(1);
    const expected = enuSun(MORNING);
    for (let i = 0; i < 3; i += 1) expect(sun?.dir[i]).toBeCloseTo(expected[i]!, 6);
    // East and up in the morning, as the almanac has it.
    expect(sun!.dir[0]).toBeGreaterThan(0.8);
    expect(useSunReport.getState().reports.a?.at).toBe("2026-06-21T15:30:00.000Z");
    // A frame whose sun barely moved hands nothing on (the overlay would draw for it).
    const handed = seen.length;
    s.preRender.raiseEvent(s.scene, JulianDate.fromIso8601("2026-06-21T15:30:01Z"));
    expect(seen.length).toBe(handed);
    // Night: sunUp 0, reported below the horizon.
    s.preRender.raiseEvent(s.scene, JulianDate.fromIso8601("2026-06-22T08:21:00Z"));
    expect(seen.at(-1)?.sunUp).toBe(0);
    expect(useSunReport.getState().reports.a?.sine).toBeLessThan(0);
    // The renderer stops drawing it: its report goes with the next frame.
    unwatch();
    expect(scanSunDrawn("a")).toBe(false);
    s.preRender.raiseEvent(s.scene, JulianDate.fromIso8601(MORNING));
    expect(useSunReport.getState().reports.a).toBeUndefined();
    handle.dispose();
  });

  it("draws the photograph for a sun that is not a number, and claims nothing", () => {
    const s = scan();
    const handle = startSunlight(s.tileset, s.scene, "a", { vec4 });
    handle.sync(true);
    const seen: (ScanSun | null)[] = [];
    const unwatch = watchScanSun("a", (sun) => seen.push(sun));
    const icrf = vi
      .spyOn(Transforms, "computeIcrfToCentralBodyFixedMatrix")
      .mockImplementation((_time, result) =>
        Matrix3.fromArray(new Array<number>(9).fill(Number.NaN), 0, result),
      );
    s.preRender.raiseEvent(s.scene, JulianDate.fromIso8601(MORNING));
    expect(seen.at(-1)?.strength).toBe(0);
    expect(useSunReport.getState().reports.a).toBeUndefined();
    icrf.mockRestore();
    s.preRender.raiseEvent(s.scene, JulianDate.fromIso8601(MORNING));
    expect(seen.at(-1)?.strength).toBe(1);
    unwatch();
    handle.dispose();
  });

  it("takes the sun back when turned off, under As captured's slot, and on dispose", () => {
    const s = scan();
    const handle = startSunlight(s.tileset, s.scene, "a", { vec4 });
    const seen: (ScanSun | null)[] = [];
    const unwatch = watchScanSun("a", (sun) => seen.push(sun));
    handle.sync(true);
    s.preRender.raiseEvent(s.scene, JulianDate.fromIso8601(MORNING));
    expect(seen.at(-1)?.strength).toBe(1);
    handle.sync(false);
    expect(seen.at(-1)).toBeNull();
    expect(useSunReport.getState().reports.a).toBeUndefined();
    // Frames while off hand nothing on.
    s.preRender.raiseEvent(s.scene, JulianDate.fromIso8601(MORNING));
    expect(seen.at(-1)).toBeNull();
    handle.sync(true);
    s.preRender.raiseEvent(s.scene, JulianDate.fromIso8601(MORNING));
    expect(seen.at(-1)?.strength).toBe(1);
    handle.dispose();
    expect(seen.at(-1)).toBeNull();
    unwatch();
  });

  it("hands nothing to Spark or the WebGPU trial, which do not draw it, and moves with the setting", () => {
    const s = scan();
    const handle = startSunlight(s.tileset, s.scene, "a", { vec4 });
    const seen: (ScanSun | null)[] = [];
    const unwatch = watchScanSun("a", (sun) => seen.push(sun));
    handle.sync(true);
    s.preRender.raiseEvent(s.scene, JulianDate.fromIso8601(MORNING));
    expect(seen.at(-1)?.strength).toBe(1);
    for (const renderer of ["spark", "playcanvas-webgpu"] as const) {
      useSettings.getState().set({ splatRenderer: renderer });
      expect(seen.at(-1)).toBeNull();
      s.preRender.raiseEvent(s.scene, JulianDate.fromIso8601(MORNING));
      expect(seen.at(-1)).toBeNull();
      expect(s.functions()).toEqual([]);
      expect(useSunReport.getState().reports.a).toBeUndefined();
    }
    // CesiumJS: its own part, and no sun handed to a renderer that is not drawing.
    useSettings.getState().set({ splatRenderer: "cesium" });
    expect(s.functions()).toEqual(["splatSunlightColor"]);
    expect(seen.at(-1)).toBeNull();
    useSettings.getState().set({ splatRenderer: "playcanvas" });
    expect(s.functions()).toEqual([]);
    s.preRender.raiseEvent(s.scene, JulianDate.fromIso8601(MORNING));
    expect(seen.at(-1)?.strength).toBe(1);
    unwatch();
    handle.dispose();
  });
});
