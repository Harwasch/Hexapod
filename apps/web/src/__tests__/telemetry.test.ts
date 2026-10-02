/**
 * Telemetry on instances (step C3): the `telemetry.json` reader and the wire format
 * (`lib/telemetry.ts`), the sources (`lib/telemetrySources.ts`), the driver
 * (`cesium/telemetry.ts`: a skinned instance moves by its constant handle, any other through the
 * rigid part, stale tracks rest, bound instances are claimed from the wind) and the rigid part
 * itself (`cesium/splatRigid.ts`: exactly the bound instance's splats move, by the motion, in
 * the baked frame).
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import {
  applyMotion,
  checksumPositions,
  pathPose,
  poseFromScan,
  quatFromYaw,
  relativeMotion,
  scanFrame,
  SkinWindField,
  type RigidMotion,
  type Vec3,
} from "@twin/world";
import { describe, expect, it } from "vitest";

import { claimInstances, claimsOf, isClaimed } from "@/cesium/motionClaims";
import { SkinWindDriver, type InstanceTraits } from "@/cesium/skinWind";
import { composeMotionGlsl, motionChainOf } from "@/cesium/splatMotionChain";
import { evaluateRigidMotion, rigidGlsl, SplatRigidMotion } from "@/cesium/splatRigid";
import { groupEyes, sortMotionOf } from "@/cesium/splatSorter";
import {
  relatedInstances,
  restPoseOf,
  TelemetryDriver,
  type TelemetryRigidTarget,
  type TelemetrySkinTarget,
} from "@/cesium/telemetry";
import {
  parseInstances,
  tileInstanceIds,
  withDescendants,
  type InstancesDoc,
} from "@/lib/instances";
import { parseSkin, skinDisplacement, type SkinDoc } from "@/lib/skin";
import { sortBackToFront } from "@/lib/splatOrder";
import {
  parsePoseMessages,
  parseTelemetry,
  telemetryRefOf,
  type SyntheticSourceConfig,
  type TelemetryDoc,
} from "@/lib/telemetry";
import {
  createTelemetrySource,
  StreamSource,
  SyntheticSource,
  type TelemetrySource,
} from "@/lib/telemetrySources";

import { buildDrawCommand, fakeFactory, FakeHookedPrimitive } from "./splatGpuFixture";
import { FakeTile, FakeTiledTileset } from "./splatTilesFixture";
import { yardBake, yardTiles } from "./splatYardFixture";

const YARD = resolve(process.cwd(), "../../data/tiles/synthetic-yard");
/** The yard's root transform (tileset.json). */
const YARD_ROOT = (
  JSON.parse(readFileSync(resolve(YARD, "splat/tileset.json"), "utf8")) as {
    root: { transform: number[] };
  }
).root.transform;

const yardTelemetryRaw: unknown = JSON.parse(
  readFileSync(resolve(YARD, "telemetry/telemetry.json"), "utf8"),
);

function yardInstances(): InstancesDoc {
  const doc = parseInstances(
    JSON.parse(readFileSync(resolve(YARD, "instances/instances.json"), "utf8")),
  );
  if (!doc) throw new Error("no instances");
  return doc;
}

function yardSkin(): SkinDoc {
  const raw: unknown = JSON.parse(readFileSync(resolve(YARD, "skin/skin.json"), "utf8"));
  const bytes = Uint8Array.from(readFileSync(resolve(YARD, "skin/skin.bin")));
  const doc = parseSkin(raw, bytes.buffer);
  if (!doc) throw new Error("no skin");
  return doc;
}

function yardTelemetry(): TelemetryDoc {
  const doc = parseTelemetry(yardTelemetryRaw);
  if (!doc) throw new Error("no telemetry");
  return doc;
}

function close(a: ArrayLike<number>, b: ArrayLike<number>, tol: number): void {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i += 1)
    expect(Math.abs((a[i] ?? 0) - (b[i] ?? 0))).toBeLessThan(tol);
}

describe("telemetry.json", () => {
  it("reads the committed yard bindings", () => {
    const doc = yardTelemetry();
    expect(doc.issues).toEqual([]);
    expect(doc.bindings.map((b) => [b.instance, b.source, b.stale])).toEqual([
      [8, "yard-loop", "rest"],
      [10, "shrub-loop", "freeze"],
    ]);
    const loop = doc.sources.get("yard-loop");
    expect(loop?.kind).toBe("synthetic");
    expect(loop?.frame).toBe("ecef");
    expect(telemetryRefOf({ telemetry: { uri: "telemetry.json", count: 2 } })).toEqual({
      uri: "telemetry.json",
      count: 2,
    });
    expect(telemetryRefOf({})).toBeNull();
  });

  it("fills defaults, keeps what is well formed and says what it dropped", () => {
    const doc = parseTelemetry({
      format: "hexapod.telemetry",
      version: 1,
      sources: {
        robot: { kind: "sse", url: "https://bridge.example/r1", frame: "geodetic" },
        ws: { kind: "websocket", url: "wss://bridge.example/fleet" },
        bad: { kind: "carrier-pigeon" },
        nopath: { kind: "synthetic", path: { kind: "circle" } },
      },
      bindings: [
        { instance: 3, source: "robot" },
        { instance: 4, source: "ws", stream: "R2", stale: "freeze", latencyMs: 50 },
        { instance: 3, source: "ws" },
        { instance: 5, source: "bad" },
        { instance: 0, source: "robot" },
        {
          instance: 6,
          source: "robot",
          rest: { frame: "geodetic", position: [-93.4, 44.9, 280], headingDeg: 90 },
        },
        { instance: 7, source: "robot", rest: { position: [1, 2] } },
      ],
    });
    if (!doc) throw new Error("not read");
    expect([...doc.sources.keys()]).toEqual(["robot", "ws"]);
    expect(doc.bindings.map((b) => b.instance)).toEqual([3, 4, 6]);
    const [first, second, third] = doc.bindings;
    expect(first).toMatchObject({
      latencyMs: 250,
      extrapolateMs: 1000,
      staleMs: 3000,
      stale: "rest",
      fadeMs: 2000,
    });
    expect(second).toMatchObject({ stream: "R2", stale: "freeze", latencyMs: 50 });
    expect(third?.rest?.frame).toBe("geodetic");
    close(third?.rest?.orientation ?? [], [0, 0, 0, 1], 1e-12);
    expect(doc.issues).toHaveLength(6);
    expect(parseTelemetry({ format: "hexapod.instances", version: 1 })).toBeNull();
  });
});

describe("the wire format", () => {
  it("reads one reading, an array, or samples; ISO or millisecond times", () => {
    const one = parsePoseMessages(
      { t: 1000, position: [1, 2, 3], orientation: [0, 0, 1, 1] },
      "scan",
    );
    expect(one).toHaveLength(1);
    close(one[0]?.orientation ?? [], [0, 0, Math.SQRT1_2, Math.SQRT1_2], 1e-12);
    const many = parsePoseMessages(
      {
        samples: [
          { t: "2026-10-02T12:00:00.500Z", position: [-93.4, 44.9, 280], headingDeg: 0 },
          { t: 5, position: [1, 2] },
          { position: [1, 2, 3] },
          { t: 6, position: [0, 0, 0], frame: "mars" },
          { t: 7, position: [0, 0, 0], stream: "R1", frame: "scan" },
        ],
      },
      "geodetic",
    );
    expect(many.map((s) => [s.t, s.frame, s.stream])).toEqual([
      [Date.parse("2026-10-02T12:00:00.500Z"), "geodetic", undefined],
      [7, "scan", "R1"],
    ]);
    // Heading 0 is north: yaw 90° in the level frame.
    close(many[0]?.orientation ?? [], quatFromYaw(Math.PI / 2), 1e-12);
    expect(parsePoseMessages([{ t: 1, position: [0, 0, 0] }], "scan")).toHaveLength(1);
  });

  it("an ECEF heading is the level frame's at the position", () => {
    const scan = scanFrame(YARD_ROOT);
    const local = { position: [16, 12, 0] as Vec3, orientation: quatFromYaw(Math.PI / 2) };
    const ecef = poseFromScan(local, "ecef", scan);
    if (!ecef) throw new Error("no pose");
    const [reading] = parsePoseMessages({ t: 0, position: ecef.position, headingDeg: 0 }, "ecef");
    // The yard's frame is level at its centre: north in ENU is +y in the scan.
    const dot = (a: ArrayLike<number>, b: ArrayLike<number>): number =>
      Math.abs(
        (a[0] ?? 0) * (b[0] ?? 0) +
          (a[1] ?? 0) * (b[1] ?? 0) +
          (a[2] ?? 0) * (b[2] ?? 0) +
          (a[3] ?? 0) * (b[3] ?? 0),
      );
    expect(1 - dot(reading?.orientation ?? [], ecef.orientation)).toBeLessThan(1e-9);
  });
});

describe("sources", () => {
  const config: SyntheticSourceConfig = {
    kind: "synthetic",
    frame: "scan",
    path: { kind: "circle", centre: [0, 0, 0], radius: 1, periodS: 4 },
    rateHz: 4,
    delayMs: 30,
    jitterMs: 50,
    dropouts: [[2000, 3000]],
  };

  function collect(source: TelemetrySource): [number, number][] {
    const got: [number, number][] = [];
    source.subscribe((s, arrival) => got.push([s.t, arrival]));
    return got;
  }

  it("a synthetic source releases what has arrived, the same at any frame rate", () => {
    const a = new SyntheticSource("a", config, undefined);
    const b = new SyntheticSource("b", config, undefined);
    const gotA = collect(a);
    const gotB = collect(b);
    for (let now = 0; now <= 4000; now += 16) a.pump(now);
    b.pump(1000);
    b.pump(4000);
    expect(gotA).toEqual(gotB);
    // 4 Hz to 4 s, less the second without readings (and the 4 s one, not yet arrived).
    expect(gotA.map(([t]) => t)).toEqual([
      0, 250, 500, 750, 1000, 1250, 1500, 1750, 3000, 3250, 3500, 3750,
    ]);
    for (const [t, arrival] of gotA) {
      expect(arrival - t).toBeGreaterThanOrEqual(30);
      expect(arrival - t).toBeLessThan(80);
    }
  });

  it("can be muted, and replays from a reset or a clock that went back", () => {
    const source = new SyntheticSource("s", config, undefined);
    const got = collect(source);
    source.muted = true;
    source.pump(1000);
    expect(got).toEqual([]);
    source.muted = false;
    source.pump(1600);
    expect(got.map(([t]) => t)).toEqual([1000, 1250, 1500]);
    source.pump(100);
    expect(got.at(-1)?.[0]).toBe(0);
    source.reset();
    source.pump(100);
    expect(got.filter(([t]) => t === 0)).toHaveLength(2);
  });

  it("emits in its frame: ECEF needs the scan's placement", () => {
    const scan = scanFrame(YARD_ROOT);
    const ecef = createTelemetrySource("e", { ...config, frame: "ecef" }, { clock: () => 0, scan });
    const unplaced = createTelemetrySource(
      "u",
      { ...config, frame: "ecef" },
      {
        clock: () => 0,
        scan: undefined,
      },
    );
    const seen: number[][] = [];
    ecef.subscribe((s) => seen.push([...s.position]));
    const none = collect(unplaced);
    ecef.pump?.(500);
    unplaced.pump?.(500);
    expect(none).toEqual([]);
    const want = poseFromScan(pathPose(config.path, 0), "ecef", scan);
    close(seen[0] ?? [], want?.position ?? [], 1e-6);
  });

  it("a stream source reads JSON messages and stamps them on arrival", () => {
    let now = 5000;
    const channel = {
      onmessage: null as ((event: { data: unknown }) => void) | null,
      onerror: null as ((event: unknown) => void) | null,
      closed: false,
      close() {
        this.closed = true;
      },
    };
    const source = new StreamSource(
      "r1",
      "sse",
      "https://x/r1",
      "geodetic",
      () => now,
      () => channel,
    );
    const got: [number, string, number][] = [];
    source.subscribe((s, arrival) => got.push([s.t, s.frame, arrival]));
    channel.onmessage?.({ data: JSON.stringify({ t: 4900, position: [1, 2, 3] }) });
    now = 5100;
    channel.onmessage?.({ data: "not json" });
    channel.onmessage?.({
      data: JSON.stringify([{ t: 5000, position: [1, 2, 3], frame: "scan" }]),
    });
    channel.onerror?.({});
    expect(got).toEqual([
      [4900, "geodetic", 5000],
      [5000, "scan", 5100],
    ]);
    expect(source.errors).toBe(1);
    source.close();
    expect(channel.closed).toBe(true);
  });
});

/** A skin part that records what it was handed. */
class FakeSkin implements TelemetrySkinTarget {
  readonly doc: SkinDoc;
  readonly current = new Map<number, Float64Array | null>();
  writes = 0;
  constructor(doc: SkinDoc) {
    this.doc = doc;
  }
  setInstanceHandles(id: number, handles: ArrayLike<number> | null): boolean {
    this.writes += 1;
    this.current.set(id, handles === null ? null : Float64Array.from(handles));
    return true;
  }
}

class FakeRigid implements TelemetryRigidTarget {
  readonly current = new Map<number, RigidMotion | null>();
  setInstanceMotion(id: number, motion: RigidMotion | null): void {
    this.current.set(id, motion);
  }
}

describe("the telemetry driver", () => {
  const instances = yardInstances();
  const skinDoc = yardSkin();
  const scan = scanFrame(YARD_ROOT);

  function driverFor(doc: TelemetryDoc): {
    driver: TelemetryDriver;
    sources: Map<string, SyntheticSource>;
  } {
    const sources = new Map<string, SyntheticSource>();
    for (const [id, config] of doc.sources) {
      if (config.kind === "synthetic") sources.set(id, new SyntheticSource(id, config, scan));
    }
    return { driver: new TelemetryDriver(doc, instances, scan, sources), sources };
  }

  it("rests every bound object before the first reading", () => {
    const { driver } = driverFor(yardTelemetry());
    const skin = new FakeSkin(skinDoc);
    const rigid = new FakeRigid();
    const tick = driver.tick(0, skin, () => rigid);
    expect(tick).toEqual({ changed: false, moving: 0 });
    expect(driver.statuses.map((s) => s.state)).toEqual(["none", "none"]);
    expect(skin.writes).toBe(0);
    expect(rigid.current.size).toBe(0);
  });

  it("moves the unskinned building through the rigid part along its path (ECEF in)", () => {
    const { driver } = driverFor(yardTelemetry());
    const skin = new FakeSkin(skinDoc);
    const rigid = new FakeRigid();
    for (let now = 0; now <= 8000; now += 100) driver.tick(now, skin, () => rigid);
    const status = driver.status(8);
    expect(status?.via).toBe("rigid");
    expect(status?.state).toBe("live");
    const motion = rigid.current.get(8);
    if (!motion || !status?.rest) throw new Error("not moved");
    // The rest pose is the building's base centre; the motion carries it onto the path.
    close(status.rest.position, [20, 6.22045, -0.113], 1e-9);
    const loop = yardTelemetry().sources.get("yard-loop");
    if (loop?.kind !== "synthetic") throw new Error("no loop");
    const truth = pathPose(loop.path, status.playoutMs / 1000);
    close(applyMotion(motion, status.rest.position), truth.position, 0.01);
    // At t = 0 the path is the rest pose: the motion starts from the identity.
    const start = relativeMotion(status.rest, pathPose(loop.path, 0));
    close(start.translation, [0, 0, 0], 1e-9);
  });

  it("moves the skinned shrub by its constant handle only, exactly the rigid motion", () => {
    const { driver } = driverFor(yardTelemetry());
    const skin = new FakeSkin(skinDoc);
    const rigid = new FakeRigid();
    for (let now = 0; now <= 3000; now += 50) driver.tick(now, skin, () => rigid);
    const status = driver.status(10);
    expect(status?.via).toBe("skin");
    const handles = skin.current.get(10);
    const entry = skinDoc.byInstance.get(10);
    if (!handles || !entry || !status?.pose || !status.rest) throw new Error("not moved");
    expect(handles.slice(12).every((v) => v === 0)).toBe(true);
    const motion = relativeMotion(status.rest, status.pose);
    const learned = new Array<number>(entry.handles - 1).fill(0.37);
    for (const x of [
      [13, 7, 0.3],
      [13.9, 7.8, 1.9],
      [12.4, 6.2, 0.22],
    ] as Vec3[]) {
      const d = skinDisplacement(handles, entry.handles, entry.origin, learned, x);
      const moved = applyMotion(motion, x);
      close([x[0] + d[0], x[1] + d[1], x[2] + d[2]], moved, 1e-9);
    }
    expect(rigid.current.has(10)).toBe(false);
  });

  it("holds, then fades to rest (rest policy) or freezes (freeze policy) when silent", () => {
    const { driver, sources } = driverFor(yardTelemetry());
    const skin = new FakeSkin(skinDoc);
    const rigid = new FakeRigid();
    for (let now = 0; now <= 6000; now += 100) driver.tick(now, skin, () => rigid);
    for (const source of sources.values()) source.muted = true;
    const shrubFrozen = skin.current.get(10);
    for (let now = 6100; now <= 7600; now += 100) driver.tick(now, skin, () => rigid);
    expect(driver.status(8)?.state).toBe("held");
    for (let now = 7700; now <= 9500; now += 100) driver.tick(now, skin, () => rigid);
    expect(driver.status(8)?.state).toBe("stale");
    expect(driver.status(10)?.state).toBe("stale");
    for (let now = 9600; now <= 12_000; now += 100) driver.tick(now, skin, () => rigid);
    expect(driver.status(8)?.state).toBe("rest");
    expect(rigid.current.get(8)).toBeNull();
    // The shrub froze where dead reckoning left it, and stays there.
    const frozen = skin.current.get(10);
    expect(frozen).not.toBeNull();
    expect(frozen).not.toEqual(shrubFrozen);
    const writes = skin.writes;
    driver.tick(20_000, skin, () => rigid);
    expect(skin.writes).toBe(writes);
  });

  it("hands an object to its skin when the skin arrives after the readings", () => {
    const { driver } = driverFor(yardTelemetry());
    const rigid = new FakeRigid();
    for (let now = 0; now <= 2000; now += 100) driver.tick(now, undefined, () => rigid);
    expect(driver.status(10)?.via).toBe("rigid");
    expect(rigid.current.get(10)).not.toBeNull();
    const skin = new FakeSkin(skinDoc);
    driver.tick(2100, skin, () => rigid);
    expect(driver.status(10)?.via).toBe("skin");
    expect(rigid.current.get(10)).toBeNull();
    expect(skin.current.get(10)).not.toBeNull();
  });

  it("resets to rest and replays the same motions from the same clock", () => {
    const { driver } = driverFor(yardTelemetry());
    const skin = new FakeSkin(skinDoc);
    const rigid = new FakeRigid();
    driver.tick(5000, skin, () => rigid);
    const first = rigid.current.get(8);
    driver.reset(skin, () => rigid);
    expect(rigid.current.get(8)).toBeNull();
    expect(skin.current.get(10)).toBeNull();
    driver.tick(5000, skin, () => rigid);
    expect(rigid.current.get(8)).toEqual(first);
  });

  it("claims the bound instances with their ancestors and descendants", () => {
    const { driver } = driverFor(yardTelemetry());
    expect([...driver.claimed].sort((a, b) => a - b)).toEqual([
      8, 10, 90, 91, 92, 93, 94, 95, 96, 97, 98,
    ]);
    expect([...relatedInstances(instances, [97])].sort((a, b) => a - b)).toEqual([10, 97]);
    expect(
      restPoseOf({ ...yardTelemetry().bindings[0]!, instance: 999 }, instances, scan),
    ).toBeNull();
  });
});

describe("the wind and telemetry", () => {
  it("the wind leaves a claimed skin alone, and sways the rest", () => {
    const doc = yardSkin();
    const part = {
      doc,
      materials: new Map([
        [1, { wind: true }],
        [10, { wind: true }],
      ]),
      current: new Map<number, number[] | null>(),
      setInstanceHandles(id: number, handles: ArrayLike<number> | null): boolean {
        this.current.set(id, handles === null ? null : Array.from(handles));
        return true;
      },
    };
    const plant: InstanceTraits = { properties: { vegetation: 0.9 }, behaviour: "in-place" };
    const owner = {};
    const release = claimInstances("yard", owner, new Set([10, 97, 98]));
    expect(isClaimed("yard", 10)).toBe(true);
    expect(isClaimed("other", 10)).toBe(false);
    const driver = new SkinWindDriver(part, () => plant, new SkinWindField(1), claimsOf("yard"));
    for (let k = 0; k < 120; k += 1) driver.tick(k / 60, { strength: 0.1, bearingDeg: 30 });
    expect(part.current.get(1)).not.toBeNull();
    expect(part.current.has(10)).toBe(false);
    expect(driver.claimed(10)).toBe(true);
    // Released, the wind takes it up again.
    release();
    expect(isClaimed("yard", 10)).toBe(false);
    for (let k = 120; k < 240; k += 1) driver.tick(k / 60, { strength: 0.1, bearingDeg: 30 });
    expect(part.current.get(10)).not.toBeNull();
  });
});

describe("the rigid part", () => {
  const instances = yardInstances();
  const leaves = [...yardTiles.keys()].filter((uri) => yardTiles.get(uri)?.leaf);
  const tiles = leaves.map(
    (uri) => new FakeTile(uri, yardTiles.get(uri)?.local ?? new Float32Array(0), yardBake),
  );
  /** Every splat's instance id and local position, in the snapshot's order. */
  const ids: number[] = [];
  const local: number[] = [];
  for (const uri of leaves) {
    const tile = yardTiles.get(uri);
    if (!tile) continue;
    const tileIds = tileInstanceIds(instances, checksumPositions(tile.local));
    if (!tileIds) throw new Error(`${uri} unlisted`);
    ids.push(...tileIds);
    local.push(...tile.local);
  }

  function setup(): {
    rigid: SplatRigidMotion;
    primitive: FakeHookedPrimitive;
    factory: ReturnType<typeof fakeFactory>;
  } {
    const factory = fakeFactory();
    const primitive = new FakeHookedPrimitive();
    primitive.commit(tiles);
    const source = {
      doc: instances,
      idTexture: { ids: true },
      covered: ids.length,
      idsCurrent: true,
      bake: yardBake,
      ids: Uint32Array.from(ids),
      idsVersion: 1,
    };
    const rigid = new SplatRigidMotion(source, factory, new FakeTiledTileset(primitive));
    rigid.sync();
    buildDrawCommand(primitive);
    return { rigid, primitive, factory };
  }

  it("joins the motion chain with its Jacobian, and does nothing until driven", () => {
    const { rigid, primitive } = setup();
    const chain = primitive.vertexMotion ? motionChainOf(primitive) : undefined;
    expect(chain?.parts).toContain(rigid);
    expect(rigidGlsl()).toContain("vec3 splatRigidMotion(uint splatIndex, vec3 position)");
    expect(composeMotionGlsl(["splatRigidMotion"], ["splatRigidJacobian"])).toContain(
      "jacobian += splatRigidJacobian(splatIndex, position);",
    );
    expect(rigid.active).toBe(false);
    const uniforms = buildDrawCommand(primitive);
    expect(uniforms.u_rigidActive?.()).toBe(0);
    expect(uniforms.u_rigidIds?.()).toEqual({ ids: true });
  });

  it("moves exactly the bound instance's splats by the motion, in the baked frame", () => {
    const { rigid, primitive } = setup();
    const motion = relativeMotion(
      { position: [20, 6.22, -0.11], orientation: [0, 0, 0, 1] },
      { position: [21.5, 8, 0.2], orientation: quatFromYaw(0.6) },
    );
    rigid.setInstanceMotion(8, motion);
    rigid.sync();
    expect(rigid.active).toBe(true);
    const members = withDescendants(instances, [8]);
    const l = (r: number, c: number): number => yardBake[c * 4 + r] ?? 0;
    let moved = 0;
    let still = 0;
    for (let i = 0; i < ids.length; i += 5) {
      const slot = rigid.slotOf(ids[i] ?? 0);
      const xb: [number, number, number] = [
        primitive._positions[i * 3] ?? 0,
        primitive._positions[i * 3 + 1] ?? 0,
        primitive._positions[i * 3 + 2] ?? 0,
      ];
      const got = evaluateRigidMotion(rigid.poseData, slot, xb);
      if (!members.has(ids[i] ?? 0)) {
        expect(slot).toBe(0);
        expect(got).toEqual([0, 0, 0]);
        still += 1;
        continue;
      }
      const x: Vec3 = [local[i * 3] ?? 0, local[i * 3 + 1] ?? 0, local[i * 3 + 2] ?? 0];
      const to = applyMotion(motion, x);
      const d = [to[0] - x[0], to[1] - x[1], to[2] - x[2]];
      for (let r = 0; r < 3; r += 1) {
        const want = l(r, 0) * (d[0] ?? 0) + l(r, 1) * (d[1] ?? 0) + l(r, 2) * (d[2] ?? 0);
        expect(Math.abs((got[r] ?? 0) - want)).toBeLessThan(2e-4);
      }
      moved += 1;
    }
    expect(moved).toBeGreaterThan(100);
    expect(still).toBeGreaterThan(1000);
    // Back at rest: the slot goes, the texels are zeros, the part stops acting.
    rigid.setInstanceMotion(8, null);
    rigid.sync();
    expect(rigid.active).toBe(false);
    expect(rigid.slotOf(8)).toBe(0);
    expect(rigid.poseData.every((v) => v === 0)).toBe(true);
  });

  it("gives a driven instance below another its own slot", () => {
    const { rigid } = setup();
    rigid.setInstanceMotion(91, { rotation: [0, 0, 0, 1], translation: [0, 0, 1] });
    rigid.setInstanceMotion(8, { rotation: [0, 0, 0, 1], translation: [1, 0, 0] });
    expect(rigid.slotOf(91)).toBe(1);
    expect([8, 90, 92, 96].map((id) => rigid.slotOf(id))).toEqual([2, 2, 2, 2]);
    expect(rigid.driven.sort((a, b) => a - b)).toEqual([8, 91]);
    rigid.rest();
    expect(rigid.driven).toEqual([]);
  });

  it("uploads poses each change and slots only when what is driven changes", () => {
    const { rigid, factory } = setup();
    const before = factory.made.length;
    rigid.setInstanceMotion(8, { rotation: [0, 0, 0, 1], translation: [1, 0, 0] });
    rigid.sync();
    const slotsMade = factory.made.length - before;
    expect(slotsMade).toBe(1);
    rigid.setInstanceMotion(8, { rotation: [0, 0, 0, 1], translation: [2, 0, 0] });
    rigid.sync();
    expect(factory.made.length - before).toBe(1);
  });
});

describe("sorting what moves", () => {
  it("a rigid group's eye, carried back, gives the moved splats' distances", () => {
    const motion = relativeMotion(
      { position: [0, 0, 0], orientation: [0, 0, 0, 1] },
      { position: [5, 1, 0], orientation: quatFromYaw(Math.PI) },
    );
    // Rows of x' = R x + t for group 1; group 0 unused (zeros).
    const rows = new Float64Array(24);
    const r = [
      applyMotion(motion, [1, 0, 0]),
      applyMotion(motion, [0, 1, 0]),
      applyMotion(motion, [0, 0, 1]),
    ];
    const t = applyMotion(motion, [0, 0, 0]);
    for (let row = 0; row < 3; row += 1) {
      for (let c = 0; c < 3; c += 1) rows[12 + row * 4 + c] = (r[c]?.[row] ?? 0) - (t[row] ?? 0);
      rows[12 + row * 4 + 3] = t[row] ?? 0;
    }
    const eye: [number, number, number] = [10, 0, 2];
    const eyes = groupEyes(rows, eye);
    close(eyes.subarray(0, 3), eye, 1e-12);
    // Two splats of the group: at rest the first is nearer the eye, moved the second is.
    const positions = new Float32Array([1, 0, 0, -1, 0, 0, 20, 0, 0]);
    const groups = new Uint16Array([1, 1, 0]);
    const moved = sortBackToFront(positions, 3, eye, undefined, { groups, eyes });
    const still = sortBackToFront(positions, 3, eye);
    expect(Array.from(still.order)).toEqual([1, 2, 0]);
    // Moved: (1,0,0) → (4,1,0), (−1,0,0) → (6,1,0); the far splat stays at 20.
    expect(Array.from(moved.order)).toEqual([2, 0, 1]);
  });

  it("the rigid part tells the sorter its splats' slots and motions, and forgets at rest", () => {
    const instances = yardInstances();
    const factory = fakeFactory();
    const primitive = new FakeHookedPrimitive();
    const leaves = [...yardTiles.keys()].filter((uri) => yardTiles.get(uri)?.leaf);
    const tiles = leaves.map(
      (uri) => new FakeTile(uri, yardTiles.get(uri)?.local ?? new Float32Array(0), yardBake),
    );
    primitive.commit(tiles);
    const ids: number[] = [];
    for (const uri of leaves) {
      const tile = yardTiles.get(uri);
      ids.push(...(tile ? (tileInstanceIds(instances, checksumPositions(tile.local)) ?? []) : []));
    }
    const rigid = new SplatRigidMotion(
      {
        doc: instances,
        idTexture: {},
        covered: ids.length,
        idsCurrent: true,
        bake: yardBake,
        ids: Uint32Array.from(ids),
        idsVersion: 1,
      },
      factory,
      new FakeTiledTileset(primitive),
    );
    rigid.sync();
    expect(sortMotionOf(primitive)).toBeUndefined();
    rigid.setInstanceMotion(8, { rotation: quatFromYaw(1), translation: [1, 2, 0] });
    rigid.sync();
    const told = sortMotionOf(primitive);
    if (!told) throw new Error("not told");
    const members = withDescendants(instances, [8]);
    for (let i = 0; i < ids.length; i += 11)
      expect(told.groups[i]).toBe(members.has(ids[i] ?? 0) ? 1 : 0);
    // Group 1's rows are I + A_b: a rotation (determinant 1).
    const m = (r: number, c: number): number => told.motions[12 + r * 4 + c] ?? 0;
    const det =
      m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) -
      m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
      m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
    expect(det).toBeCloseTo(1, 5);
    rigid.rest();
    rigid.sync();
    expect(sortMotionOf(primitive)).toBeUndefined();
  });
});
