import { DEG_TO_RAD, EARTH_RADIUS_M, type LonLat } from "@twin/geo";
import { describe, expect, it } from "vitest";

import { MACHINE_CLASSES } from "@/missions/machineClasses";
import {
  capabilityOf,
  classOfModel,
  elevationAt,
  formatSimDuration,
  gradientAt,
  planRehearsal,
  positionAt,
  type CreekSource,
  type ElevationGrid,
  type ElevationSource,
  type RehearsalInput,
  type RehearsalZoneInput,
} from "@/missions/rehearsal";

const origin = { longitude: -120.04, latitude: 34.698 };
const kx = Math.cos(origin.latitude * DEG_TO_RAD) * DEG_TO_RAD * EARTH_RADIUS_M;
const ky = DEG_TO_RAD * EARTH_RADIUS_M;

/** A point `x` metres east and `y` metres north of the origin. */
function at(x: number, y: number): LonLat {
  return { longitude: origin.longitude + x / kx, latitude: origin.latitude + y / ky };
}

/**
 * 10 m cells over 1 km x 1 km around the origin: flat west of it, rising 30 % to the east
 * east of it (a ramp under half of the zone).
 */
function rampGrid(): ElevationGrid {
  const columns = 100;
  const rows = 100;
  const values = new Float32Array(columns * rows);
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < columns; col++) {
      const x = (col + 0.5) * 10 - 500;
      values[row * columns + col] = 300 + (x > 0 ? 0.3 * x : 0);
    }
  }
  const sw = at(-500, -500);
  const ne = at(500, 500);
  return {
    west: sw.longitude,
    south: sw.latitude,
    east: ne.longitude,
    north: ne.latitude,
    columns,
    rows,
    values,
  };
}

const elevation: ElevationSource = {
  grid: rampGrid(),
  label: "USGS 3DEP 1/3″ bare-earth DEM, 10 m",
  ref: "raster 8c5becbb",
};

const creeks: CreekSource = {
  label: "USGS NHD flowlines (1:24,000)",
  lines: [
    // 10 m east of the flat north-running lane at x = -200.
    {
      id: "nhd-1",
      name: "Figueroa Creek",
      kind: "intermittent",
      coordinates: [at(-190, -400), at(-190, 400)],
    },
    // Next to the ramp lane, but ephemeral: not a creek unless asked.
    {
      id: "nhd-2",
      name: null,
      kind: "ephemeral",
      coordinates: [at(205, -400), at(205, 400)],
    },
  ],
};

function northLane(x: number) {
  return { points: [at(x, -300), at(x, 300)], pass: 0 };
}

function zone(overrides: Partial<RehearsalZoneInput> = {}): RehearsalZoneInput {
  return {
    zoneId: "A-01",
    label: "A-01",
    lanes: [northLane(-200), northLane(200)],
    machineIds: ["TR-04"],
    classId: "mower",
    ...overrides,
  };
}

const machines = [
  { id: "TR-04", name: "TR-04 Kestrel", model: "Ranger 2 · mower" },
  { id: "TR-12", name: "TR-12 Pipit", model: "Ranger 2 · mower" },
  { id: "TR-18", name: "TR-18 Shrike", model: "Hawk 3 · drone" },
  { id: "TR-99", name: "TR-99", model: "Mystery 9" },
];

function input(overrides: Partial<RehearsalInput> = {}): RehearsalInput {
  return { zones: [zone()], machines, elevation, creeks, ...overrides };
}

describe("the ground a rehearsal reads", () => {
  it("interpolates heights and takes the slope over a cell either side", () => {
    const grid = rampGrid();
    const flat = at(-250, 0);
    const ramp = at(250, 0);
    expect(elevationAt(grid, flat.longitude, flat.latitude)).toBeCloseTo(300, 3);
    expect(elevationAt(grid, ramp.longitude, ramp.latitude)).toBeCloseTo(375, 0);
    const slope = gradientAt(grid, ramp.longitude, ramp.latitude);
    expect(slope?.east).toBeCloseTo(0.3, 2);
    expect(slope?.north).toBeCloseTo(0, 3);
    expect(gradientAt(grid, flat.longitude, flat.latitude)?.east).toBeCloseTo(0, 3);
    const outside = at(900, 0);
    expect(gradientAt(grid, outside.longitude, outside.latitude)).toBeNull();
    // At the edge of the data the difference is one-sided, not lost.
    const values = Float32Array.from(grid.values);
    for (let row = 0; row < grid.rows; row++) values[row * grid.columns + 80] = Number.NaN;
    const edge = at(295, 0);
    expect(gradientAt({ ...grid, values }, edge.longitude, edge.latitude)?.east).toBeCloseTo(
      0.3,
      1,
    );
  });

  it("names the class a machine's model ends with, and its assumed limits", () => {
    expect(classOfModel("Ranger 2 · mower")).toBe("mower");
    expect(classOfModel("Drill 1 · seeder")).toBe("seeder");
    expect(classOfModel("Mystery 9")).toBeNull();
    const mower = capabilityOf("mower", "its model");
    expect(mower).toMatchObject({
      kind: "ground",
      speedMps: MACHINE_CLASSES.mower.speedMps,
      maxSlopePct: MACHINE_CLASSES.mower.maxSlopePct,
      creekBufferM: MACHINE_CLASSES.mower.creekBufferM,
    });
    expect(mower.label).toContain("(assumed)");
    expect(capabilityOf("drone", "its model")).toMatchObject({
      kind: "air",
      maxSlopePct: null,
      creekBufferM: null,
    });
  });
});

describe("planRehearsal", () => {
  it("raises a cross-slope hand on a lane running north across the ramp, not on the flat", () => {
    const result = planRehearsal(input({ creeks: null }));
    const slopeHands = result.hands.filter((h) => h.kind !== "creek-buffer");
    expect(slopeHands).toHaveLength(1);
    const hand = slopeHands[0];
    expect(hand).toMatchObject({ kind: "cross-slope", zoneId: "A-01", index: 1, limit: 25 });
    expect(hand?.value).toBeCloseTo(30, 0);
    expect(hand?.label).toBe("30 % cross-slope · limit 25 % (assumed)");
    expect(hand?.rule).toContain("A-01 lane 2");
    expect(hand?.rule).toContain("Assumed limit for a robotic brush mower: 25 %.");
    expect(hand?.source).toBe("Slope: USGS 3DEP 1/3″ bare-earth DEM, 10 m, raster 8c5becbb");
    expect(result.summary).toContain("1 of 2 lanes over 25 % cross-slope (A-01 lane 2)");
    expect(result.sources[0]).toContain("raster 8c5becbb");
  });

  it("judges a lane running east up the ramp by its grade, one hand per lane at its worst", () => {
    const east = { points: [at(50, 100), at(450, 100)], pass: 0 };
    const result = planRehearsal(input({ zones: [zone({ lanes: [east] })], creeks: null }));
    expect(result.hands.map((h) => h.kind)).toEqual(["grade"]);
    expect(result.hands[0]?.value).toBeCloseTo(30, 0);
    // The worst point is on the lane, and the machine reaches it while driving it.
    const lane = result.lanes[0];
    expect(lane && result.hands[0]!.t >= lane.t0 && result.hands[0]!.t <= lane.t1).toBe(true);
    expect(result.summary).toContain("1 lane over 25 % grade (A-01 lane 1)");
  });

  it("raises exactly one creek hand for the lane 10 m from a creek with a 15 m buffer", () => {
    const result = planRehearsal(input({ elevation: null }));
    const creek = result.hands.filter((h) => h.kind === "creek-buffer");
    expect(creek).toHaveLength(1);
    expect(creek[0]).toMatchObject({ index: 0, limit: 15, creekId: "nhd-1" });
    expect(creek[0]?.value).toBeCloseTo(10, 0);
    expect(creek[0]?.label).toBe("10 m from Figueroa Creek · buffer 15 m (assumed)");
    expect(result.summary).toContain(
      "1 of 2 lanes inside the 15 m buffer of Figueroa Creek (A-01 lane 1)",
    );
    expect(result.skipped.join(" ")).toContain("Ephemeral drainages");
    // Asked to, the ephemeral channel beside the ramp lane counts too.
    const withEphemeral = planRehearsal(input({ elevation: null, includeEphemeral: true }));
    expect(withEphemeral.hands.filter((h) => h.kind === "creek-buffer")).toHaveLength(2);
  });

  it("says a lane crosses a creek when it runs over the line", () => {
    const across: CreekSource = {
      label: creeks.label,
      lines: [
        {
          id: "nhd-3",
          name: "Figueroa Creek",
          kind: "perennial",
          coordinates: [at(-300, 0), at(-100, 0)],
        },
      ],
    };
    const result = planRehearsal(input({ elevation: null, creeks: across }));
    const hand = result.hands.find((h) => h.kind === "creek-buffer");
    expect(hand?.label).toBe("Crosses Figueroa Creek · buffer 15 m (assumed)");
    expect(hand?.rule).toBe(
      "A-01 lane 1 crosses Figueroa Creek. Assumed buffer for a robotic brush mower: 15 m from creeks.",
    );
    expect(hand?.source).toBe("Creek: USGS NHD flowlines (1:24,000), flowline nhd-3 (perennial)");
  });

  it("says what it could not check", () => {
    const result = planRehearsal(
      input({
        elevation: null,
        creeks: null,
        elevationMissing: "Slope not checked: no terrain for this ground.",
      }),
    );
    expect(result.hands).toEqual([]);
    expect(result.skipped).toContain("Slope not checked: no terrain for this ground.");
    expect(result.skipped.join(" ")).toContain("Creek buffer not checked");
    expect(result.summary.join(" ")).not.toContain("No hands raised");
  });

  it("skips the ground rules for a machine that flies, and says so", () => {
    const result = planRehearsal(input({ zones: [zone({ machineIds: ["TR-18"] })] }));
    expect(result.hands).toEqual([]);
    expect(result.machines[0]?.capability.kind).toBe("air");
    expect(result.skipped.join(" ")).toContain("TR-18 is a Mapping drone");
    expect(result.skipped.join(" ")).toContain("it flies");
  });

  it("rehearses an unknown model with the plan's class, or flags the default", () => {
    const planned = planRehearsal(input({ zones: [zone({ machineIds: ["TR-99"] })] }));
    expect(planned.machines[0]?.capability).toMatchObject({ classId: "mower", defaulted: false });
    expect(planned.skipped.join(" ")).toContain("the plan's class for its zone");
    const guessed = planRehearsal(
      input({ zones: [zone({ machineIds: ["TR-99"], classId: null })] }),
    );
    expect(guessed.machines[0]?.capability.defaulted).toBe(true);
    expect(guessed.skipped.join(" ")).toContain("a default assumption");
  });

  it("splits a zone's lanes round-robin and times lanes, turns and moves", () => {
    const lanes = [northLane(-300), northLane(-280), northLane(-260)];
    const result = planRehearsal(
      input({
        zones: [zone({ lanes, machineIds: ["TR-04", "TR-12"] })],
        elevation: null,
        creeks: null,
      }),
    );
    expect(result.lanes.map((l) => [l.index, l.machineId])).toEqual([
      [0, "TR-04"],
      [1, "TR-12"],
      [2, "TR-04"],
    ]);
    const speed = MACHINE_CLASSES.mower.speedMps;
    const lane = 600 / speed;
    const first = result.tracks.find((t) => t.machine.id === "TR-04");
    // Lane 1 (600 m), a turn and the 40 m move across to lane 3 (from its north end to its
    // south end: 600 m back), then lane 3.
    const move = Math.hypot(40, 600) / speed + MACHINE_CLASSES.mower.turnS;
    expect(first?.endS).toBeCloseTo(lane + move + lane, 0);
    expect(result.durationS).toBeCloseTo(first?.endS ?? 0, 6);
    const mid = positionAt(first!, lane / 2);
    expect(mid?.lane).toEqual({ zoneId: "A-01", index: 0 });
    expect(mid?.at.latitude).toBeCloseTo(origin.latitude, 5);
    expect(positionAt(first!, lane + 1)?.lane).toBeNull();
  });

  it("drives a zone without machines with one assumed generic machine per class", () => {
    const result = planRehearsal(
      input({
        zones: [
          zone({ zoneId: "L-1", label: "Oak Bench Ranch", machineIds: [] }),
          zone({ zoneId: "L-2", label: "Fence", machineIds: [], lanes: [northLane(-100)] }),
        ],
        creeks: null,
      }),
    );
    expect(result.machines).toHaveLength(1);
    expect(result.machines[0]).toMatchObject({
      id: "SIM-1",
      generic: true,
      name: "Assumed generic ground machine (simulated)",
    });
    expect(result.lanes.every((l) => l.machineId === "SIM-1")).toBe(true);
    expect(result.skipped[0]).toContain(
      "one assumed generic machine (simulated) per machine class",
    );
    expect(result.summary[0]).toMatch(/^1 simulated machine drove 3 drawn lanes \(1\.8 km\) in /);
  });

  it("follows a mapped line as one lane", () => {
    const line = { points: [at(-200, -300), at(-200, 0), at(-100, 200)], pass: 0 };
    const result = planRehearsal(
      input({ zones: [zone({ lanes: [line] })], elevation: null, creeks: null }),
    );
    expect(result.lanes).toHaveLength(1);
    expect(result.lanes[0]?.lengthM).toBeCloseTo(300 + Math.hypot(100, 200), 0);
    expect(result.tracks[0]?.segments).toHaveLength(2);
  });

  it("formats the simulated clock", () => {
    expect(formatSimDuration(20)).toBe("under a minute");
    expect(formatSimDuration(45 * 60)).toBe("45 min");
    expect(formatSimDuration(2 * 3600 + 14 * 60)).toBe("2 h 14 min");
    expect(formatSimDuration(3 * 3600)).toBe("3 h");
  });
});
