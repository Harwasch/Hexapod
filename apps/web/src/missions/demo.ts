/**
 * Demo mission data for the demo ranch: a real place on California's Central Coast (oak
 * savanna along Figueroa Creek, near Santa Ynez) with a simulated fleet working it.
 *
 * The fleet, zones, plans and work log are SIMULATED (the UI says so). Every number is
 * computed from the ranch file (`demoRanchData.ts`, apps/api/app/seed/demo_ranch.json) rather
 * than typed in: a zone's acres are its footprint's measured acres, a plan's acres are the sum
 * of its zones', and each working machine stands on the coverage passes the plan preview draws
 * for its zone. The ids (Z-14/Z-21/Z-08, TR-04…TR-18, thistle, survey3d, seed, fuelbreak) are
 * what plans, tests and the e2e suite hold on to; the creek plan was the pipeline plan.
 */

import type { Position } from "geojson";

import { bearing, coverageLanes, destination, haversineDistance, toAcres } from "@twin/geo";

import {
  BUILTIN_RANCH_ID,
  DEMO_RANCH,
  DEMO_RANCH_PROJECT_ID,
  DEMO_RANCH_SLUG,
  sourceText,
  type DemoRanch,
  type DemoZoneId,
} from "./demoRanchData";
import {
  eyeOf,
  frameGround,
  framingContains,
  projectGroundPoints,
  viewportOptions,
  type EyePose,
  type Framing,
  type Viewport,
} from "./framing";
import { MACHINE_CLASSES } from "./machineClasses";
import { PREVIEW_MAX_PASSES, PREVIEW_SWATH_M } from "./preview";
import type { LonLat, Machine, MissionProvider, Plan, Project, WorkLogRow, Zone } from "./types";

/**
 * The assumed machine classes the simulated work log follows (the canonical table is
 * apps/api/app/services/machine_classes.json): working width and ground speed, cutting
 * `FIELD_EFFICIENCY` of the time and turning the rest. A 2 m mower at 1.4 m/s covers about
 * two acres an hour, so a shift is a dozen acres, never two hundred.
 */
// The same assumed classes the planner sizes work with (machine_classes.json), so the demo's
// rates and a drafted plan's hours never disagree.
const ASSUMED = {
  mow: MACHINE_CLASSES.mower,
  survey: MACHINE_CLASSES.survey,
  seed: MACHINE_CLASSES.seeder,
} as const;
const FIELD_EFFICIENCY = 0.8;

export function assumedAcresPerHour(kind: keyof typeof ASSUMED): number {
  const { swathM, speedMps } = ASSUMED[kind];
  return toAcres(swathM * speedMps * 3600) * FIELD_EFFICIENCY;
}

/** Where along its zone's coverage passes each working machine is, as a share of the work. */
const PROGRESS = { "Z-14": 0.58, "Z-21": 0.12, "Z-08": 0.34 } as const;

/** Examples the plan composer offers on the demo ranch: work this place actually has. */
export const DEMO_EXAMPLES = [
  "Mow the star thistle on the west grassland by Friday",
  "Survey Figueroa Creek for blackberry every month",
  "3D scan this field into a gaussian splat",
];

function clock(hours: number): string {
  const minutes = Math.round(hours * 60);
  return `${Math.floor(minutes / 60)}:${String(minutes % 60).padStart(2, "0")}`;
}

function hoursOf(text: string): number {
  const [h, m] = text.split(":").map(Number);
  return (h ?? 0) + (m ?? 0) / 60;
}

function miles(meters: number): string {
  return (meters / 1609.344).toFixed(1);
}

/**
 * The point `fraction` of the way through the coverage passes' working length (the lanes
 * inside the zone, holes left out; the turns between them are not work), with the path
 * vertices before it and the direction of travel there. The passes are the ones the plan
 * preview draws (`footprintLanes` in MissionManager.showPlan), so a machine never stands in a
 * hole the preview leaves open.
 */
export function alongPasses(
  polygon: Position[][],
  fraction: number,
): { position: LonLat; track: LonLat[]; headingDeg: number } {
  const path = coverageLanes(polygon, {
    swathM: PREVIEW_SWATH_M,
    maxPasses: PREVIEW_MAX_PASSES,
  }).lanes.flatMap((lane) => [lane.a, lane.b]);
  // Spans are (start, end) pairs: segment i→i+1 is work when i is even, a turn when it is odd.
  let total = 0;
  for (let i = 0; i + 1 < path.length; i += 2) {
    const a = path[i];
    const b = path[i + 1];
    if (a && b) total += haversineDistance(a, b);
  }
  let left = total * fraction;
  for (let i = 0; i + 1 < path.length; i += 2) {
    const a = path[i];
    const b = path[i + 1];
    if (!a || !b) continue;
    const length = haversineDistance(a, b);
    if (left <= length || i + 3 >= path.length) {
      const headingDeg = Math.round(bearing(a, b));
      const position = destination(a, bearing(a, b), Math.min(left, length));
      return {
        position,
        track: [...path.slice(Math.max(0, i - 3), i + 1), position],
        headingDeg,
      };
    }
    left -= length;
  }
  const only = path[0] ?? { longitude: 0, latitude: 0 };
  return { position: only, track: [only], headingDeg: 0 };
}

/** The simulated project on the demo ranch, built from the ranch file. */
export function buildDemoProject(ranch: DemoRanch): Project {
  const z = ranch.derived.zones;
  const acres = (id: DemoZoneId) => Math.round(toAcres(z[id].areaM2));
  const ac14 = acres("Z-14");
  const ac21 = acres("Z-21");
  const ac08 = acres("Z-08");
  const siteAcres = Math.round(toAcres(ranch.derived.siteAreaM2));
  const strip = ranch.features.fence.stripWidthM;
  const corridor = ranch.features.creek.corridorWidthM;
  const creekName = ranch.features.creek.name ?? "the creek";
  const creekMiles = miles(ranch.derived.creekLengthM);
  const fenceMiles = miles(ranch.derived.fenceLengthM);
  const attribution = (id: DemoZoneId) => `Outline: ${z[id].basis}. ${sourceText(z[id].source)}`;
  const cut14 = Math.round(PROGRESS["Z-14"] * ac14);
  const cut21 = Math.round(PROGRESS["Z-21"] * ac21);
  const thistleAcres = ac14 + ac21;
  const thistlePct = Math.round(((cut14 + cut21) / thistleAcres) * 100);

  const mowRate = assumedAcresPerHour("mow");
  const surveyRate = assumedAcresPerHour("survey");
  const seedRate = assumedAcresPerHour("seed");

  const tr04 = alongPasses(z["Z-14"].footprint.coordinates, PROGRESS["Z-14"]);
  const tr12 = alongPasses(z["Z-21"].footprint.coordinates, PROGRESS["Z-21"]);
  const tr18 = alongPasses(z["Z-08"].footprint.coordinates, PROGRESS["Z-08"]);
  const base1 = ranch.bases["Base 1"] ?? ranch.derived.centroid;
  const base2 = ranch.bases["Base 2"] ?? ranch.derived.centroid;
  // Beside Base 1 rather than on it, so the two markers there do not stack.
  const holding = destination(base1, 90, 60);

  const shiftAcres = (shift: string) => Math.round(hoursOf(shift) * mowRate);
  const machines: Machine[] = [
    {
      id: "TR-04",
      name: "TR-04 Kestrel",
      model: "Ranger 2 · mower",
      task: "Mowing",
      status: "working",
      position: tr04.position,
      headingDeg: tr04.headingDeg,
      batteryPct: 82,
      acresDone: shiftAcres("6:14"),
      shift: "6:14",
      zoneLabel: "Z-14 west grassland",
      primaryAction: "Pause",
      track: tr04.track,
    },
    {
      id: "TR-12",
      name: "TR-12 Pipit",
      model: "Ranger 2 · mower",
      task: "Mowing",
      status: "working",
      position: tr12.position,
      headingDeg: tr12.headingDeg,
      batteryPct: 67,
      acresDone: shiftAcres("4:02"),
      shift: "4:02",
      zoneLabel: "Z-21 fence line",
      primaryAction: "Pause",
    },
    {
      id: "TR-07",
      name: "TR-07 Harrier",
      model: "Ranger 2 · mower",
      task: "Service due",
      status: "attention",
      position: holding,
      headingDeg: 128,
      batteryPct: 41,
      acresDone: shiftAcres("2:40"),
      shift: "2:40",
      zoneLabel: "Holding",
      primaryAction: "Recall",
    },
    {
      id: "TR-18",
      name: "TR-18 Shrike",
      model: "Scout 1 · survey",
      task: "Low signal",
      status: "attention",
      position: tr18.position,
      headingDeg: tr18.headingDeg,
      batteryPct: 54,
      acresDone: null,
      shift: "5:31",
      zoneLabel: "Z-08 creek draw",
      primaryAction: "Retry link",
    },
    {
      id: "TR-11",
      name: "TR-11 Vireo",
      model: "Scout 1 · survey",
      task: "Docked",
      status: "idle",
      position: base2,
      headingDeg: 0,
      batteryPct: 96,
      acresDone: 0,
      shift: "0:00",
      zoneLabel: "Base 2",
      primaryAction: "Deploy",
    },
    {
      id: "TR-15",
      name: "TR-15 Towhee",
      model: "Drill 1 · seeder",
      task: "Loading seed",
      status: "idle",
      position: base1,
      headingDeg: 90,
      batteryPct: 88,
      acresDone: 0,
      shift: "1:18",
      zoneLabel: "Base 1",
      primaryAction: "Deploy",
    },
  ];

  const zones: Zone[] = [
    {
      id: "Z-14",
      name: "Z-14 West grassland",
      short: "mowing 58%",
      tone: "teal",
      treated: true,
      acres: ac14,
      task: "Mow yellow star thistle",
      machines: "TR-04, TR-12",
      progressPct: 58,
      note: `${cut14} ac cut. Second pass starts Thursday 06:00.`,
      anchor: z["Z-14"].anchor,
      footprint: z["Z-14"].footprint,
      attribution: attribution("Z-14"),
    },
    {
      id: "Z-21",
      name: "Z-21 Fence line",
      short: "buffer queued",
      tone: "amber",
      treated: false,
      acres: ac21,
      task: "Buffer + mow queued",
      machines: "TR-12 (queued)",
      progressPct: 12,
      note: `Agent redrew the ${strip} m fence strip. Waiting for your approval.`,
      anchor: z["Z-21"].anchor,
      footprint: z["Z-21"].footprint,
      attribution: attribution("Z-21"),
    },
    {
      id: "Z-08",
      name: "Z-08 Creek draw",
      short: "survey sweep",
      tone: "blue",
      treated: false,
      acres: ac08,
      task: "Transect survey",
      machines: "TR-18",
      progressPct: 34,
      note: "Imagery gap in the north half of the draw, where TR-18's link is weak.",
      anchor: z["Z-08"].anchor,
      footprint: z["Z-08"].footprint,
      attribution: attribution("Z-08"),
    },
  ];

  const plans: Plan[] = [
    {
      id: "thistle",
      title: "Mow yellow star thistle on the west grassland",
      state: "Behind",
      status: "warn",
      meta: `Treatment · Z-14, Z-21 · ${thistleAcres.toLocaleString()} acres`,
      progressPct: thistlePct,
      progressLabel: `${thistlePct}%`,
      objective:
        "Mow yellow star thistle on the west grassland and along the fence before it sets seed, then hold the ground with two passes each season until cover drops under 10%.",
      stats: [
        { value: String(cut14 + cut21), label: "Acres cut" },
        { value: "2", label: "Machines" },
        { value: "11 d", label: "Behind plan", accent: true },
      ],
      facts: [
        { k: "Cadence", v: "Two passes per season, spring and late summer" },
        { k: "Next run", v: "Thursday 06:00 · TR-04, TR-12" },
        { k: "Area", v: "Z-14 west grassland, Z-21 fence line" },
        { k: "Target", v: "Cover under 10% by 2028" },
      ],
      agentNote: "Revised mow sequence recovers 11 days. Waiting on your approval.",
      action: "Pause plan",
      span: "Mar 2 – Nov 20, 2026",
      start: "Mar 2",
      end: "Nov 20, 2026",
      todayPct: 65,
      ongoing: false,
      zoneIds: ["Z-14", "Z-21"],
    },
    {
      id: "creek",
      title: "Clear Arundo and blackberry from the creek draw",
      state: "On track",
      status: "ok",
      meta: `Recurring · ${creekMiles} mi of ${creekName} · monthly`,
      progressPct: 82,
      progressLabel: "9 of 11 runs",
      objective: `Walk the ${creekName} corridor monthly with the survey rover, flag giant reed (Arundo) and Himalayan blackberry stands and fresh bank erosion, and send each flag to the hand crew for cutting. Machines stay out of the channel.`,
      stats: [
        { value: creekMiles, label: "Miles" },
        { value: "9", label: "Runs done" },
        { value: "3", label: "Open flags", accent: true },
      ],
      facts: [
        { k: "Cadence", v: "Monthly, first dry day" },
        { k: "Next run", v: "First dry day next month · TR-18 Shrike" },
        { k: "Area", v: `Z-08, a ${corridor} m corridor along ${creekName}` },
        { k: "Output", v: "Flags with photos to the crew folder" },
      ],
      agentNote: `Two blackberry flags from the last run are still open, at mile ${miles(
        0.35 * ranch.derived.creekLengthM,
      )} and mile ${miles(0.8 * ranch.derived.creekLengthM)}.`,
      action: "Pause plan",
      span: "Monthly since Jan 2025 · no end date",
      start: "Jan 2025",
      end: "Ongoing",
      todayPct: 0,
      ongoing: true,
      zoneIds: ["Z-08"],
    },
    {
      id: "survey3d",
      title: "Collect a detailed 3D survey of the ranch",
      state: "Running",
      status: "run",
      meta: `One-off · ${siteAcres.toLocaleString()} acres · lidar + photogrammetry`,
      progressPct: 62,
      progressLabel: "62%",
      objective:
        "Build a terrain and canopy model of the whole demo area for oak canopy counts, fuel load estimates, creek-bank change detection and drainage planning.",
      stats: [
        { value: String(Math.round(0.62 * siteAcres)), label: "Acres flown" },
        { value: "2 cm", label: "Resolution" },
        { value: "6 d", label: "Remaining", accent: true },
      ],
      facts: [
        { k: "Cadence", v: "One-off, repeat every 3 years" },
        { k: "Next run", v: "Tomorrow 07:15 · TR-11 Vireo" },
        { k: "Area", v: "Whole demo area, 12 survey blocks" },
        { k: "Output", v: "Point cloud, DSM, canopy height model" },
      ],
      agentNote: "Blocks 9 and 10 need a re-fly. Cloud shadow exceeded tolerance.",
      action: "Pause plan",
      span: "Aug 24 – Sep 30, 2026",
      start: "Aug 24",
      end: "Sep 30, 2026",
      todayPct: 62,
      ongoing: false,
      zoneIds: ["Z-14", "Z-21", "Z-08"],
    },
    {
      id: "seed",
      title: "Drill native seed after the first rain",
      state: "Queued",
      status: "idle",
      meta: `Seasonal window · ${thistleAcres.toLocaleString()} acres · Sep 23 – Oct 5`,
      progressPct: 0,
      progressLabel: "Not started",
      objective:
        "Drill a native perennial mix (purple needlegrass, blue wildrye) into the mown ground as soon as the first soaking rain lands, before the window closes in early October.",
      stats: [
        { value: String(thistleAcres), label: "Acres" },
        { value: "1", label: "Machine" },
        { value: "13 d", label: "Window left", accent: true },
      ],
      facts: [
        { k: "Cadence", v: "Annual, weather-triggered" },
        { k: "Trigger", v: "First 15 mm rain event" },
        { k: "Area", v: "Z-14 west grassland, Z-21 fence line" },
        { k: "Machine", v: "TR-15 Towhee · Drill 1" },
      ],
      agentNote:
        "Forecast shows no qualifying rain in the next 10 days. Seed is loaded and staged at Base 1.",
      action: "Run now",
      span: "Sep 23 – Oct 5, 2026",
      start: "Sep 23",
      end: "Oct 5, 2026",
      todayPct: 0,
      ongoing: false,
      zoneIds: ["Z-14", "Z-21"],
    },
    {
      id: "fuelbreak",
      title: "Inspect the fence line and keep its fuel break mown",
      state: "On track",
      status: "ok",
      meta: `Recurring · ${fenceMiles} mi · quarterly`,
      progressPct: 74,
      progressLabel: "74%",
      objective: `Each quarter, mow a ${strip} m break inside the east, north and west fence to slow a grass fire, and flag down wire, leaning posts and washouts for the fence crew.`,
      stats: [
        { value: fenceMiles, label: "Miles" },
        { value: `${strip} m`, label: "Width" },
        { value: "Q4", label: "Next cycle", accent: true },
      ],
      facts: [
        { k: "Cadence", v: "Quarterly" },
        { k: "Next run", v: "Late October · TR-04 Kestrel" },
        { k: "Area", v: `Z-21, ${strip} m inside the fence line` },
        { k: "Standard", v: "Under 10 cm residual height" },
      ],
      agentNote: "Last cycle finished two days early. One leaning post flagged on the north fence.",
      action: "Pause plan",
      span: "Quarterly since Apr 2024 · no end date",
      start: "Apr 2024",
      end: "Ongoing",
      todayPct: 0,
      ongoing: true,
      zoneIds: ["Z-21"],
    },
  ];

  // The log is the last shifts, at the assumed class rates; a zone's progress is its season.
  // Z-21's rows are all of its 12% and the Z-08 transect is its 34%; Z-14's rows are the last
  // few of many.
  const mowRow = (
    date: string,
    machine: string,
    task: string,
    hours: string,
    rate: string,
    zone: string,
  ): WorkLogRow => ({
    date,
    machine,
    task,
    acres: Math.round(hoursOf(hours) * mowRate),
    hours,
    rate,
    zone,
    tone: "teal",
  });
  const half21 = Math.ceil(cut21 / 2);
  const transect = Math.round(PROGRESS["Z-08"] * ac08);
  const seedTrial = "1:45";
  const workLog: WorkLogRow[] = [
    mowRow("09-12", "TR-04", "Mow pass 2", "5:40", "0.38", "Z-14 west grassland"),
    {
      date: "09-12",
      machine: "TR-12",
      task: "Mow pass 1",
      acres: half21,
      hours: clock(half21 / mowRate),
      rate: "0.44",
      zone: "Z-21 fence line",
      tone: "teal",
    },
    {
      date: "09-11",
      machine: "TR-18",
      task: "Transect survey",
      acres: transect,
      hours: clock(transect / surveyRate),
      rate: "—",
      zone: "Z-08 creek draw",
      tone: "blue",
    },
    mowRow("09-10", "TR-04", "Mow pass 1", "6:20", "0.41", "Z-14 west grassland"),
    {
      date: "09-09",
      machine: "TR-07",
      task: "Mow pass 1",
      acres: cut21 - half21,
      hours: clock((cut21 - half21) / mowRate),
      rate: "0.52",
      zone: "Z-21 fence line",
      tone: "amber",
    },
    {
      date: "09-08",
      machine: "TR-15",
      task: "Seed drill trial",
      acres: Math.round(hoursOf(seedTrial) * seedRate),
      hours: seedTrial,
      rate: "—",
      zone: "Z-14 west grassland",
      tone: "neutral",
    },
    mowRow("09-05", "TR-12", "Mow pass 1", "5:05", "0.40", "Z-14 west grassland"),
  ];
  const lastDate = workLog
    .map((row) => row.date)
    .sort()
    .at(-1);
  const lastShift = workLog
    .filter((row) => row.date === lastDate)
    .reduce((sum, row) => sum + (row.acres ?? 0), 0);

  return {
    id: DEMO_RANCH_PROJECT_ID,
    siteId: null,
    name: ranch.name,
    // No numbers: the badge puts the site's area in front of it, in the user's units.
    meta: ranch.purpose,
    simulated: true,
    machines,
    zones,
    plans,
    agent: {
      headline: "Rebuilding the fence-line mow sequence",
      summary: "Working on the mow schedule",
      footer: "Plan recovers 11 days · review when ready",
      actions: [
        { id: "a1", text: "Sequencing passes against the 48 h wind forecast", status: "run" },
        { id: "a2", text: "Merging the fence strip into one revised task", status: "run" },
        { id: "a3", text: "Holding TR-12 for two passes from 06:00", status: "wait" },
        { id: "a4", text: "Drafting the gate and access note for the field crew", status: "wait" },
        { id: "a5", text: "Checking the mow finishes before the seed window", status: "wait" },
        {
          id: "a6",
          text: `Drew the ${strip} m strip inside the fence on Z-21 · ${ac21} ac`,
          status: "done",
        },
        { id: "a7", text: "Read slope, fence line and oak-canopy layers", status: "done" },
        { id: "a8", text: "Pulled the 3-hourly forecast through Sunday", status: "done" },
        { id: "a9", text: "Found star thistle cover as the only goal behind plan", status: "done" },
      ],
    },
    fleetStats: [
      { value: String(machines.filter((m) => m.status !== "idle").length), label: "Working now" },
      { value: `${lastShift} ac`, label: "Treated, last shift" },
      { value: "92%", label: "Uptime, 7 days" },
      {
        value: String(machines.filter((m) => m.status === "attention").length),
        label: "Need attention",
        accent: true,
      },
    ],
    fleetNote:
      "TR-07 blade wear is at 86%. The agent booked service for Thursday and moved its passes to TR-12.",
    workLog,
    feeds: [
      { id: "TR-04", camera: "front", state: "live", time: "14:22:06" },
      { id: "TR-12", camera: "front", state: "live", time: "14:22:06" },
      { id: "TR-18", camera: "mast", state: "weak", time: "14:21:58" },
      { id: "TR-15", camera: "hopper", state: "idle", time: "14:22:04" },
    ],
  };
}

export const DEMO_PROJECT: Project = buildDemoProject(DEMO_RANCH);

/** How steeply a portrait window looks down at the ranch on arrival, degrees. */
const PORTRAIT_PITCH_DEG = -55;

/** The ranch's authored arrival (the Overview bookmark), as a framing. */
export function demoRanchFraming(ranch: DemoRanch = DEMO_RANCH): Framing {
  const { target, headingDeg, pitchDeg, rangeM } = ranch.camera;
  return { target, headingDeg, pitchDeg, rangeM };
}

/**
 * Where flying to the demo ranch arrives in this window: null (its authored Overview) when
 * that shows every zone outline, chip and machine clear of the HUD, as it does in a desktop
 * window; else the closest pose that does, looking the same way up the valley (a phone held
 * upright sees a narrow slice of a landscape framing).
 */
export function demoRanchArrival(viewport: Viewport): EyePose | null {
  const camera = DEMO_RANCH.camera;
  const points = projectGroundPoints(DEMO_PROJECT);
  const authored = viewportOptions(viewport, camera.headingDeg, camera.pitchDeg);
  if (framingContains(demoRanchFraming(), points, authored).inside) return null;
  // Held upright, the width is what limits the fit; looking more steeply down lets the ranch
  // fill the height instead of the hills behind it.
  const pitch = viewport.width < viewport.height ? PORTRAIT_PITCH_DEG : camera.pitchDeg;
  return eyeOf(
    frameGround(points, viewportOptions(viewport, camera.headingDeg, pitch)),
    camera.targetHeightM,
  );
}

/** The demo project as it attaches to a site (the built-in ranch's by default), for tests. */
export function demoRanchProject(siteId: string | null = BUILTIN_RANCH_ID): Project {
  return { ...DEMO_PROJECT, siteId };
}

/**
 * Attaches the simulated project to the demo ranch: the seeded `demo-ranch` site, or its
 * built-in offline copy. No other site, built-in or not, gets the fleet.
 */
export class DemoMissionProvider implements MissionProvider {
  readonly name = "demo";

  projectForSite(siteId: string | null, siteSlug: string | null): Project | null {
    if (siteSlug === DEMO_RANCH_SLUG || siteId === BUILTIN_RANCH_ID) {
      return demoRanchProject(siteId);
    }
    return null;
  }
}
