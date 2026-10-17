/**
 * The plan card's dress rehearsal as state: "Rehearse" reads the ground once
 * (`rehearsalGround.ts`), plans the run (`missions/rehearsal.ts`) and plays it on the map
 * (`cesium/RehearsalManager.ts`); the card shows it (`Rehearsal.tsx`) and the approve button
 * counts its raised hands. The planner and the ground reader load on the first "Rehearse":
 * most visits never rehearse, so they are not in the app's first download.
 *
 * When hands go up on a Land's ground, it also asks the API what the machines could work
 * there instead (`workableGround.ts`), so the card can offer "Work only the ground it can"
 * with the acres it would cut; taking it redrafts the plan, and the redraft's review runs the
 * rehearsal again on its own (`useWorkable`'s rerun), straight to the end.
 */

import { useQueries } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { PlanDraft } from "@twin/contracts";

import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import type { RehearsalFrame } from "@/cesium/RehearsalManager";
import { useScene } from "@/cesium/SceneContext";
import { describeError } from "@/lib/log";
import { boundaryZoneOf } from "@/missions/landGround";
import { overlayFor } from "@/missions/planDraft";
import type * as PlannerModule from "@/missions/rehearsal";
import type { RaisedHand, Rehearsal, RehearsalZoneInput } from "@/missions/rehearsal";
import { REHEARSAL_SPEEDS, type RehearsalSpeed } from "@/missions/rehearsalClock";
import type { Project, Zone } from "@/missions/types";
import { useLandScope } from "@/state/landIdentity";
import { missionLandId, useMission } from "@/state/mission";
import { useWorkable } from "@/state/workable";

import type * as GroundModule from "./rehearsalGround";
import type { RehearsalGround } from "./rehearsalGround";
import {
  creeksRead,
  redraftOnWorkable,
  requestWorkableGround,
  unworkableText,
  workableOfferText,
  workableRequest,
  workableTargets,
  workableTotals,
  type KeptZone,
  type WorkableTarget,
  type WorkableTotals,
} from "./workableGround";

type Planner = typeof PlannerModule;
type GroundReader = typeof GroundModule;

let loading: Promise<[Planner, GroundReader]> | null = null;

/** The planner and the ground reader, fetched once; a failed fetch is tried again next time. */
function loadRehearsal(): Promise<[Planner, GroundReader]> {
  loading ??= Promise.all([import("@/missions/rehearsal"), import("./rehearsalGround")]);
  loading.catch(() => {
    loading = null;
  });
  return loading;
}

/** How a zone is named in the rehearsal's lists: its id, or a Land zone's name. */
function zoneLabel(zone: Zone): string {
  return zone.id.startsWith("L-") || zone.id.startsWith("LA-") ? zone.name : zone.id;
}

/**
 * The plan's zones in step order with the lanes the map draws for them. A zone's machines are
 * every machine its steps name (a survey pass by one, the treatment by two, say), so the work
 * is rehearsed with the whole crew the plan puts there.
 */
export function rehearsalZones(
  draft: Pick<PlanDraft, "zoneIds" | "machineIds" | "steps" | "coverage">,
  project: Project,
  scene: Pick<CesiumSceneManager, "mission">,
): RehearsalZoneInput[] {
  const overlay = overlayFor({
    zoneIds: draft.zoneIds,
    machineIds: draft.machineIds,
    steps: draft.steps ?? [],
  });
  const fallbackClass = draft.coverage?.[0]?.machineClass ?? null;
  return overlay.zones.flatMap((item) => {
    const zone = project.zones.find((z) => z.id === item.zoneId);
    if (!zone) return [];
    const line = zone.land?.line?.coordinates ?? null;
    const numbers = scene.mission.previewLaneNumbers(zone.id);
    const lanes =
      line && line.length >= 2
        ? [
            {
              points: line.map(([longitude, latitude]) => ({
                longitude: longitude ?? 0,
                latitude: latitude ?? 0,
              })),
              pass: 0,
              followsLine: true,
            },
          ]
        : scene.mission.previewLanes(zone.id).map((lane, index) => ({
            points: [lane.a, lane.b],
            pass: lane.pass,
            number: numbers[index] ?? index + 1,
          }));
    const coverage = draft.coverage?.find((c) => c.zoneId === zone.id);
    const crew = (draft.steps ?? []).filter((step) => step.zoneId === zone.id);
    const machineIds = crew.length
      ? [...new Set(crew.flatMap((step) => step.machineIds))]
      : item.machineIds;
    return [
      {
        zoneId: zone.id,
        label: zoneLabel(zone),
        lanes,
        machineIds,
        classId: coverage?.machineClass ?? fallbackClass,
      },
    ];
  });
}

export function approveLabel(conflicts: number, hands: number): string {
  if (hands > 0) {
    const raised = `${hands} hand${hands === 1 ? "" : "s"}`;
    return conflicts ? `Approve despite conflicts and ${raised}` : `Approve despite ${raised}`;
  }
  return conflicts ? "Approve despite conflicts" : "Approve plan";
}

export interface LeftOutLane {
  zoneId: string;
  label: string;
  number: number;
  /**
   * The lane's hands when it was left out. The approved plan still drives the lane (only the
   * preview lost it), so they still count on the approve button.
   */
  hands: number;
}

/**
 * "Work only the ground it can", as the card offers it. none: no slope or creek hand to answer.
 * needs-land / needs-terrain: what would make it possible (a saved Land; its elevation).
 * line-only: every hand is on a followed line (a fence), which has no ground to leave out.
 * nothing: every cell is within the limits, so there is nothing to leave out.
 * unworkable: no ground is left within the limits, so there is nothing to redraft on.
 * loading / ready / error: the API's answer for the zones it can rework.
 */
export interface WorkableOffer {
  state:
    | "none"
    | "needs-land"
    | "needs-terrain"
    | "line-only"
    | "nothing"
    | "unworkable"
    | "loading"
    | "ready"
    | "error";
  /** Why it is not offered, what went wrong, or what it rests on, in a sentence. */
  message: string | null;
  /** "−137 ac too steep, −14 ac creek buffer → 263 ac" once the answer is in. */
  text: string | null;
  totals: WorkableTotals | null;
  /** The zones it would rework, as the rehearsal names them. */
  labels: string[];
  /** Zones whose hands stay (a followed line, or every lane left out of the preview). */
  kept: KeptZone[];
  /** Where the numbers came from. */
  sources: string[];
  /** The elevation the slope was read on, with its date: "USGS 3DEP … (published 2025-08-27)". */
  elevation: string | null;
  /** Whether the creek buffer was applied (NHD answered for every zone). */
  creeksRead: boolean;
  apply: () => void;
  retry: () => void;
}

export interface DressRehearsal {
  status: "idle" | "loading" | "ready" | "error";
  error: string | null;
  rehearsal: Rehearsal | null;
  ground: RehearsalGround | null;
  frame: RehearsalFrame | null;
  focused: string | null;
  leftOut: LeftOutLane[];
  includeEphemeral: boolean;
  /**
   * What the approve button counts: the hands the ghosts have reached so far, plus every hand
   * of a lane left out of the preview, which the approved plan still drives.
   */
  raised: number;
  workable: WorkableOffer;
  /**
   * This run is the one a redraft onto workable ground asked for: how many hands Approve
   * counted before, how many of them were on the zones whose ground was cut (named as the
   * rehearsal named them), and the workable areas that took their place.
   */
  after: {
    handsBefore: number;
    reworkedBefore: number;
    /** Whether the cut applied the creek buffer: creek hands that stay say why. */
    creeksRead: boolean;
    labels: string[];
    zoneIds: string[];
  } | null;
  start: () => void;
  stop: () => void;
  play: () => void;
  pause: () => void;
  setSpeed: (speed: RehearsalSpeed) => void;
  seek: (t: number) => void;
  skipToEnd: () => void;
  focus: (handId: string | null) => void;
  leaveOut: (hand: RaisedHand) => void;
  restoreLanes: () => void;
  setIncludeEphemeral: (on: boolean) => void;
}

/**
 * The dress rehearsal of the draft in the plan card: gathers the ground once ("Rehearse"),
 * plans the run (`missions/rehearsal.ts`) and plays it on the map (`scene.rehearsal`).
 * `planKey` names what the run depends on (zones, steps, machines, classes): when it changes
 * the run is stopped, because it no longer shows this plan.
 */
export function useDressRehearsal(
  draft: PlanDraft,
  project: Project,
  planKey: string,
): DressRehearsal {
  const scene = useScene();
  const [status, setStatus] = useState<DressRehearsal["status"]>("idle");
  const [error, setError] = useState<string | null>(null);
  const [rehearsal, setRehearsal] = useState<Rehearsal | null>(null);
  const [frame, setFrame] = useState<RehearsalFrame | null>(null);
  const [focused, setFocused] = useState<string | null>(null);
  const [leftOut, setLeftOut] = useState<LeftOutLane[]>([]);
  const [includeEphemeral, setEphemeral] = useState(false);
  const [speed, setSpeedState] = useState<RehearsalSpeed>(REHEARSAL_SPEEDS[0]);
  const [read, setRead] = useState<RehearsalGround | null>(null);
  const ground = useRef<RehearsalGround | null>(null);
  const planner = useRef<Planner | null>(null);
  const run = useRef(0);

  const reset = useCallback(() => {
    run.current++;
    scene?.rehearsal.stop();
    scene?.mission.restoreLanes();
    ground.current = null;
    setRead(null);
    setStatus("idle");
    setError(null);
    setRehearsal(null);
    setFrame(null);
    setFocused(null);
    setLeftOut([]);
  }, [scene]);

  useEffect(() => {
    if (!scene) return;
    const offFrame = scene.rehearsal.onFrame(setFrame);
    const offPick = scene.events.on("rehearsal-hand", ({ id }) => {
      setFocused(id);
      scene.rehearsal.focus(id);
    });
    return () => {
      offFrame();
      offPick();
    };
  }, [scene]);
  // A changed plan is not the one rehearsed; leaving the card ends the run.
  useEffect(() => reset, [planKey, reset]);

  /** Plans the run again on the ground already read (lanes left out, ephemeral toggled). */
  const replan = useCallback(
    (options: { includeEphemeral: boolean }, keepT: number) => {
      const read = ground.current;
      const planRehearsal = planner.current?.planRehearsal;
      if (!scene || !read || !planRehearsal) return null;
      const result = planRehearsal({
        zones: rehearsalZones(draft, project, scene),
        machines: project.machines,
        elevation: read.elevation,
        elevationMissing: read.elevationMissing,
        creeks: read.creeks,
        creeksMissing: read.creeksMissing,
        includeEphemeral: options.includeEphemeral,
      });
      // The project's machines keep their colours; the others (assumed generic ones) take the
      // palette slots after them, so no two ghosts share one.
      const own = project.machines.map((m) => m.id);
      const others = result.machines.map((m) => m.id).filter((id) => !own.includes(id));
      const colorOf = (id: string) =>
        own.includes(id)
          ? scene.mission.machineColor(id)
          : scene.mission.paletteColor(own.length + Math.max(0, others.indexOf(id)));
      scene.rehearsal.load(result, { colorOf, heightAt: read.heightAt }, keepT);
      setRehearsal(result);
      return result;
    },
    [scene, draft, project],
  );

  const begin = useCallback(
    (options: { toEnd?: boolean } = {}) => {
      if (!scene) return;
      const ticket = ++run.current;
      scene.rehearsal.stop();
      setStatus("loading");
      setError(null);
      setFocused(null);
      const zones = rehearsalZones(draft, project, scene);
      const points = zones.flatMap((zone) => zone.lanes.flatMap((lane) => lane.points));
      if (points.length === 0) {
        setStatus("error");
        setError("Nothing to rehearse: this plan has no lanes drawn on the map.");
        return;
      }
      const landId = missionLandId();
      const boundary = landId ? boundaryZoneOf(project.zones, landId) : null;
      loadRehearsal()
        .then(
          ([loadedPlanner, reader]) => {
            planner.current = loadedPlanner;
            return reader.rehearsalGround(scene, {
              landId,
              projectId: project.id,
              points,
              land: boundary ? { name: boundary.name, footprint: boundary.footprint } : null,
            });
          },
          (cause: unknown) => {
            throw new Error(`the rehearsal did not load (${describeError(cause)}). Try again.`);
          },
        )
        .then((read) => {
          if (run.current !== ticket) return;
          ground.current = read;
          setRead(read);
          const result = replan({ includeEphemeral }, 0);
          if (!result) return;
          // A rerun after a redraft goes straight to its verdict; "Play" replays it.
          if (options.toEnd) scene.rehearsal.seek(Number.POSITIVE_INFINITY);
          else scene.rehearsal.play(speed);
          setStatus("ready");
        })
        .catch((cause: unknown) => {
          if (run.current !== ticket) return;
          setStatus("error");
          setError(`The rehearsal could not read the ground: ${describeError(cause)}`);
        });
    },
    [scene, draft, project, replan, includeEphemeral, speed],
  );
  // A redraft onto workable ground waits for this draft to rehearse it again. Taken in the
  // timeout, after the review has drawn the draft's lanes; React's development double mount
  // clears the first timeout before it runs, so the rerun is taken once.
  const start = useCallback(() => begin(), [begin]);
  const beginRef = useRef(begin);
  beginRef.current = begin;
  const zoneKey = draft.zoneIds.join(",");
  useEffect(() => {
    if (!scene) return;
    const timer = window.setTimeout(() => {
      if (useWorkable.getState().takeRerun(zoneKey.split(","))) beginRef.current({ toEnd: true });
    }, 0);
    return () => window.clearTimeout(timer);
  }, [scene, zoneKey]);

  const leaveOut = useCallback(
    (hand: RaisedHand) => {
      if (!scene) return;
      const lane = scene.mission.previewLanes(hand.zoneId)[hand.index];
      if (!lane) return;
      // Every hand on the lane, raised yet or not: the approved plan still drives all of it.
      const hands =
        rehearsal?.hands.filter((h) => h.zoneId === hand.zoneId && h.index === hand.index).length ??
        1;
      scene.mission.leaveOutLane(hand.zoneId, lane);
      setLeftOut((list) => [
        ...list,
        { zoneId: hand.zoneId, label: hand.zoneLabel, number: hand.number, hands },
      ]);
      setFocused(null);
      replan({ includeEphemeral }, frame?.t ?? 0);
    },
    [scene, replan, includeEphemeral, frame, rehearsal],
  );

  const workable = useWorkableOffer({
    status,
    rehearsal,
    ground: read,
    project,
    draft,
    includeEphemeral,
    leftOut,
  });
  const done = useWorkable((s) => s.done);
  const after =
    done && status === "ready" && done.zoneIds.every((id) => draft.zoneIds.includes(id))
      ? {
          handsBefore: done.handsBefore,
          reworkedBefore: done.reworkedBefore,
          creeksRead: done.creeksRead,
          labels: done.labels,
          zoneIds: done.zoneIds,
        }
      : null;
  const leftOutHands = leftOut.reduce((sum, lane) => sum + lane.hands, 0);
  return {
    status,
    error,
    rehearsal,
    ground: read,
    frame,
    focused,
    leftOut,
    includeEphemeral,
    raised: status === "ready" ? (frame?.raised ?? 0) + leftOutHands : 0,
    workable,
    after,
    start,
    stop: reset,
    play: () => scene?.rehearsal.play(speed),
    pause: () => scene?.rehearsal.pause(),
    setSpeed: (next) => {
      setSpeedState(next);
      scene?.rehearsal.setSpeed(next);
    },
    seek: (t) => scene?.rehearsal.seek(t),
    skipToEnd: () => scene?.rehearsal.seek(Number.POSITIVE_INFINITY),
    focus: (handId) => {
      setFocused(handId);
      scene?.rehearsal.focus(handId);
    },
    leaveOut,
    restoreLanes: () => {
      scene?.mission.restoreLanes();
      setLeftOut([]);
      replan({ includeEphemeral }, frame?.t ?? 0);
    },
    setIncludeEphemeral: (on) => {
      setEphemeral(on);
      replan({ includeEphemeral: on }, frame?.t ?? 0);
    },
  };
}

/**
 * A footprint's identity for a cache key: a 53-bit hash of all of its GeoJSON (cyrb53: two
 * 32-bit multiply-xor lanes) and its length. A reshaped zone keeps its id, so the key must
 * change with any corner of it; the whole outline in the key would be hashed by the query
 * cache on every render, and a workable area can run to thousands of corners.
 */
export function footprintKey(footprint: Zone["footprint"]): string {
  const text = JSON.stringify(footprint);
  let h1 = 0xdeadbeef;
  let h2 = 0x41c6ce57;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    h1 = Math.imul(h1 ^ c, 2654435761);
    h2 = Math.imul(h2 ^ c, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  const hash = 4294967296 * (2097151 & h2) + (h1 >>> 0);
  return `${text.length}:${hash.toString(36)}`;
}

/**
 * The card's "Work only the ground it can": once the run is ready with slope or creek hands,
 * asks the API for each zone it can rework (cached, so a replay or a toggle back is free) and
 * says what it would cut; or, without a Land or its elevation, says what is needed.
 */
function useWorkableOffer(input: {
  status: DressRehearsal["status"];
  rehearsal: Rehearsal | null;
  ground: RehearsalGround | null;
  project: Project;
  draft: PlanDraft;
  includeEphemeral: boolean;
  leftOut: readonly LeftOutLane[];
}): WorkableOffer {
  const { status, rehearsal, ground, project, draft, includeEphemeral, leftOut } = input;
  const scope = useLandScope();
  const landId = useMission((s) => missionLandId(s));
  const [failure, setFailure] = useState<string | null>(null);
  const { targets, kept, keys } = useMemo(() => {
    if (!rehearsal || status !== "ready")
      return { targets: [] as WorkableTarget[], kept: [] as KeptZone[], keys: [] as string[] };
    const found = workableTargets(rehearsal, project.zones, leftOut);
    return { ...found, keys: found.targets.map((target) => footprintKey(target.zone.footprint)) };
  }, [rehearsal, status, project.zones, leftOut]);
  const rasterId = ground?.elevation?.rasterId ?? null;
  const queries = useQueries({
    queries: targets.map((target, index) => ({
      queryKey: [
        "land-workable",
        scope,
        landId,
        target.zoneId,
        keys[index],
        target.machineClass,
        rasterId,
        includeEphemeral,
      ],
      queryFn: () =>
        requestWorkableGround(
          landId ?? "",
          workableRequest(target, { demRasterId: rasterId, includeEphemeral }),
        ),
      enabled: Boolean(landId && rasterId),
      staleTime: 5 * 60_000,
      retry: false,
    })),
  });
  const labels = targets.map((t) => t.label);
  const base = {
    labels,
    kept,
    totals: null,
    text: null,
    sources: [] as string[],
    elevation: null,
    creeksRead: false,
    retry: () => {
      setFailure(null);
      for (const query of queries) if (query.isError) void query.refetch();
    },
    apply: () => undefined,
  };
  // Every hand Approve counts: those of lanes left out of the preview too.
  const handsBefore = (rehearsal?.hands.length ?? 0) + leftOut.reduce((s, l) => s + l.hands, 0);
  if (status !== "ready" || !rehearsal || handsBefore === 0)
    return { ...base, state: "none", message: null };
  if (targets.length === 0)
    return {
      ...base,
      state: "line-only",
      message: `Every hand is on a line the plan follows (${kept.map((k) => k.label).join(", ")}), which has no ground to leave out. Leave lanes out, or change the plan in the bar.`,
    };
  if (!landId)
    return {
      ...base,
      state: "needs-land",
      message:
        "Save this ground as a Land (or open the Land it is on) to read its terrain: working only the ground the machines can needs the Land's elevation (USGS 3DEP) and the creeks around it.",
    };
  if (!rasterId)
    return {
      ...base,
      state: "needs-terrain",
      message:
        "Analyze bare-earth terrain (USGS 3DEP) for this land in Land to read its slope, then rehearse again.",
    };
  const error = queries.find((q) => q.isError)?.error;
  if (error || failure)
    return {
      ...base,
      state: "error",
      message: failure ?? `The workable ground could not be read: ${describeError(error)}`,
    };
  const results = queries.map((q) => q.data);
  if (results.some((r) => r === undefined))
    return {
      ...base,
      state: "loading",
      message: "Reading the terrain and creeks under the hands…",
    };
  const answers = targets.flatMap((target, index) => {
    const result = results[index];
    return result ? [{ target, result }] : [];
  });
  // A zone with nothing left within the limits is not reworked: the redraft keeps it whole,
  // so its acres are not in the offer's total, and the card says it stays.
  const usable = answers.filter((a) => a.result.workable);
  const whole = unworkableText(answers.filter((a) => !a.result.workable).map((a) => a.target));
  const warnings = [...new Set(answers.flatMap((a) => a.result.warnings))];
  if (usable.length === 0)
    return {
      ...base,
      state: "unworkable",
      message: [
        `Nothing is left to work on ${labels.join(" and ")} (${workableOfferText(workableTotals(answers.map((a) => a.result)))}): every part is too steep, inside a creek buffer or too small for its machines' assumed limits. Change the plan in the bar, or approve despite the hands.`,
        ...warnings,
      ].join(" "),
    };
  const totals = workableTotals(usable.map((a) => a.result));
  // Every cell within the limits: the hands come from lanes read between cells, and there is
  // no ground to leave out. Say so rather than offer a redraft onto the same ground.
  if (totals.footprintM2 - totals.workableM2 < 0.05 * 4046.8564224)
    return {
      ...base,
      state: "nothing",
      message: [
        `On ${usable[0]?.result.elevation.label ?? "the Land's terrain"} every cell of ${usable.map((a) => a.target.label).join(" and ")} is within its machines' assumed limits, so there is no ground to leave out: these hands are read between cells, where a lane's samples take in a steeper neighbour.`,
        whole,
      ]
        .filter(Boolean)
        .join(" "),
    };
  return {
    ...base,
    state: "ready",
    labels: usable.map((a) => a.target.label),
    totals,
    text: workableOfferText(totals),
    sources: [...new Set(usable.flatMap((a) => a.result.sources))],
    creeksRead: usable.every((a) => creeksRead(a.result)),
    elevation: (() => {
      const read = usable[0]?.result.elevation;
      return read ? `${read.label}${read.published ? ` (published ${read.published})` : ""}` : null;
    })(),
    message: [whole, ...warnings].filter(Boolean).join(" ") || null,
    // The redraft starts at once and the review goes while it drafts, so one click is all.
    apply: () => {
      if (!ground) return;
      setFailure(redraftOnWorkable({ draft, answers, handsBefore, heightAt: ground.heightAt }));
    },
  };
}
