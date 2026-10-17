/**
 * The plan card's dress rehearsal as state: "Rehearse" reads the ground once
 * (`rehearsalGround.ts`), plans the run (`missions/rehearsal.ts`) and plays it on the map
 * (`cesium/RehearsalManager.ts`); the card shows it (`Rehearsal.tsx`) and the approve button
 * counts its raised hands. The planner and the ground reader load on the first "Rehearse":
 * most visits never rehearse, so they are not in the app's first download.
 */

import { useCallback, useEffect, useRef, useState } from "react";

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
import { missionLandId } from "@/state/mission";

import type * as GroundModule from "./rehearsalGround";
import type { RehearsalGround } from "./rehearsalGround";

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

  const start = useCallback(() => {
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
        scene.rehearsal.play(speed);
        setStatus("ready");
      })
      .catch((cause: unknown) => {
        if (run.current !== ticket) return;
        setStatus("error");
        setError(`The rehearsal could not read the ground: ${describeError(cause)}`);
      });
  }, [scene, draft, project, replan, includeEphemeral, speed]);

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
