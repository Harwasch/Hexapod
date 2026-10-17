/**
 * The plan card's dress rehearsal as state: "Rehearse" reads the ground once
 * (`rehearsalGround.ts`), plans the run (`missions/rehearsal.ts`) and plays it on the map
 * (`cesium/RehearsalManager.ts`); the card shows it (`Rehearsal.tsx`) and the approve button
 * counts its raised hands.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import type { PlanDraft } from "@twin/contracts";

import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import type { RehearsalFrame } from "@/cesium/RehearsalManager";
import { useScene } from "@/cesium/SceneContext";
import { describeError } from "@/lib/log";
import { overlayFor } from "@/missions/planDraft";
import {
  planRehearsal,
  REHEARSAL_SPEEDS,
  type RaisedHand,
  type Rehearsal,
  type RehearsalSpeed,
  type RehearsalZoneInput,
} from "@/missions/rehearsal";
import type { Project, Zone } from "@/missions/types";
import { missionLandId } from "@/state/mission";

import { rehearsalGround, type RehearsalGround } from "./rehearsalGround";

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
  /** Hands the ghosts have reached so far: what the approve button says. */
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
    const offPick = scene.events.on("rehearsal-pick", ({ id }) => {
      if (!id.startsWith("hand:")) return;
      const handId = id.slice("hand:".length);
      setFocused(handId);
      scene.rehearsal.focus(handId);
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
      if (!scene || !read) return null;
      const result = planRehearsal({
        zones: rehearsalZones(draft, project, scene),
        machines: project.machines,
        elevation: read.elevation,
        elevationMissing: read.elevationMissing,
        creeks: read.creeks,
        creeksMissing: read.creeksMissing,
        includeEphemeral: options.includeEphemeral,
      });
      scene.rehearsal.load(
        result,
        { colorOf: (id) => scene.mission.machineColor(id), heightAt: read.heightAt },
        keepT,
      );
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
    rehearsalGround(scene, { landId: missionLandId(), projectId: project.id, points })
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
      scene.mission.leaveOutLane(hand.zoneId, lane);
      setLeftOut((list) => [
        ...list,
        { zoneId: hand.zoneId, label: hand.zoneLabel, number: hand.number },
      ]);
      setFocused(null);
      replan({ includeEphemeral }, frame?.t ?? 0);
    },
    [scene, replan, includeEphemeral, frame],
  );

  return {
    status,
    error,
    rehearsal,
    ground: read,
    frame,
    focused,
    leftOut,
    includeEphemeral,
    raised: status === "ready" ? (frame?.raised ?? 0) : 0,
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
