/**
 * "Work only the ground it can": the dress rehearsal's one-click answer to its raised hands.
 *
 * The rehearsal (`missions/rehearsal.ts`) raises a hand where a drawn lane goes past a limit
 * its machine class is assumed to keep. Leaving one lane out changes nothing that is approved;
 * this changes the plan. Each zone with slope or creek hands, driven by ground machines, is
 * sent to the API (`POST /land/{id}/workable-ground`) with the strictest class among them: the
 * answer is the zone less the ground over that class's assumed slope on the Land's elevation
 * raster and inside its assumed buffer around NHD creeks, less slivers. The answer becomes a
 * private area derived from the Land (attributed to every source and assumption), the draft is
 * redrafted on it with the same goal, machines and answers (`startPlanDraft`), the left-out
 * ground is hatched on the map, and the rehearsal runs again once the new draft is ready.
 *
 * It needs the Land: the slope comes from its elevation raster and the creeks from around its
 * outline. Without one the card says what would make it possible instead of offering it.
 */

import { useEffect, useRef } from "react";

import type {
  MachineClassId,
  PlanDraft,
  WorkableGround,
  WorkableGroundRequest,
} from "@twin/contracts";

import { api, unwrap } from "@/api/client";
import type { LeftOutGround } from "@/cesium/RehearsalManager";
import { useScene } from "@/cesium/SceneContext";
import { nextAreaId, zoneFromFootprint } from "@/missions/areas";
import type { Rehearsal } from "@/missions/rehearsal";
import type { Project, Zone } from "@/missions/types";
import { useMission } from "@/state/mission";
import { useWorkable, type WorkableRecord } from "@/state/workable";

import { adoptAreas } from "./planFlow";
import { startPlanDraft } from "./planDrafting";

const ACRE_M2 = 4046.8564224;

export type GroundClassId = Exclude<MachineClassId, "drone">;

/** A zone the redraft can rework: its outline, and whose limits it is cut to. */
export interface WorkableTarget {
  zoneId: string;
  /** How the rehearsal names it: "Z-14", or a Land zone's name. */
  label: string;
  zone: Zone;
  machineClass: GroundClassId;
  /** Its slope and creek hands. */
  hands: number;
}

/** A zone whose hands stay: a followed line (a fence) has no ground to leave out. */
export interface KeptZone {
  label: string;
  hands: number;
}

/**
 * The zones a redraft onto workable ground can rework: every zone with a raised hand whose
 * lanes ground machines drive over an outline. A zone driven by several classes is cut to the
 * strictest (the lowest slope limit, then the widest creek buffer), so no class's hands stay.
 */
export function workableTargets(
  rehearsal: Pick<Rehearsal, "hands" | "lanes" | "machines">,
  zones: readonly Zone[],
): { targets: WorkableTarget[]; kept: KeptZone[] } {
  const targets: WorkableTarget[] = [];
  const kept: KeptZone[] = [];
  const zoneIds = [...new Set(rehearsal.hands.map((h) => h.zoneId))];
  for (const zoneId of zoneIds) {
    const hands = rehearsal.hands.filter((h) => h.zoneId === zoneId);
    const label = hands[0]?.zoneLabel ?? zoneId;
    const zone = zones.find((z) => z.id === zoneId);
    const lanes = rehearsal.lanes.filter((l) => l.zoneId === zoneId);
    const classes = lanes.flatMap((lane) => {
      const capability = rehearsal.machines.find((m) => m.id === lane.machineId)?.capability;
      return capability?.kind === "ground" && capability.maxSlopePct !== null ? [capability] : [];
    });
    const strictest = [...classes].sort(
      (a, b) =>
        (a.maxSlopePct ?? 0) - (b.maxSlopePct ?? 0) ||
        (b.creekBufferM ?? 0) - (a.creekBufferM ?? 0),
    )[0];
    const followsLine = lanes.length > 0 && lanes.every((lane) => !lane.removable);
    if (!zone || followsLine || !strictest || strictest.classId === "drone") {
      kept.push({ label, hands: hands.length });
      continue;
    }
    targets.push({
      zoneId,
      label,
      zone,
      machineClass: strictest.classId,
      hands: hands.length,
    });
  }
  return { targets, kept };
}

export function workableRequest(
  target: Pick<WorkableTarget, "zone" | "machineClass">,
  options: { demRasterId: string | null; includeEphemeral: boolean },
): WorkableGroundRequest {
  return {
    footprint: target.zone.footprint,
    machineClass: target.machineClass,
    ...(options.demRasterId ? { demRasterId: options.demRasterId } : {}),
    includeEphemeralCreeks: options.includeEphemeral,
  };
}

export function requestWorkableGround(
  landId: string,
  body: WorkableGroundRequest,
): Promise<WorkableGround> {
  return unwrap(
    api.POST("/api/v1/land/{land_id}/workable-ground", {
      params: { path: { land_id: landId } },
      body,
    }),
  );
}

export interface WorkableTotals {
  footprintM2: number;
  workableM2: number;
  steepM2: number;
  creekBufferM2: number;
  sliversM2: number;
  noDataM2: number;
}

export function workableTotals(results: readonly WorkableGround[]): WorkableTotals {
  const sum = (pick: (r: WorkableGround) => number) => results.reduce((s, r) => s + pick(r), 0);
  return {
    footprintM2: sum((r) => r.footprintM2),
    workableM2: sum((r) => r.workableM2),
    steepM2: sum((r) => r.removed.steepM2),
    creekBufferM2: sum((r) => r.removed.creekBufferM2),
    sliversM2: sum((r) => r.removed.sliversM2),
    noDataM2: sum((r) => r.removed.noDataM2),
  };
}

/** "263 ac", "4.2 ac": whole acres from ten up. */
export function formatAcres(m2: number): string {
  const acres = m2 / ACRE_M2;
  return acres >= 10
    ? `${Math.round(acres).toLocaleString("en-US")} ac`
    : `${(Math.round(acres * 10) / 10).toLocaleString("en-US")} ac`;
}

/**
 * The cut in a line, every part that rounds to something: "−137 ac too steep, −14 ac creek
 * buffer, −1.2 ac too small to work → 263 ac". The minus is a real minus sign.
 */
export function workableOfferText(totals: WorkableTotals): string {
  const parts: string[] = [];
  const cut = (m2: number, what: string) => {
    if (m2 / ACRE_M2 >= 0.05) parts.push(`−${formatAcres(m2)} ${what}`);
  };
  cut(totals.steepM2, "too steep");
  cut(totals.creekBufferM2, "creek buffer");
  cut(totals.sliversM2, "too small to work");
  cut(totals.noDataM2, "no slope data");
  if (parts.length === 0) return `Nothing to leave out: all ${formatAcres(totals.workableM2)}`;
  return `${parts.join(", ")} → ${formatAcres(totals.workableM2)}`;
}

/** "a robotic brush mower", from "Robotic brush mower (assumed)". */
function classPhrase(label: string): string {
  const plain = label.replace(/\s*\(assumed\)$/, "");
  return `a ${plain.slice(0, 1).toLowerCase()}${plain.slice(1)}`;
}

/** The elevation the slope came from, short: "USGS 3DEP 1/3″ DEM, published 2025-08-27". */
function elevationShort(result: WorkableGround): string {
  const { elevation } = result;
  const name =
    elevation.dataset === "usgs-3dep-13"
      ? "USGS 3DEP 1/3″ DEM"
      : "Copernicus GLO-30 surface model (incl. canopy)";
  return elevation.published ? `${name}, published ${elevation.published}` : name;
}

/**
 * Where a workable area came from, in one line (a zone's attribution, at most 300
 * characters): "Workable ground for a robotic brush mower: slope ≤ 25 % (assumed) on USGS 3DEP
 * 1/3″ DEM, published 2025-08-27, 15 m creek buffer (assumed) on USGS NHD, from West grassland
 * (Z-14)".
 */
export function workableAttribution(result: WorkableGround, from: string): string {
  const { limits, creeks } = result;
  const buffer =
    creeks.status === "available" || creeks.status === "empty"
      ? `${limits.creekBufferM} m creek buffer (assumed) on USGS NHD`
      : "creek buffer not applied (USGS NHD not read)";
  return `Workable ground for ${classPhrase(limits.label)}: slope ≤ ${limits.maxSlopePct} % (assumed) on ${elevationShort(result)}, ${buffer}, from ${from}`.slice(
    0,
    300,
  );
}

/** A zone's name without the id some names lead with ("Z-14 West grassland"). */
function plainName(zone: Pick<Zone, "id" | "name">): string {
  return zone.name.startsWith(`${zone.id} `) ? zone.name.slice(zone.id.length + 1) : zone.name;
}

/** "West grassland (Z-14)", or a Land zone's name alone. */
function fromLabel(target: Pick<WorkableTarget, "zone" | "label">): string {
  const name = plainName(target.zone);
  return target.label === target.zone.name ? name : `${name} (${target.label})`;
}

/** The workable ground as a private zone derived from the Land; null when nothing is left. */
export function workableZone(
  result: WorkableGround,
  target: Pick<WorkableTarget, "zone" | "label">,
  id: string,
): Zone | null {
  if (!result.workable) return null;
  const totals = workableTotals([result]);
  const zone = zoneFromFootprint(id, `${plainName(target.zone)}, workable`, result.workable);
  const cut = [
    totals.steepM2 > 0 &&
      `${formatAcres(totals.steepM2)} over ${result.limits.maxSlopePct} % slope`,
    totals.creekBufferM2 > 0 &&
      `${formatAcres(totals.creekBufferM2)} within ${result.limits.creekBufferM} m of creeks`,
  ].filter(Boolean);
  return {
    ...zone,
    tone: "teal",
    short: "derived",
    task: target.zone.task,
    note: `Derived: the part of ${plainName(target.zone)} ${classPhrase(result.limits.label)} is assumed to work${cut.length ? `, ${cut.join(" and ")} left out` : ""}. Not measured on the ground.`,
    attribution: workableAttribution(result, fromLabel(target)),
    private: true,
    land: {
      kind: "derived",
      landId: result.landId,
      featureId: null,
      boundaryRevision: result.boundaryRevision,
    },
  };
}

/** The left-out ground of an answer, for the hatch. */
export function leftOutGround(result: WorkableGround): LeftOutGround[] {
  return result.leftOut.flatMap((piece) =>
    piece.geometry ? [{ reason: piece.reason, label: piece.label, footprint: piece.geometry }] : [],
  );
}

/**
 * Redrafts the open plan on workable ground: each reworked zone's place in the plan's scope
 * goes to its workable area, the machines and answers stay, the goal is the same, and the
 * rehearsal runs again on the new draft. Returns what went wrong, or null.
 */
export function redraftOnWorkable(input: {
  draft: Pick<PlanDraft, "zoneIds" | "machineIds">;
  answers: readonly { target: WorkableTarget; result: WorkableGround }[];
  handsBefore: number;
  heightAt: (longitude: number, latitude: number) => number;
}): string | null {
  const state = useMission.getState();
  const project = state.project;
  const composer = state.composer;
  if (!project || !composer) return "The plan is no longer open.";
  const made: { target: WorkableTarget; result: WorkableGround; zone: Zone }[] = [];
  for (const { target, result } of input.answers) {
    const id = nextAreaId([...project.zones, ...made.map((m) => m.zone)]);
    const zone = workableZone(result, target, id);
    if (zone) made.push({ target, result, zone });
  }
  if (made.length === 0)
    return "Nothing is left to work: every part of this ground is too steep or inside a creek buffer for its machines' assumed limits.";
  const swap = new Map(made.map(({ target, zone }) => [target.zoneId, zone.id]));
  // The planner may have chosen the zones itself (the card's scope empty): the draft's are it.
  const scope = composer.zoneIds.length ? composer.zoneIds : input.draft.zoneIds;
  const zoneIds = [...new Set(scope.map((id) => swap.get(id) ?? id))];
  const totals = workableTotals(made.map((m) => m.result));
  const note = `${made.map((m) => fromLabel(m.target)).join(" and ")} less the ground its machines are assumed unable to work: ${workableOfferText(totals)}. Derived from ${made[0]?.result.elevation.label ?? "the Land's elevation"} and USGS NHD creeks; the limits are assumed.`;
  adoptAreas(
    made.map((m) => m.zone),
    { source: "workable", note },
  );
  useMission.getState().updateComposer({
    zoneIds,
    machineIds: input.draft.machineIds,
    setAside: {
      ...composer.setAside,
      ...Object.fromEntries(made.map(({ target, zone }) => [target.zoneId, zone.id])),
    },
  });
  const records: WorkableRecord[] = made.map(({ target, result, zone }) => ({
    zoneId: zone.id,
    sourceLabel: target.label,
    leftOut: leftOutGround(result),
    heightAt: input.heightAt,
  }));
  useWorkable.getState().remember(records, {
    zoneIds: made.map((m) => m.zone.id),
    handsBefore: input.handsBefore,
    labels: made.map((m) => m.target.label),
  });
  useMission.getState().appendLog("agent", note);
  void startPlanDraft(composer.goal, {
    zoneIds,
    machineIds: input.draft.machineIds,
    reshaped: true,
  });
  return null;
}

/**
 * Hatches the ground left out of the open card's workable areas while the card is open, and
 * forgets it when the card closes. Mounted by the plan card, which outlives the draft review
 * (the review goes while the plan redrafts).
 */
export function useLeftOutHatch(project: Project): void {
  const scene = useScene();
  const records = useWorkable((s) => s.records);
  const scope = useMission((s) => s.composer?.zoneIds.join(",") ?? "");
  // Whether anything is hatched: a card that never cut any ground leaves the map alone.
  const hatched = useRef(false);
  useEffect(() => {
    if (!scene) return;
    const ids = scope.split(",");
    const shown = Object.values(records).filter(
      (record) => ids.includes(record.zoneId) && project.zones.some((z) => z.id === record.zoneId),
    );
    if (shown.length === 0 && !hatched.current) return;
    hatched.current = shown.length > 0;
    scene.rehearsal.showLeftOut(
      shown.flatMap((record) => record.leftOut),
      shown[0]?.heightAt,
    );
  }, [scene, records, scope, project.zones]);
  useEffect(
    () => () => {
      if (hatched.current) scene?.rehearsal.showLeftOut([]);
      hatched.current = false;
      if (!useMission.getState().composer) useWorkable.getState().clear();
    },
    [scene],
  );
}
