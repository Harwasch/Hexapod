import { create } from "zustand";
import { persist } from "zustand/middleware";

import type { Footprint, PlanAnswerValue, PlanClarification, PlanDraft } from "@twin/contracts";

import { planningOpened } from "@/lib/planningMetrics";
import { anywhereProject } from "@/missions/anywhere";
import { isAreaZone, reshapedZone } from "@/missions/areas";
import type { GroundSource } from "@/missions/ground";
import { isPrivateZone, landIdOfProject } from "@/missions/landGround";
import type { Plan, Project, Zone } from "@/missions/types";

import { record } from "./history";

export type MissionView = "map" | "plan" | "fleet";
export type MissionSelection =
  { kind: "machine"; id: string } | { kind: "zone"; id: string } | null;

export interface AgentLogEntry {
  id: string;
  at: number;
  role: "you" | "agent";
  text: string;
}

/** The plan composer: a goal in progress, its draft, and what the operator pre-selected. */
export interface PlanComposerState {
  goal: string;
  zoneIds: string[];
  machineIds: string[];
  /** When editing an approved plan, the id the approved draft replaces. */
  replacePlanId: string | null;
  status: "idle" | "awaiting-ground" | "locating" | "drafting" | "ready" | "error";
  /** How the ground in `zoneIds` was found, for the card to say. */
  ground: { source: GroundSource; note: string } | null;
  draft: PlanDraft | null;
  /** The draft before the last redraft, so the review can say what changed. */
  previousDraft: PlanDraft | null;
  /** Answers to the agent's clarifications; sent with every redraft. */
  answers: Record<string, PlanAnswerValue>;
  /** Every clarification a draft has carried, by id, so an answered one stays a chip. */
  asked: Record<string, PlanClarification>;
  error: string | null;
}

/**
 * A saved Land's zones engaged as ground (`features/mission/landPlanning.ts`): layered onto
 * `projectId` when that project's ground overlaps the Land, or carried by the Land's own
 * `land:<id>` project. Never persisted: Land ground is workspace-private.
 */
export interface EngagedLandGround {
  landId: string;
  projectId: string;
  zones: Zone[];
}

export interface ComposerSeed {
  goal?: string;
  zoneIds?: string[];
  machineIds?: string[];
  replacePlanId?: string | null;
}

interface MissionState {
  project: Project | null;
  view: MissionView;
  selection: MissionSelection;
  hoveredMachineId: string | null;
  planId: string | null;
  /**
   * Raised by flows that want the operator to see what the agent just said (a reply, a
   * question). The status line answers by drawing the eye to its agent line, then lowers it;
   * the full activity log opens only when the operator asks for it (`ui.activityOpen`).
   */
  streamOpen: boolean;
  projectsOpen: boolean;
  feedsOpen: boolean;
  workLogOpen: boolean;
  layers: { zones: boolean; tracks: boolean; vegetation: boolean };
  log: AgentLogEntry[];
  composer: PlanComposerState | null;
  /** True when the API drafts with Claude (so the agent can look at imagery too). */
  plannerConfigured: boolean;
  setPlannerConfigured: (configured: boolean) => void;
  /** Plans approved in the console, per project id. Kept in the browser until a fleet backend persists them. */
  approvedPlans: Record<string, Plan[]>;
  /** Areas drawn in the console, per project id; the plans that cover them carry them too. */
  areas: Record<string, Zone[]>;
  /** The selected Land's zones, when it is engaged as ground. Not persisted. */
  landGround: EngagedLandGround | null;
  setProject: (project: Project | null) => void;
  /**
   * The current project again with fresh details (its site renamed): composed as `setProject`
   * composes it, but what is selected, the open plan and the composer stay as they are.
   */
  refreshProject: (project: Project) => void;
  setView: (view: MissionView) => void;
  select: (selection: MissionSelection) => void;
  setHoveredMachine: (id: string | null) => void;
  openPlan: (planId: string | null) => void;
  setStreamOpen: (open: boolean) => void;
  setProjectsOpen: (open: boolean) => void;
  setFeedsOpen: (open: boolean) => void;
  setWorkLogOpen: (open: boolean) => void;
  toggleLayer: (key: "zones" | "tracks" | "vegetation") => void;
  appendLog: (role: AgentLogEntry["role"], text: string) => void;
  openComposer: (seed?: ComposerSeed) => void;
  closeComposer: () => void;
  updateComposer: (patch: Partial<PlanComposerState>) => void;
  approvePlan: (plan: Plan) => void;
  updatePlan: (planId: string, patch: Partial<Plan>) => void;
  removePlan: (planId: string) => void;
  /** Plans the API holds for a project replace the persisted ones; local-only plans stay. */
  setRemotePlans: (projectId: string, plans: Plan[]) => void;
  addArea: (projectId: string, zone: Zone) => void;
  removeArea: (projectId: string, zoneId: string) => void;
  /** Areas that arrived with persisted plans join the drawn ones (same id: the stored one wins). */
  mergeAreas: (projectId: string, zones: Zone[]) => void;
  /**
   * Engages a Land's zones as ground. With `project` (the Land's own project) it switches to
   * it as `setProject` does; while a composer is open the switch waits, and the zones join
   * the current project instead so the card in progress keeps its goal. Without `project`
   * the zones join the current project (its fleet and zones stay).
   */
  engageLand: (ground: { landId: string; zones: Zone[] }, project?: Project) => void;
  /** The Land's zones leave the current project; a Land's own project keeps its zones. */
  disengageLand: () => void;
  /**
   * Forgets every private plan and zone this browser holds (a sign-out or a workspace switch),
   * and leaves a Land's project for "anywhere".
   */
  dropPrivate: () => void;
}

let logCounter = 0;

/**
 * Provider plans and zones first, then the console's: approved plans (replacing any with the
 * same id) and drawn areas (after the provider's zones).
 */
function compose(
  project: Project | null,
  approved: Record<string, Plan[]>,
  areas: Record<string, Zone[]>,
  landGround: EngagedLandGround | null,
): Project | null {
  if (!project) return null;
  const extra = approved[project.id] ?? [];
  const replaced = new Set(extra.map((p) => p.id));
  const drawn = areas[project.id] ?? [];
  // An engaged Land's zones come last and win over a zone with the same id.
  const land = landGround?.projectId === project.id ? landGround.zones : [];
  const landIds = new Set(land.map((z) => z.id));
  const drawnIds = new Set(drawn.map((z) => z.id));
  // A project passed back in is already composed, so stored ground with ids other than
  // "A-…" (a mission's "LA-…" areas) is in it once already: the console's copy replaces it.
  // A zone the project carries with its Land metadata keeps it over a stored copy.
  const own = project.zones.filter(
    (z) => !isAreaZone(z) && !landIds.has(z.id) && (z.land !== undefined || !drawnIds.has(z.id)),
  );
  const ownIds = new Set(own.map((z) => z.id));
  return {
    ...project,
    plans: [...project.plans.filter((p) => !replaced.has(p.id)), ...extra],
    zones: [...own, ...drawn.filter((z) => !landIds.has(z.id) && !ownIds.has(z.id)), ...land],
  };
}

/** What may be written to browser storage: never a workspace's plans or private ground. */
export function persistable(state: {
  approvedPlans: Record<string, Plan[]>;
  areas: Record<string, Zone[]>;
}): { approvedPlans: Record<string, Plan[]>; areas: Record<string, Zone[]> } {
  const approvedPlans: Record<string, Plan[]> = {};
  for (const [projectId, plans] of Object.entries(state.approvedPlans)) {
    if (landIdOfProject(projectId)) continue;
    approvedPlans[projectId] = plans.filter((p) => p.scope !== "workspace");
  }
  const areas: Record<string, Zone[]> = {};
  for (const [projectId, zones] of Object.entries(state.areas)) {
    if (landIdOfProject(projectId)) continue;
    areas[projectId] = zones.filter((z) => !isPrivateZone(z));
  }
  return { approvedPlans, areas };
}

/** Storage written before version 2 could not tell private ground; drop anything land-shaped. */
export function migrateMissionStorage(persisted: unknown, version: number): unknown {
  if (version >= 2 || !persisted || typeof persisted !== "object") return persisted;
  const state = persisted as {
    approvedPlans?: Record<string, Plan[]>;
    areas?: Record<string, Zone[]>;
  };
  const landShaped = (id: string) => id.startsWith("L-") || id.startsWith("LA-");
  const areas: Record<string, Zone[]> = {};
  for (const [projectId, zones] of Object.entries(state.areas ?? {})) {
    areas[projectId] = zones.filter((z) => !landShaped(z.id));
  }
  return persistable({ approvedPlans: state.approvedPlans ?? {}, areas });
}

/**
 * The Land planning is engaged on: the layered ground's, else the Land whose own project is
 * current. One selector so every flow asks the same question (`useLand.active` stays the
 * single source of truth for which Land is selected).
 */
export function missionLandId(
  state: Pick<MissionState, "landGround" | "project"> = useMission.getState(),
): string | null {
  return state.landGround?.landId ?? landIdOfProject(state.project?.id);
}

export const useMission = create<MissionState>()(
  persist(
    (set, get) => ({
      project: null,
      view: "map",
      selection: null,
      hoveredMachineId: null,
      planId: null,
      streamOpen: false,
      projectsOpen: false,
      feedsOpen: false,
      workLogOpen: false,
      layers: { zones: true, tracks: true, vegetation: false },
      log: [],
      composer: null,
      approvedPlans: {},
      areas: {},
      landGround: null,
      setProject: (project) =>
        set((s) => ({
          project: compose(project, s.approvedPlans, s.areas, s.landGround),
          selection: null,
          planId: null,
          composer: null,
        })),
      refreshProject: (project) =>
        set((s) =>
          s.project?.id === project.id
            ? { project: compose(project, s.approvedPlans, s.areas, s.landGround) }
            : {},
        ),
      setView: (view) => set({ view, projectsOpen: false }),
      select: (selection) => set({ selection }),
      setHoveredMachine: (hoveredMachineId) => set({ hoveredMachineId }),
      openPlan: (planId) => set({ planId, view: "plan", composer: null }),
      setStreamOpen: (streamOpen) => set({ streamOpen }),
      setProjectsOpen: (projectsOpen) => set({ projectsOpen }),
      setFeedsOpen: (feedsOpen) => set({ feedsOpen }),
      setWorkLogOpen: (workLogOpen) => set({ workLogOpen }),
      toggleLayer: (key) => set((s) => ({ layers: { ...s.layers, [key]: !s.layers[key] } })),
      appendLog: (role, text) =>
        set((s) => ({
          log: [...s.log.slice(-49), { id: `log-${++logCounter}`, at: Date.now(), role, text }],
        })),
      openComposer: (seed = {}) => {
        planningOpened();
        set({
          view: "plan",
          planId: null,
          projectsOpen: false,
          composer: {
            goal: seed.goal ?? "",
            zoneIds: seed.zoneIds ?? [],
            machineIds: seed.machineIds ?? [],
            replacePlanId: seed.replacePlanId ?? null,
            status: "idle",
            ground: null,
            draft: null,
            previousDraft: null,
            answers: {},
            asked: {},
            error: null,
          },
        });
      },
      closeComposer: () => set({ composer: null }),
      plannerConfigured: false,
      setPlannerConfigured: (plannerConfigured) => set({ plannerConfigured }),
      updateComposer: (patch) =>
        set((s) => (s.composer ? { composer: { ...s.composer, ...patch } } : {})),
      approvePlan: (plan) => {
        const project = get().project;
        if (!project) return;
        set((s) => {
          const current = (s.approvedPlans[project.id] ?? []).filter((p) => p.id !== plan.id);
          const approvedPlans = { ...s.approvedPlans, [project.id]: [...current, plan] };
          return {
            approvedPlans,
            project: compose(
              { ...project, plans: project.plans.filter((p) => p.id !== plan.id) },
              approvedPlans,
              s.areas,
              s.landGround,
            ),
            composer: null,
            planId: plan.id,
            view: "plan",
          };
        });
      },
      updatePlan: (planId, patch) => {
        const project = get().project;
        if (!project) return;
        set((s) => {
          const approved = (s.approvedPlans[project.id] ?? []).map((p) =>
            p.id === planId ? { ...p, ...patch } : p,
          );
          const approvedPlans = { ...s.approvedPlans, [project.id]: approved };
          return {
            approvedPlans,
            project: {
              ...project,
              plans: project.plans.map((p) => (p.id === planId ? { ...p, ...patch } : p)),
            },
          };
        });
      },
      setRemotePlans: (projectId, plans) =>
        set((s) => {
          const local = (s.approvedPlans[projectId] ?? []).filter((p) => !p.persisted);
          const approvedPlans = { ...s.approvedPlans, [projectId]: [...plans, ...local] };
          const project = s.project;
          if (project?.id !== projectId) return { approvedPlans };
          // Provider plans carry no goal; everything else came from the console and is rebuilt.
          const base = { ...project, plans: project.plans.filter((p) => p.goal === undefined) };
          return { approvedPlans, project: compose(base, approvedPlans, s.areas, s.landGround) };
        }),
      addArea: (projectId, zone) =>
        set((s) => {
          const areas = {
            ...s.areas,
            [projectId]: [...(s.areas[projectId] ?? []).filter((z) => z.id !== zone.id), zone],
          };
          const project = s.project;
          return {
            areas,
            project:
              project?.id === projectId
                ? compose(project, s.approvedPlans, areas, s.landGround)
                : project,
          };
        }),
      removeArea: (projectId, zoneId) =>
        set((s) => {
          const areas = {
            ...s.areas,
            [projectId]: (s.areas[projectId] ?? []).filter((z) => z.id !== zoneId),
          };
          const project = s.project;
          return {
            areas,
            project:
              project?.id === projectId
                ? compose(project, s.approvedPlans, areas, s.landGround)
                : project,
          };
        }),
      engageLand: (ground, project) =>
        set((s) => {
          const current = s.project;
          // A Land's own project, unless a card in progress would lose its goal: then the
          // zones join the current project and the switch waits.
          if (project && (!s.composer || current?.id === project.id || !current)) {
            const landGround = {
              landId: ground.landId,
              projectId: project.id,
              zones: ground.zones,
            };
            if (current?.id === project.id) {
              return {
                landGround,
                project: compose(project, s.approvedPlans, s.areas, landGround),
              };
            }
            return {
              landGround,
              project: compose(project, s.approvedPlans, s.areas, landGround),
              selection: null,
              planId: null,
              composer: null,
            };
          }
          if (!current) return {};
          const landGround = { landId: ground.landId, projectId: current.id, zones: ground.zones };
          return { landGround, project: compose(current, s.approvedPlans, s.areas, landGround) };
        }),
      disengageLand: () =>
        set((s) => {
          if (!s.landGround) return {};
          const project = s.project;
          // A Land's own project carries its zones; only layered ones leave.
          if (!project || landIdOfProject(project.id) === s.landGround.landId)
            return { landGround: null };
          const zones = new Set(s.landGround.zones.map((z) => z.id));
          return {
            landGround: null,
            project: compose(
              { ...project, zones: project.zones.filter((z) => !zones.has(z.id)) },
              s.approvedPlans,
              s.areas,
              null,
            ),
          };
        }),
      dropPrivate: () =>
        set((s) => {
          const kept = persistable(s);
          const current = s.project;
          const base =
            !current || landIdOfProject(current.id)
              ? current && anywhereProject()
              : {
                  ...current,
                  plans: current.plans.filter((p) => p.scope !== "workspace"),
                  zones: current.zones.filter((z) => !isPrivateZone(z)),
                };
          const project = compose(base, kept.approvedPlans, kept.areas, null);
          const planOpen = project?.plans.some((p) => p.id === s.planId) ?? false;
          const composerPrivate =
            s.composer?.zoneIds.some((id) => !project?.zones.some((z) => z.id === id)) ?? false;
          return {
            ...kept,
            landGround: null,
            project,
            planId: planOpen ? s.planId : null,
            composer: composerPrivate || project?.id !== current?.id ? null : s.composer,
            selection: project?.id === current?.id ? s.selection : null,
          };
        }),
      mergeAreas: (projectId, zones) =>
        set((s) => {
          // An area shaped here (it carries its view origin) is never clobbered by the stored
          // copy that a plans refetch brings back; stored areas fill in the rest.
          const local = s.areas[projectId] ?? [];
          const shaped = new Set(local.filter((z) => z.view ?? z.shaped).map((z) => z.id));
          const incoming = zones.filter((z) => !shaped.has(z.id));
          if (incoming.length === 0) return {};
          const ids = new Set(incoming.map((z) => z.id));
          const areas = {
            ...s.areas,
            [projectId]: [...local.filter((z) => !ids.has(z.id)), ...incoming],
          };
          const project = s.project;
          return {
            areas,
            project:
              project?.id === projectId
                ? compose(project, s.approvedPlans, areas, s.landGround)
                : project,
          };
        }),
      removePlan: (planId) => {
        const project = get().project;
        if (!project) return;
        set((s) => {
          const approvedPlans = {
            ...s.approvedPlans,
            [project.id]: (s.approvedPlans[project.id] ?? []).filter((p) => p.id !== planId),
          };
          return {
            approvedPlans,
            project: { ...project, plans: project.plans.filter((p) => p.id !== planId) },
            planId: s.planId === planId ? null : s.planId,
          };
        });
      },
    }),
    {
      name: "twin.mission.v1",
      // Version 2: Land ground and workspace plans are never written to browser storage.
      version: 2,
      partialize: (s) => persistable(s),
      migrate: (persisted, version) =>
        migrateMissionStorage(persisted, version) as Pick<MissionState, "approvedPlans" | "areas">,
    },
  ),
);

/**
 * An area's corners moved on the map (`AreaEditor`, through `SceneBridge`): the area takes the
 * new outline as one undoable step (`state/history.ts`). Undo and redo put an outline back
 * and `announce` it as the editor announces an edit, so an open plan redrafts for the outline
 * it has again (`PlanCard`). Nothing when the area is not the project's.
 */
export function reshapeArea(
  zoneId: string,
  footprint: Footprint,
  announce: (zoneId: string, footprint: Footprint) => void,
): void {
  const project = useMission.getState().project;
  const zone = project?.zones.find((z) => z.id === zoneId);
  if (!project || !zone) return;
  const reshaped = reshapedZone(zone, footprint);
  const put = (area: Zone): void => {
    useMission.getState().addArea(project.id, area);
    announce(zoneId, area.footprint);
  };
  useMission.getState().addArea(project.id, reshaped);
  record({
    label: `Reshape ${zone.name}`,
    scope: "site",
    alive: () => useMission.getState().project?.id === project.id,
    undo: () => put(zone),
    redo: () => put(reshaped),
  });
}
