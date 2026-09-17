import { create } from "zustand";
import { persist } from "zustand/middleware";

import type { PlanDraft } from "@twin/contracts";

import { planningOpened } from "@/lib/planningMetrics";
import { isAreaZone } from "@/missions/areas";
import type { Plan, Project, Zone } from "@/missions/types";

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
  status: "idle" | "drafting" | "ready" | "error";
  draft: PlanDraft | null;
  /** The draft before the last redraft, so the review can say what changed. */
  previousDraft: PlanDraft | null;
  error: string | null;
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
  streamOpen: boolean;
  projectsOpen: boolean;
  feedsOpen: boolean;
  workLogOpen: boolean;
  layers: { zones: boolean; tracks: boolean; vegetation: boolean };
  log: AgentLogEntry[];
  composer: PlanComposerState | null;
  /** Plans approved in the console, per project id. Kept in the browser until a fleet backend persists them. */
  approvedPlans: Record<string, Plan[]>;
  /** Areas drawn in the console, per project id; the plans that cover them carry them too. */
  areas: Record<string, Zone[]>;
  setProject: (project: Project | null) => void;
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
): Project | null {
  if (!project) return null;
  const extra = approved[project.id] ?? [];
  const replaced = new Set(extra.map((p) => p.id));
  const drawn = areas[project.id] ?? [];
  return {
    ...project,
    plans: [...project.plans.filter((p) => !replaced.has(p.id)), ...extra],
    zones: [...project.zones.filter((z) => !isAreaZone(z)), ...drawn],
  };
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
      setProject: (project) =>
        set((s) => ({
          project: compose(project, s.approvedPlans, s.areas),
          selection: null,
          planId: null,
          composer: null,
        })),
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
            draft: null,
            previousDraft: null,
            error: null,
          },
        });
      },
      closeComposer: () => set({ composer: null }),
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
          return { approvedPlans, project: compose(base, approvedPlans, s.areas) };
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
            project: project?.id === projectId ? compose(project, s.approvedPlans, areas) : project,
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
            project: project?.id === projectId ? compose(project, s.approvedPlans, areas) : project,
          };
        }),
      mergeAreas: (projectId, zones) =>
        set((s) => {
          if (zones.length === 0) return {};
          const ids = new Set(zones.map((z) => z.id));
          const areas = {
            ...s.areas,
            [projectId]: [...(s.areas[projectId] ?? []).filter((z) => !ids.has(z.id)), ...zones],
          };
          const project = s.project;
          return {
            areas,
            project: project?.id === projectId ? compose(project, s.approvedPlans, areas) : project,
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
      version: 1,
      partialize: (s) => ({ approvedPlans: s.approvedPlans, areas: s.areas }),
    },
  ),
);
