import { create } from "zustand";
import { persist } from "zustand/middleware";

import type { PlanDraft } from "@twin/contracts";

import type { Plan, Project } from "@/missions/types";

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
  removePlan: (planId: string) => void;
}

let logCounter = 0;

/** Demo/provider plans first, then the console's approved plans (replacing any with the same id). */
function withApproved(project: Project | null, approved: Record<string, Plan[]>): Project | null {
  if (!project) return null;
  const extra = approved[project.id] ?? [];
  if (extra.length === 0) return project;
  const replaced = new Set(extra.map((p) => p.id));
  return { ...project, plans: [...project.plans.filter((p) => !replaced.has(p.id)), ...extra] };
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
      setProject: (project) =>
        set((s) => ({
          project: withApproved(project, s.approvedPlans),
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
      openComposer: (seed = {}) =>
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
            error: null,
          },
        }),
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
            project: withApproved(
              { ...project, plans: project.plans.filter((p) => p.id !== plan.id) },
              approvedPlans,
            ),
            composer: null,
            planId: plan.id,
            view: "plan",
          };
        });
      },
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
      partialize: (s) => ({ approvedPlans: s.approvedPlans }),
    },
  ),
);
