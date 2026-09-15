import { create } from "zustand";

import type { Project } from "@/missions/types";

export type MissionView = "map" | "plan" | "fleet";
export type MissionSelection =
  { kind: "machine"; id: string } | { kind: "zone"; id: string } | null;

export interface AgentLogEntry {
  id: string;
  at: number;
  role: "you" | "agent";
  text: string;
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
}

let logCounter = 0;

export const useMission = create<MissionState>()((set) => ({
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
  setProject: (project) => set({ project, selection: null, planId: null }),
  setView: (view) => set({ view, projectsOpen: false }),
  select: (selection) => set({ selection }),
  setHoveredMachine: (hoveredMachineId) => set({ hoveredMachineId }),
  openPlan: (planId) => set({ planId, view: "plan" }),
  setStreamOpen: (streamOpen) => set({ streamOpen }),
  setProjectsOpen: (projectsOpen) => set({ projectsOpen }),
  setFeedsOpen: (feedsOpen) => set({ feedsOpen }),
  setWorkLogOpen: (workLogOpen) => set({ workLogOpen }),
  toggleLayer: (key) => set((s) => ({ layers: { ...s.layers, [key]: !s.layers[key] } })),
  appendLog: (role, text) =>
    set((s) => ({
      log: [...s.log.slice(-49), { id: `log-${++logCounter}`, at: Date.now(), role, text }],
    })),
}));
