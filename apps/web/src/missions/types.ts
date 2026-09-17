/**
 * Mission-control domain: projects, machines, zones, plans and agent activity.
 *
 * These are the objects the operator manages. Today they come from a demo
 * provider; the `MissionProvider` seam is where a backend / ROS bridge plugs in.
 */

import type { Footprint, PlanStep } from "@twin/contracts";

export type MachineStatus = "working" | "attention" | "idle";
export type PlanStatus = "ok" | "run" | "warn" | "idle";
export type StepStatus = "done" | "run" | "wait";

export interface LonLat {
  longitude: number;
  latitude: number;
}

export interface Machine {
  id: string;
  name: string;
  model: string;
  task: string;
  status: MachineStatus;
  position: LonLat;
  headingDeg: number;
  batteryPct: number;
  acresDone: number | null;
  shift: string;
  zoneLabel: string;
  primaryAction: string;
  /** Recent path in lon/lat, oldest first. */
  track?: LonLat[];
}

export interface Zone {
  id: string;
  name: string;
  short: string;
  footprint: Footprint;
  tone: "teal" | "amber" | "blue";
  treated: boolean;
  acres: number;
  task: string;
  machines: string;
  progressPct: number;
  note: string;
  /** Where the label chip anchors. */
  anchor: LonLat;
}

export interface PlanFact {
  k: string;
  v: string;
}

export interface Plan {
  id: string;
  title: string;
  state: string;
  status: PlanStatus;
  meta: string;
  progressPct: number;
  progressLabel: string;
  objective: string;
  stats: { value: string; label: string; accent?: boolean }[];
  facts: PlanFact[];
  agentNote: string;
  action: string;
  span: string;
  start: string;
  end: string;
  todayPct: number;
  ongoing: boolean;
  zoneIds: string[];
  /** Plans drafted in the console carry their steps, scope and origin; demo plans do not. */
  machineIds?: string[];
  steps?: PlanStep[];
  assumptions?: string[];
  goal?: string;
  source?: { kind: "claude" | "rules"; model: string | null };
}

/** What the map draws for a plan or draft: zones in step order with the machines assigned. */
export interface PlanOverlay {
  zones: { zoneId: string; machineIds: string[] }[];
}

export interface AgentAction {
  id: string;
  text: string;
  status: StepStatus;
  at?: string;
}

export interface WorkLogRow {
  date: string;
  machine: string;
  task: string;
  acres: number | null;
  hours: string;
  rate: string;
  zone: string;
  tone: "teal" | "amber" | "blue" | "neutral";
}

export interface Feed {
  id: string;
  camera: string;
  state: "live" | "weak" | "idle";
  time: string;
}

export interface Project {
  id: string;
  siteId: string | null;
  name: string;
  meta: string;
  /** Whether the numbers are simulated demo data. Always shown in the UI. */
  simulated: boolean;
  machines: Machine[];
  zones: Zone[];
  plans: Plan[];
  agent: {
    headline: string;
    summary: string;
    footer: string;
    actions: AgentAction[];
  };
  fleetStats: { value: string; label: string; accent?: boolean }[];
  fleetNote: string;
  workLog: WorkLogRow[];
  feeds: Feed[];
}

export interface MissionProvider {
  readonly name: string;
  /** Resolves the project for a catalog site, or null when the site has no mission data. */
  projectForSite(siteId: string | null, siteSlug: string | null): Project | null;
}
