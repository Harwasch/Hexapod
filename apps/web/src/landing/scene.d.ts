/** Types for scene.js, the landing page's ported three.js scenes: what main.ts calls. */
import type { ThemedLook } from "./looks";

/** One crew's work cell in the hero, in world metres (x east, y south). */
export interface HeroZone {
  id: string;
  c: readonly [number, number];
  phase?: number;
  /** Height of the label's anchor above the ground. */
  lz?: number;
  wide?: number;
  /** Extra clearance of world trees around the cell. */
  clear?: number;
}

/** One stop on the camera's tour. */
export interface HeroStop {
  c: readonly [number, number];
  zoom?: number;
  az?: number;
  el?: number;
}

/** Where a crew's label sits on screen this frame. */
export interface HeroLabel {
  id: string;
  xy: readonly [number, number];
  W: number;
  H: number;
}

export interface HeroOptions {
  canvas: HTMLCanvasElement;
  /** Route index the tour starts at. */
  startAtPoint: number;
  /** An `.iso` element the scene reads its --anatomy-* colours from. */
  themeEl: HTMLElement;
  zones: readonly HeroZone[];
  route: readonly HeroStop[];
  still: MediaQueryList;
  maxPixels: () => number;
  /** View width in metres. */
  width: () => number;
  /** Where the focus sits in the frame, as fractions of its size. */
  shift: (width: number, height: number) => { x?: number; y?: number };
  labels: (list: HeroLabel[]) => void;
  /** A look to draw in, in both themes' versions (looks.ts); none is the line drawing. */
  look?: ThemedLook;
}

export interface Hero {
  /** Flies the camera to a stop on the route. */
  goTo: (routeIndex: number) => void;
  /** The point the camera looks at, in world metres. */
  focus: () => readonly number[];
}

export interface FleetCard {
  canvas: HTMLCanvasElement;
  unit: string;
}

export interface FleetOptions {
  cards: FleetCard[];
  themeEl: HTMLElement;
  still: MediaQueryList;
  maxPixels: () => number;
  look?: ThemedLook;
}

export type MissionStage = "plan" | "deploy" | "scan" | "operate";

/** What the mission scene reports each time it draws. */
export interface MissionFrame {
  stage: MissionStage;
  /** The stage's place in the order, 0 to 3. */
  index: number;
  progress: number;
  /** The status line: label and value pairs. */
  status: [string, string][];
  /** The one callout for this moment, and where its subject is on screen. */
  callout: { text: string; xy: readonly [number, number]; W: number; H: number } | null;
}

export interface MissionOptions {
  canvas: HTMLCanvasElement;
  themeEl: HTMLElement;
  still: MediaQueryList;
  maxPixels: () => number;
  /** How far through the mission the reader has scrolled, 0 to 1. */
  progress: () => number;
  /** Close shots' width on this screen, as a fraction of the desktop's. */
  zoom?: (width: number, height: number) => number;
  shift?: (width: number, height: number) => { x?: number; y?: number };
  onFrame: (frame: MissionFrame) => void;
  look?: ThemedLook;
  scenario?: string;
}

export interface Mission {
  spans: { name: MissionStage; t0: number; t1: number }[];
}

/** Starts the hero; throws when WebGL is unavailable. */
export function startHeroScene(options: HeroOptions): Hero;
/** Starts the mission scene; throws when WebGL is unavailable. */
export function startMissionScene(options: MissionOptions): Mission;
/** Starts the fleet cards' turntables, one WebGL context for all of them. */
export function startFleetCards(options: FleetOptions): void;

/** One robot as the story places it at a moment (scene.js makeStory). */
export interface StoryBot {
  p?: number[];
  h: number;
  moving?: boolean;
  carried?: boolean;
  local?: number[];
}

/** For tests: one hero crew's story, with no renderer. */
export function crewStory(zone: { id: string; c: [number, number] }): {
  OPS: { end: number };
  model: (
    stage: "operate",
    t: number,
  ) => { hex: (StoryBot | undefined)[]; atvs: StoryBot[]; quad: StoryBot & { p: number[] } };
};

/** For tests: the mission's story, with no renderer. */
export function missionStory(): {
  spans: { name: MissionStage; t0: number; t1: number }[];
  /** Per stage: [t, target x, target y, width, azimuth, elevation]. */
  keys: Record<MissionStage, number[][]>;
  at: (
    stage: MissionStage,
    t: number,
  ) => {
    m: {
      atvs: StoryBot[];
      staged?: StoryBot[];
      hex: (StoryBot | undefined)[];
      vtol: StoryBot & { p: number[] };
      quad: StoryBot & { p: number[] };
    };
    status: [string, string][];
  };
};
