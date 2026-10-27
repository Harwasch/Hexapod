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

/** Starts the hero; throws when WebGL is unavailable. */
export function startHeroScene(options: HeroOptions): Hero;
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
