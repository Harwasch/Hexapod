/**
 * Where a scan's sunlight relight (cesium/sunlight.ts) meets the dedicated renderer that draws
 * the scan (PlayCanvas: scanView/sunHook.ts), per asset id: the sun to light by now -- its
 * direction in the scan's frame, how far up it is, the strength -- or none, which is the
 * photograph. The relight puts it here each frame it works it out; a renderer drawing that
 * asset watches it, and its watching is what says the relight is on screen (`scanSunDrawn`),
 * so the badge never claims a relight no renderer draws.
 *
 * A sun is put here only when it moved enough to show (`SUN_EPSILON`): the overlay draws a
 * frame for each one, and the sun of now moves on its own.
 *
 * Both sides are loaded on demand (the relight when the sun is first on, a renderer when
 * chosen), and so is this: nothing of it is on the globe's first load.
 */

/** The sun a renderer lights a scan by. */
export interface ScanSun {
  /** Unit direction to the sun in the scan's frame (east, north, up of the tileset's root). */
  readonly dir: readonly [number, number, number];
  /** `sunUpFactor` of its elevation at the scan's centre: 0 at night. */
  readonly sunUp: number;
  /** 0 draws the photograph (no sun to light by: not placed, or not a number), 1 the relight. */
  readonly strength: number;
}

/** Smallest change of direction (radians, about) or of `sunUp` that is handed on. */
export const SUN_EPSILON = 1e-4;

type Listener = (sun: ScanSun | null) => void;

const suns = new Map<string, ScanSun>();
const listeners = new Map<string, Set<Listener>>();
const drawerListeners = new Map<string, Set<() => void>>();

function same(a: ScanSun | null, b: ScanSun | null): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return (
    a.strength === b.strength &&
    Math.abs(a.sunUp - b.sunUp) < SUN_EPSILON &&
    Math.abs(a.dir[0] - b.dir[0]) < SUN_EPSILON &&
    Math.abs(a.dir[1] - b.dir[1]) < SUN_EPSILON &&
    Math.abs(a.dir[2] - b.dir[2]) < SUN_EPSILON
  );
}

/** Puts asset `assetId`'s sun (null: the photograph) where its renderer reads it. */
export function publishScanSun(assetId: string, sun: ScanSun | null): void {
  if (same(suns.get(assetId) ?? null, sun)) return;
  if (sun) suns.set(assetId, sun);
  else suns.delete(assetId);
  for (const listener of listeners.get(assetId) ?? []) listener(sun);
}

/**
 * Calls `listener` with asset `assetId`'s sun now and on every change; returns the remover.
 * A renderer watches only while it can draw the relight: watching is what says it is drawn.
 */
export function watchScanSun(assetId: string, listener: Listener): () => void {
  let set = listeners.get(assetId);
  if (!set) listeners.set(assetId, (set = new Set()));
  set.add(listener);
  for (const changed of drawerListeners.get(assetId) ?? []) changed();
  listener(suns.get(assetId) ?? null);
  return () => {
    set.delete(listener);
    if (set.size === 0) listeners.delete(assetId);
    for (const changed of drawerListeners.get(assetId) ?? []) changed();
  };
}

/** Whether a renderer that relights is drawing asset `assetId` now. */
export function scanSunDrawn(assetId: string): boolean {
  return (listeners.get(assetId)?.size ?? 0) > 0;
}

/** Calls `changed` when a renderer starts or stops drawing asset `assetId`; returns the remover. */
export function onScanSunDrawers(assetId: string, changed: () => void): () => void {
  let set = drawerListeners.get(assetId);
  if (!set) drawerListeners.set(assetId, (set = new Set()));
  set.add(changed);
  return () => {
    set.delete(changed);
    if (set.size === 0) drawerListeners.delete(assetId);
  };
}

/** What a dedicated renderer drew of the relight, last time it was asked. */
export interface ScanSunReading {
  renderer: string;
  /** The hook is in the draw material now. */
  installed: boolean;
  /** The work buffer carries the word the copy pass writes (sunHook.ts). */
  stream: boolean;
  /** The uniforms of the last frame drawn with the hook, or null. */
  sun: ScanSun | null;
  /** Frames drawn with the hook in. */
  frames: number;
}

const readings = new Map<string, () => ScanSunReading>();

/** Registers how asset `assetId`'s renderer reads out its relight; returns the remover. */
export function reportScanSun(assetId: string, read: () => ScanSunReading): () => void {
  readings.set(assetId, read);
  return () => {
    if (readings.get(assetId) === read) readings.delete(assetId);
  };
}

/** What the renderer drawing asset `assetId` says of its relight, or null when none does. */
export function scanSunReading(assetId: string): ScanSunReading | null {
  return readings.get(assetId)?.() ?? null;
}
