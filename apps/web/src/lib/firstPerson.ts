/**
 * First-person movement for the scanned world: walk (gravity, eye height, jump) and fly
 * (hover, no gravity), in a local east/north/up frame in metres. Pure: the caller supplies
 * the world (ground under a point, and a collision sweep) and the input each frame, so the
 * feel -- speeds, acceleration, gravity, step height -- is tested without a scene.
 *
 * The model is the one games settled on (and the web splat viewers follow): the mouse turns
 * the view, WASD moves relative to where you look -- flattened to the ground when walking --
 * Shift sprints, Space jumps (walk) or rises (fly), and velocity eases toward what the keys
 * ask rather than jumping to it, so starts and stops are smooth instead of mechanical.
 */

import type { Vec3 } from "./occupancy";

export type MoveMode = "walk" | "fly";

export interface MoveInput {
  /** -1..1: forward (W) minus back (S). */
  forward: number;
  /** -1..1: right (D) minus left (A). */
  right: number;
  /** -1..1 in fly mode: up (Space / E) minus down (Q / C). */
  up: number;
  sprint: boolean;
  /** Ctrl held: a slow, careful pace. */
  slow?: boolean;
  /** Jump pressed this frame (walk mode, on the ground). */
  jump: boolean;
}

export interface MoveWorld {
  /** Height of the ground under (x, y), searched from `fromZ` down; null when none is known. */
  groundBelow(x: number, y: number, fromZ: number): number | null;
  /** Where a body moving from `from` to `to` may go (collision with sliding). */
  sweep(from: Vec3, to: Vec3): Vec3;
}

export interface MoveState {
  /** Eye position. */
  position: Vec3;
  velocity: Vec3;
  onGround: boolean;
}

/**
 * Tuning, in metres and seconds, matched to SuperSplat's viewer (playcanvas/supersplat-viewer,
 * MIT; its walk and fly controllers), the reference for exploring a splat on the web: walk
 * about 2.6 m/s, Shift twice that, Ctrl half; eye about 1.5 m above the floor; gravity 9.8;
 * a 4 m/s jump (about 0.8 m); fly 4 m/s, Shift four times, Ctrl a quarter.
 */
export const WALK = {
  speed: 2.6,
  sprint: 2,
  slow: 0.5,
  /** Eye height above the ground. */
  eyeHeight: 1.5,
  gravity: 9.8,
  /** Jump take-off speed: an apex of v^2 / 2g, about 0.8 m. */
  jumpSpeed: 4,
  /** A rise up to this is walked up (a kerb, a step, a root); more is a wall. */
  stepHeight: 0.35,
  /** Time constant of reaching the asked-for speed on the ground, and in the air. */
  groundResponse: 0.08,
  airResponse: 0.6,
} as const;

export const FLY = {
  speed: 4,
  sprint: 4,
  slow: 0.25,
  response: 0.13,
} as const;

/** Unit forward and right vectors on the ground plane for a heading (radians, 0 = north, clockwise). */
export function groundAxes(heading: number): { forward: Vec3; right: Vec3 } {
  return {
    forward: [Math.sin(heading), Math.cos(heading), 0],
    right: [Math.cos(heading), -Math.sin(heading), 0],
  };
}

/** Unit view direction for a heading and pitch (radians, positive pitch looks up). */
export function viewAxes(heading: number, pitch: number): { forward: Vec3; right: Vec3 } {
  const c = Math.cos(pitch);
  return {
    forward: [Math.sin(heading) * c, Math.cos(heading) * c, Math.sin(pitch)],
    right: [Math.cos(heading), -Math.sin(heading), 0],
  };
}

function ease(current: number, target: number, dt: number, tau: number): number {
  return target + (current - target) * Math.exp(-dt / Math.max(tau, 1e-3));
}

/**
 * One step of `dt` seconds. `speedScale` scales the body -- speeds, eye height, step, jump --
 * (a hand-sized scan is walked at a tiny scale); `pace` scales the speeds alone (the
 * person's chosen walking speed).
 */
export function step(
  state: MoveState,
  input: MoveInput,
  look: { heading: number; pitch: number },
  mode: MoveMode,
  world: MoveWorld,
  dt: number,
  speedScale = 1,
  pace = 1,
): MoveState {
  const clampedDt = Math.min(Math.max(dt, 0), 0.1);
  const [x, y, z] = state.position;
  let [vx, vy, vz] = state.velocity;
  const length = Math.hypot(input.forward, input.right) || 1;
  const f = input.forward / Math.max(length, 1);
  const r = input.right / Math.max(length, 1);

  if (mode === "fly") {
    const speed =
      FLY.speed * speedScale * pace * (input.sprint ? FLY.sprint : input.slow ? FLY.slow : 1);
    const { forward, right } = viewAxes(look.heading, look.pitch);
    const tx = (forward[0] * f + right[0] * r) * speed;
    const ty = (forward[1] * f + right[1] * r) * speed;
    // Up and down (Q / E) along the world's up, as a drone does.
    const tz = (forward[2] * f + input.up) * speed;
    vx = ease(vx, tx, clampedDt, FLY.response);
    vy = ease(vy, ty, clampedDt, FLY.response);
    vz = ease(vz, tz, clampedDt, FLY.response);
    const to: Vec3 = [x + vx * clampedDt, y + vy * clampedDt, z + vz * clampedDt];
    const position = world.sweep(state.position, to);
    // Hitting something takes the speed into it away, so the next frame does not push on.
    const velocity: Vec3 =
      clampedDt > 0
        ? [
            (position[0] - x) / clampedDt,
            (position[1] - y) / clampedDt,
            (position[2] - z) / clampedDt,
          ]
        : [vx, vy, vz];
    return { position, velocity, onGround: false };
  }

  // Walk: horizontal motion from the keys, vertical from gravity and the ground.
  const speed =
    WALK.speed * speedScale * pace * (input.sprint ? WALK.sprint : input.slow ? WALK.slow : 1);
  const { forward, right } = groundAxes(look.heading);
  const tx = (forward[0] * f + right[0] * r) * speed;
  const ty = (forward[1] * f + right[1] * r) * speed;
  const response = state.onGround ? WALK.groundResponse : WALK.airResponse;
  vx = ease(vx, tx, clampedDt, response);
  vy = ease(vy, ty, clampedDt, response);
  const eye = WALK.eyeHeight * speedScale;
  const step = WALK.stepHeight * speedScale;

  let onGround = state.onGround;
  if (onGround && input.jump) {
    vz = WALK.jumpSpeed * Math.sqrt(speedScale);
    onGround = false;
  }
  if (!onGround) vz -= WALK.gravity * speedScale * clampedDt;

  // Horizontal first, at knee height would be a capsule; one sphere at the eye plus the step
  // test below is the model: the sweep stops walls, the ground test takes kerbs.
  const moved = world.sweep(state.position, [x + vx * clampedDt, y + vy * clampedDt, z]);
  let [nx, ny] = moved;
  // Anything under the eye that rises more than a step above the feet is an obstacle the eye
  // cleared but the legs cannot (a low wall, a bench): stay.
  const underEye = world.groundBelow(nx, ny, z);
  if (underEye !== null && onGround && underEye > z - eye + step) {
    nx = x;
    ny = y;
    vx = 0;
    vy = 0;
  }
  const ground = world.groundBelow(nx, ny, z - eye + step);
  let nz = z + vz * clampedDt;
  const falling = vz <= 0;
  // Walking down a slope or off a kerb keeps the feet on the ground; a drop of more than a
  // step is a fall.
  const stepDown = onGround && falling && ground !== null && nz - (ground + eye) <= step;
  if (ground !== null && (nz <= ground + eye || stepDown)) {
    // Landed, or walking: the eye rides at its height above the ground (steps and slopes
    // included), eased a little so a bumpy scan does not jitter the view.
    const target = ground + eye;
    nz = onGround ? ease(z, target, clampedDt, 0.05) : target;
    if (onGround && Math.abs(target - nz) > step) nz = target;
    vz = 0;
    onGround = true;
  } else if (ground === null && onGround) {
    // No ground known here (off the scan, terrain not loaded): stay level rather than fall.
    nz = z;
    vz = 0;
  } else {
    onGround = false;
  }
  return { position: [nx, ny, nz], velocity: [vx, vy, vz], onGround };
}
