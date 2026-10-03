/**
 * Split objects (docs/SCENE_OBJECTS.md §4 "Split objects", C4): a scan's movable instances
 * taken out of its spatial tiles into small tilesets of their own
 * (tools/captures/split_objects.py), declared on the scan's root as `extras.objects`. Each
 * is drawn beside the scan where its pose puts it, and is the same instance ids as before --
 * its tile is bound in the scan's `instances.json` -- so hide, highlight and search do not
 * know it moved.
 *
 * Frames: an object's tileset root transform is the scan's times the translation to its
 * `origin` (scan local ENU, metres): at rest it draws exactly where it was measured. Its
 * `pose` is a rigid motion about that origin, in the scan's frame: a point `p` of the object
 * (scan frame) goes to `origin + t + R (p - origin)`.
 */

export type Vec3 = readonly [number, number, number];
/** A unit quaternion, `x, y, z, w`. */
export type Quat = readonly [number, number, number, number];

export interface ObjectPose {
  translation: Vec3;
  rotation: Quat;
}

export interface SplitObjectRef {
  /** The object's tileset.json, relative to the scan's. */
  uri: string;
  /** Its instance id in the scan's instances.json. */
  instance: number;
  /** Where its own frame sits in the scan's (local ENU, metres). */
  origin: Vec3;
  /** Where it is drawn, about `origin`; the rest pose is the identity. */
  pose: ObjectPose;
  splats: number;
  /** The inferred layer that fills the hole it left, when there is one. */
  fill: string | null;
}

export const REST_POSE: ObjectPose = { translation: [0, 0, 0], rotation: [0, 0, 0, 1] };

function vec3(value: unknown): Vec3 | null {
  if (!Array.isArray(value) || value.length !== 3) return null;
  if (!value.every((v) => typeof v === "number" && Number.isFinite(v))) return null;
  return [value[0] as number, value[1] as number, value[2] as number];
}

function quat(value: unknown): Quat | null {
  if (!Array.isArray(value) || value.length !== 4) return null;
  if (!value.every((v) => typeof v === "number" && Number.isFinite(v))) return null;
  const [x, y, z, w] = value as [number, number, number, number];
  const n = Math.hypot(x, y, z, w);
  if (n < 1e-9) return null;
  return [x / n, y / n, z / n, w / n];
}

/** A pose from JSON; the rest pose for what is missing or malformed. */
export function poseOf(raw: unknown): ObjectPose {
  const r = raw as { translation?: unknown; rotation?: unknown } | null | undefined;
  return {
    translation: vec3(r?.translation) ?? REST_POSE.translation,
    rotation: quat(r?.rotation) ?? REST_POSE.rotation,
  };
}

/** The split objects a scan's root declares (`extras.objects`), malformed entries skipped. */
export function splitObjectsOf(extras: unknown): SplitObjectRef[] {
  const list = (extras as { objects?: unknown } | null | undefined)?.objects;
  if (!Array.isArray(list)) return [];
  const out: SplitObjectRef[] = [];
  const seen = new Set<number>();
  for (const entry of list as Record<string, unknown>[]) {
    if (typeof entry !== "object" || entry === null) continue;
    const origin = vec3(entry.origin);
    const instance = entry.instance;
    if (typeof entry.uri !== "string" || entry.uri.length === 0 || !origin) continue;
    if (!Number.isInteger(instance) || (instance as number) < 1 || seen.has(instance as number))
      continue;
    seen.add(instance as number);
    out.push({
      uri: entry.uri,
      instance: instance as number,
      origin,
      pose: poseOf(entry.pose),
      splats: typeof entry.splats === "number" ? Math.max(0, Math.round(entry.splats)) : 0,
      fill: typeof entry.fill === "string" && entry.fill.length > 0 ? entry.fill : null,
    });
  }
  return out;
}

/** Whether a pose is the rest pose (to the float). */
export function isRest(pose: ObjectPose): boolean {
  const [x, y, z] = pose.translation;
  const [qx, qy, qz, qw] = pose.rotation;
  return x === 0 && y === 0 && z === 0 && qx === 0 && qy === 0 && qz === 0 && Math.abs(qw) === 1;
}

/**
 * The pose as a motion of the scan's local frame, column-major 4x4: `T(origin + t) R
 * T(-origin)`, so `p -> origin + t + R (p - origin)`.
 */
export function poseLocalMatrix(origin: Vec3, pose: ObjectPose): number[] {
  const [x, y, z, w] = pose.rotation;
  // Row-major entries of R.
  const r00 = 1 - 2 * (y * y + z * z);
  const r01 = 2 * (x * y - z * w);
  const r02 = 2 * (x * z + y * w);
  const r10 = 2 * (x * y + z * w);
  const r11 = 1 - 2 * (x * x + z * z);
  const r12 = 2 * (y * z - x * w);
  const r20 = 2 * (x * z - y * w);
  const r21 = 2 * (y * z + x * w);
  const r22 = 1 - 2 * (x * x + y * y);
  const [ox, oy, oz] = origin;
  const [tx, ty, tz] = pose.translation;
  // Column-major, the translation `origin + t - R origin` last.
  return [
    r00,
    r10,
    r20,
    0,
    r01,
    r11,
    r21,
    0,
    r02,
    r12,
    r22,
    0,
    ox + tx - (r00 * ox + r01 * oy + r02 * oz),
    oy + ty - (r10 * ox + r11 * oy + r12 * oz),
    oz + tz - (r20 * ox + r21 * oy + r22 * oz),
    1,
  ];
}

/** A point of the scan's local frame moved by `poseLocalMatrix`. */
export function applyLocal(m: readonly number[], p: Vec3): Vec3 {
  const at = (i: number): number => m[i] ?? 0;
  return [
    at(0) * p[0] + at(4) * p[1] + at(8) * p[2] + at(12),
    at(1) * p[0] + at(5) * p[1] + at(9) * p[2] + at(13),
    at(2) * p[0] + at(6) * p[1] + at(10) * p[2] + at(14),
  ];
}

/** A rotation about the scan's up axis (local +Z), as a pose quaternion. */
export function yawQuat(radians: number): Quat {
  return [0, 0, Math.sin(radians / 2), Math.cos(radians / 2)];
}
