/**
 * The live viewer's data: `GET /captures/{id}/live`, decoded into things three.js draws.
 *
 * The cameras and points arrive exactly as the pipeline logged them
 * (`tools/pipeline/live.py`): base64, little-endian, positions quantised to int16 against
 * an origin and a scale in the same payload. 12 bytes per camera -- int16 centre, int8
 * view direction, int8 up -- and 9 per point -- int16 position, uint8 colour. Everything
 * is in the pose solve's own (COLMAP) frame, which is not east/north/up: `up` is the mean
 * camera-up in that frame, and the viewer turns it to face the sky.
 */
import type { LiveCameras, LiveState } from "@twin/contracts";

export interface DecodedCameras {
  count: number;
  /** x, y, z per camera. */
  centres: Float32Array;
  /** Unit view direction per camera. */
  forward: Float32Array;
  /** Unit camera-up per camera. */
  up: Float32Array;
}

export interface DecodedPoints {
  count: number;
  positions: Float32Array;
  /** r, g, b in 0..1, sRGB as stored. */
  colors: Float32Array;
}

const CAMERA_BYTES = 12;
const POINT_BYTES = 9;
const INT16 = 32767;

function bytesOf(base64: string): DataView {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new DataView(bytes.buffer);
}

function unit(values: Float32Array, at: number): void {
  const length = Math.hypot(values[at] ?? 0, values[at + 1] ?? 0, values[at + 2] ?? 0) || 1;
  for (let k = 0; k < 3; k += 1) values[at + k] = (values[at + k] ?? 0) / length;
}

export function decodeCameras(payload: LiveCameras): DecodedCameras {
  const view = bytesOf(payload.cameras);
  const count = Math.min(payload.cameraCount, Math.floor(view.byteLength / CAMERA_BYTES));
  const centres = new Float32Array(count * 3);
  const forward = new Float32Array(count * 3);
  const up = new Float32Array(count * 3);
  const [ox = 0, oy = 0, oz = 0] = payload.origin;
  const origin = [ox, oy, oz];
  for (let i = 0; i < count; i += 1) {
    const at = i * CAMERA_BYTES;
    for (let k = 0; k < 3; k += 1) {
      centres[i * 3 + k] =
        (origin[k] ?? 0) + (view.getInt16(at + 2 * k, true) / INT16) * payload.scale;
      forward[i * 3 + k] = view.getInt8(at + 6 + k) / 127;
      up[i * 3 + k] = view.getInt8(at + 9 + k) / 127;
    }
    unit(forward, i * 3);
    unit(up, i * 3);
  }
  return { count, centres, forward, up };
}

export function decodePoints(payload: LiveCameras): DecodedPoints {
  const view = bytesOf(payload.points);
  const count = Math.min(payload.pointCount, Math.floor(view.byteLength / POINT_BYTES));
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const [ox = 0, oy = 0, oz = 0] = payload.origin;
  const origin = [ox, oy, oz];
  for (let i = 0; i < count; i += 1) {
    const at = i * POINT_BYTES;
    for (let k = 0; k < 3; k += 1) {
      positions[i * 3 + k] =
        (origin[k] ?? 0) + (view.getInt16(at + 2 * k, true) / INT16) * payload.scale;
      colors[i * 3 + k] = view.getUint8(at + 6 + k) / 255;
    }
  }
  return { count, positions, colors };
}

/** Which way is up in the solve's frame: the payload's estimate, else COLMAP's -y. */
export function upOf(state: Pick<LiveState, "cameras" | "splat">): [number, number, number] {
  const up = state.cameras?.up ?? state.splat?.up;
  if (up?.length === 3) {
    const [x = 0, y = 0, z = 0] = up;
    const length = Math.hypot(x, y, z);
    if (length > 1e-6) return [x / length, y / length, z / length];
  }
  // COLMAP's image y points down, so a phone held upright has world up near -y.
  return [0, -1, 0];
}

/** The median of each axis: where to look, robust to a stray point. */
export function medianCentre(positions: Float32Array): [number, number, number] | null {
  const count = Math.floor(positions.length / 3);
  if (count === 0) return null;
  const axis = (k: number): number => {
    const values: number[] = [];
    for (let i = 0; i < count; i += 1) values.push(positions[i * 3 + k] ?? 0);
    values.sort((a, b) => a - b);
    return values[Math.floor(values.length / 2)] ?? 0;
  };
  return [axis(0), axis(1), axis(2)];
}

/** Names a person reads for the stages a live run spends its time in. */
const STAGE_NAMES: Record<string, string> = {
  normalize: "Preparing the frames",
  mask: "Masking",
  pose: "Solving camera positions",
  train: "Training the splat",
  quality: "Checking quality",
  place: "Placing it",
  package: "Packaging",
  register: "Publishing",
};

export function stageName(stageId: string): string {
  return STAGE_NAMES[stageId] ?? stageId;
}

function duration(seconds: number): string {
  if (seconds < 60) return `${String(Math.max(1, Math.round(seconds)))} s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${String(minutes)} min`;
  return `${String(Math.floor(minutes / 60))} h ${String(minutes % 60)} min`;
}

const count = (value: number): string => value.toLocaleString("en-US");

export interface LiveStatus {
  headline: string;
  detail: string;
  /** 0..1 for the bar, or null when the stage reports no progress. */
  fraction: number | null;
  ended: boolean;
}

/** The status line under the live view: stage, progress, step x of y, time left. */
export function describeLive(state: LiveState): LiveStatus {
  const stage = state.stage;
  if (state.status === "complete") {
    return {
      headline: "Finished",
      detail: state.siteId
        ? "The final scan is ready."
        : "Done, but it did not become a scan anyone can open.",
      fraction: 1,
      ended: true,
    };
  }
  if (state.status === "error" || state.status === "cancelled") {
    return {
      headline: `Stopped${stage ? ` at: ${stageName(stage.stageId)}` : ""}`,
      detail: state.error ?? "",
      fraction: null,
      ended: true,
    };
  }
  if (!stage) {
    return {
      headline: "Queued",
      detail: "Waiting for the worker to pick it up.",
      fraction: null,
      ended: false,
    };
  }
  const parts: string[] = [];
  let fraction: number | null = null;
  const progress = stage.progress;
  if (stage.status === "in-progress" && progress && progress.total > 0) {
    fraction = Math.min(1, progress.done / progress.total);
    const unit = stage.stageId === "train" ? "Step" : "";
    parts.push(
      `${unit ? `${unit} ` : ""}${count(progress.done)} of ${count(progress.total)} · ${String(Math.floor(fraction * 100))}%`,
    );
    if (progress.remainingS != null) parts.push(`about ${duration(progress.remainingS)} left`);
  }
  const cameras = state.cameras;
  if (stage.stageId === "pose" && cameras) {
    parts.push(
      cameras.frames
        ? `${count(cameras.registered)} of ${count(cameras.frames)} frames placed`
        : `${count(cameras.registered)} frames placed`,
    );
  }
  const splat = state.splat;
  if (stage.stageId === "train" && splat?.stageId === "train") {
    parts.push(`showing step ${count(splat.step)}${splat.url ? "" : " (on its way)"}`);
  }
  return {
    headline: stage.status === "in-progress" ? `${stageName(stage.stageId)}…` : "Between steps…",
    detail: parts.join(" · "),
    fraction,
    ended: false,
  };
}
