import type { CaptureKind } from "@twin/contracts";

/**
 * What the dropped files are, and what should be done with them — in plain language.
 *
 * This is the step that turns "supports different capture methods" into a user-facing
 * sentence instead of a configuration file: the console looks at what it was given,
 * proposes a recipe, and says roughly how long it will take.
 */
export interface Proposal {
  kind: CaptureKind;
  /** A recipe the API knows: `splat-ingest` (lane 1) or `photo-reconstruct` (lane 2). */
  recipe: "splat-ingest" | "photo-reconstruct";
  /** "Package and place" / "Reconstruct" — the verb, for a button and a badge. */
  action: string;
  /** One line: what was recognised and what will happen to it. */
  summary: string;
  /** Rough wall-clock, deliberately vague: "About 30 seconds". */
  estimate: string;
}

/*
 * Exactly what the pipeline reads, and no more: `SPLAT_SUFFIXES` in
 * tools/pipeline/gaussians.py and `VIDEO_SUFFIXES` / `IMAGE_SUFFIXES` in
 * tools/pipeline/video.py. Offering a format the pipeline refuses turns a clear "not
 * this file" at the drop into a failed run minutes later, so a format joins these sets
 * when a stage learns to read it. HEIC is absent on purpose: iOS converts photos to JPEG
 * when a picker does not list HEIC, so the phone page gets JPEGs without asking.
 */
const SPLAT = new Set(["ply", "spz"]);
const VIDEO = new Set(["mp4", "mov", "m4v", "avi", "mkv", "webm"]);
const IMAGE = new Set(["jpg", "jpeg", "png"]);

/** The file picker's `accept`, built from the same sets so the two cannot drift. */
export const ACCEPT = [
  "video/mp4",
  "video/quicktime",
  "image/jpeg",
  "image/png",
  ...[...VIDEO, ...IMAGE, ...SPLAT].map((ext) => `.${ext}`),
].join(",");

/** Plain-language list for an error message. */
export const SUPPORTED_TEXT =
  "a video (.mp4, .mov, .m4v, .avi, .mkv, .webm), photos (.jpg, .png) or a splat (.ply, .spz)";

/** The files the pipeline could not read, by name. Empty means the drop is fine. */
export function unsupported(files: { name: string }[]): string[] {
  return files
    .filter((file) => {
      const ext = extensionOf(file.name);
      return !SPLAT.has(ext) && !VIDEO.has(ext) && !IMAGE.has(ext);
    })
    .map((file) => file.name);
}

export function extensionOf(filename: string): string {
  const base = filename.replace(/\\/g, "/").split("/").pop() ?? filename;
  const dot = base.lastIndexOf(".");
  return dot <= 0 ? "" : base.slice(dot + 1).toLowerCase();
}

const GIB = 1024 ** 3;

/**
 * A0 clocked an 11 GB iPhone video at roughly 40 minutes of cloud-GPU reconstruction,
 * so about 3.5 minutes per gigabyte, with a floor for the fixed cost of a short clip.
 * It is an estimate shown to a person, not a scheduling input.
 */
function reconstructEstimate(bytes: number): string {
  const minutes = Math.max(8, Math.round((bytes / GIB) * 3.5));
  if (minutes < 60) return `About ${minutes} minutes on the cloud GPU`;
  const hours = Math.round((minutes / 60) * 10) / 10;
  return `About ${hours} hours on the cloud GPU`;
}

export function classify(files: { name: string; size: number }[]): Proposal {
  const extensions = files.map((file) => extensionOf(file.name));
  const bytes = files.reduce((sum, file) => sum + file.size, 0);
  const count = files.length;
  const noun = count === 1 ? "file" : "files";

  if (extensions.some((ext) => SPLAT.has(ext))) {
    return {
      kind: "gaussian-splat",
      recipe: "splat-ingest",
      action: "Package and place",
      summary: `${count} splat ${noun} → placed as is`,
      estimate: "About 30 seconds",
    };
  }
  if (extensions.some((ext) => VIDEO.has(ext))) {
    return {
      kind: "video",
      recipe: "photo-reconstruct",
      action: "Reconstruct",
      summary: `${count} ${count === 1 ? "video" : "videos"} → 3D model`,
      estimate: reconstructEstimate(bytes),
    };
  }
  if (extensions.some((ext) => IMAGE.has(ext))) {
    return {
      kind: "images",
      recipe: "photo-reconstruct",
      action: "Reconstruct",
      summary: `${count} ${count === 1 ? "photo" : "photos"} → 3D model`,
      estimate: reconstructEstimate(bytes),
    };
  }
  // Nothing recognised. `unsupported` refuses such a drop before it gets here; this is
  // only what a card shows for a capture whose files predate that check.
  return {
    kind: "images",
    recipe: "photo-reconstruct",
    action: "Reconstruct",
    summary: `${count} unsupported ${noun}`,
    estimate: reconstructEstimate(bytes),
  };
}

/** A name for the capture, from what was dropped: one file keeps its name, many get a count. */
export function captureName(files: { name: string }[]): string {
  const first = files[0]?.name ?? "Capture";
  if (files.length === 1) return first.replace(/\.[^.]+$/, "") || first;
  return `${first.replace(/\.[^.]+$/, "")} + ${files.length - 1} more`;
}
