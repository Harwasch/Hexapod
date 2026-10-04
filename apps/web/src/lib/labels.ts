/**
 * Plain words for the API's enum values, shared by the globe app and the data console so a
 * thing has one name on every page (docs/GLOSSARY.md). The raw value stays wherever a test or
 * a curious reader needs it (`data-status`, a `title`); people read these.
 *
 * Pure and free of the scene: `admin.html` imports it without pulling in a globe.
 */

/** A capture's state, from the person who sent it: "Ready" means uploaded, not yet processed. */
export const CAPTURE_STATUS: Readonly<Record<string, string>> = {
  "awaiting-files": "Waiting for files",
  "not-started": "Ready",
  "in-progress": "Processing",
  complete: "Done",
  error: "Failed",
  cancelled: "Cancelled",
};

/** A run's (or one of its stages') state. */
export const RUN_STATUS: Readonly<Record<string, string>> = {
  "not-started": "Queued",
  "in-progress": "Running",
  complete: "Done",
  error: "Failed",
  cancelled: "Cancelled",
};

/** A file's upload state. */
export const UPLOAD_STATUS: Readonly<Record<string, string>> = {
  "not-started": "Waiting",
  "in-progress": "Uploading",
  complete: "Uploaded",
  error: "Failed",
  aborted: "Stopped",
};

/** What a capture is made of. */
export const CAPTURE_KIND: Readonly<Record<string, string>> = {
  video: "Video",
  images: "Photos",
  "gaussian-splat": "Splat",
  "point-cloud": "Point cloud",
};

/** What a run wrote. Representations read as they do on the Splat / Mesh / Points strip. */
export const ARTIFACT_KIND: Readonly<Record<string, string>> = {
  frames: "Frames",
  poses: "Camera poses",
  masks: "Masks",
  splat: "Splat",
  "deformation-field": "Deformation field",
  mesh: "Mesh",
  "point-cloud": "Points",
  "3d-tiles": "3D Tiles",
  thumbnail: "Thumbnail",
  "ground-samples": "Ground samples",
  manifest: "Manifest",
  clip: "Clip",
  metadata: "Metadata",
};

function label(table: Readonly<Record<string, string>>, value: string): string {
  return table[value] ?? value;
}

export const captureStatusLabel = (value: string): string => label(CAPTURE_STATUS, value);
export const runStatusLabel = (value: string): string => label(RUN_STATUS, value);
export const uploadStatusLabel = (value: string): string => label(UPLOAD_STATUS, value);
export const captureKindLabel = (value: string): string => label(CAPTURE_KIND, value);
export const artifactKindLabel = (value: string): string => label(ARTIFACT_KIND, value);
