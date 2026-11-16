/**
 * The Ask contract between the iPhone app and the reasoner: an AskRequest goes up, a stream of AskEvents
 * comes back. Hand-written to mirror apps/ios/Modules/EntitySchema (AskContract.swift, Entity.swift) and
 * its JSON Schemas (schema/ask.schema.json, schema/entity.schema.json). Change all three together.
 */

export type JsonValue =
  null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };

export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

/** WGS84 degrees and altitude in metres above the ellipsoid. */
export interface GeoCoordinate {
  lat: number;
  lon: number;
  alt: number;
}

export interface Entity {
  label: string;
  /** `world`: ARKit session coordinates in metres, right-handed, Y up. */
  anchor: { world: Vector3; geo?: GeoCoordinate };
  /** Axis-aligned dimensions in metres. */
  extent: Vector3;
  masks: { keyframe: string; ref: string }[];
  confidence: number;
  attrs: Record<string, JsonValue>;
  source: { module: string; model: string };
}

export interface KeyframeRef {
  /** Frame index in the bundle's frames.jsonl. */
  index: number;
  image: string;
  depth?: string;
  /** `frames.jsonl#<index>`. */
  pose: string;
}

export interface AskRequest {
  id: string;
  question: string;
  scanId?: string;
  location?: GeoCoordinate;
  keyframes: KeyframeRef[];
  deviceEntities: Entity[];
}

export type AskEvent =
  | { type: "entity"; entity: Entity }
  | { type: "speech"; text: string }
  | { type: "status"; text: string }
  | { type: "done" };
