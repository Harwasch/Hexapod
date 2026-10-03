/**
 * Per-instance materials for the skin wind (hexapod.materials v1, docs/SCENE_OBJECTS.md §4):
 * what the video teacher (step C2) fits and writes beside the tiles as `materials.json`,
 * declared on their root as `extras.materials = { uri, count }`.
 *
 * Every field of a record is optional; what a record leaves out comes from the instance's
 * property prior (`materialPrior` in `@twin/world`). A scan without the file sways on priors.
 */

import { MOTION_EVIDENCE_LADDER, type MaterialEvidence, type SkinMaterial } from "@twin/world";

import { resolveBeside } from "./instances";

export const MATERIALS_FORMAT = "hexapod.materials";
export const MATERIALS_VERSION = 1;

/** `root.extras.materials`. */
export interface MaterialsRef {
  uri: string;
  count: number;
}

/** Per instance id, the fields its record sets. */
export type MaterialTable = ReadonlyMap<number, Partial<SkinMaterial>>;

/** `extras.materials` off a tileset's root tile; null when absent or malformed. */
export function materialsRefOf(extras: unknown): MaterialsRef | null {
  const ref = (extras as { materials?: Record<string, unknown> } | null | undefined)?.materials;
  if (typeof ref?.uri !== "string" || ref.uri.length === 0) return null;
  const count = typeof ref.count === "number" && Number.isFinite(ref.count) ? ref.count : 0;
  return { uri: ref.uri, count: Math.max(0, Math.round(count)) };
}

const EVIDENCE = new Set<string>([...MOTION_EVIDENCE_LADDER, "prior"]);

function positive(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : undefined;
}

/**
 * Reads a `materials.json` document: null when it is not one, otherwise every record with an
 * instance id, keeping only the fields that are well formed.
 */
export function parseMaterials(raw: unknown): MaterialTable | null {
  const doc = raw as Record<string, unknown> | null | undefined;
  if (doc?.format !== MATERIALS_FORMAT || doc.version !== MATERIALS_VERSION) return null;
  if (!Array.isArray(doc.materials)) return null;
  const table = new Map<number, Partial<SkinMaterial>>();
  for (const entry of doc.materials as unknown[]) {
    const r = entry as Record<string, unknown> | null;
    if (typeof r !== "object" || r === null) continue;
    const instance = r.instance;
    if (!Number.isInteger(instance) || (instance as number) < 1) continue;
    if (table.has(instance as number)) continue;
    const record: {
      -readonly [K in keyof SkinMaterial]?: SkinMaterial[K];
    } = {};
    const stiffness = positive(r.stiffness);
    if (stiffness !== undefined) record.stiffness = stiffness;
    if (typeof r.damping === "number" && Number.isFinite(r.damping) && r.damping >= 0)
      record.damping = r.damping;
    if (typeof r.drag === "number" && Number.isFinite(r.drag) && r.drag >= 0) record.drag = r.drag;
    if (typeof r.wind === "boolean") record.wind = r.wind;
    if (typeof r.evidence === "string" && EVIDENCE.has(r.evidence))
      record.evidence = r.evidence as MaterialEvidence;
    table.set(instance as number, record);
  }
  return table;
}

/** Fetches and reads a scan's `materials.json`. Throws when it is missing or not one. */
export async function loadMaterials(tilesetUrl: string, ref: MaterialsRef): Promise<MaterialTable> {
  const response = await fetch(resolveBeside(tilesetUrl, ref.uri));
  if (!response.ok) throw new Error(`materials answered ${String(response.status)}`);
  const table = parseMaterials(await response.json());
  if (!table) throw new Error("materials: not a hexapod.materials v1 document");
  return table;
}
