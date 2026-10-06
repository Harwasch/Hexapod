/**
 * Method variants of a scan (the owner's bake-offs): other methods' outputs for the same
 * measured splats, published beside its tiles under `variants/<system>/<name>/` and declared
 * on the measured tileset's root as `extras.variants` (docs/SCENE_OBJECTS.md, "Variants"):
 *
 *     "variants": {
 *       "objects": [{ "name", "label", "about", "instances": "variants/objects/<name>/instances.json" }],
 *       "fill":    [{ "name", "label", "about", "inferredLayers": [{ "uri", "evidence" }] }],
 *       "skins":   [{ "name", "label", "about", "skin": "variants/skins/<name>/skin.json" }]
 *     }
 *
 * Paths are relative to the measured `tileset.json`, as `extras.instances`, `extras.skin` and
 * `extras.inferredLayers` are, and resolve the same way (`resolveLayerUrl`, keeping a signed
 * URL's query). The files are today's formats. Today's own files stay the default ("Today"):
 * a viewer who never picks a variant sees exactly what the scan published before. An entry
 * may also carry `look`: one short line on what to look for to judge it, in place of its
 * system's (`LOOK_FOR`).
 *
 * Read defensively, like `inferredLayersOf`: a malformed entry costs that entry, not the list.
 */

import { inferredLayersOf, resolveLayerUrl, type InferredLayerRef } from "./inferred";
import { instancesRefOf, type InstancesRef } from "./instances";
import { skinRefOf, type SkinRef } from "./skin";

/** The systems a bake-off compares, as `extras.variants` keys them. */
export type VariantSystem = "objects" | "fill" | "skins";

/** Every system, in the order the viewer lists them. */
export const VARIANT_SYSTEMS: readonly VariantSystem[] = ["objects", "fill", "skins"];

/** What the viewer calls each system. */
export const SYSTEM_LABELS: Record<VariantSystem, string> = {
  objects: "Objects",
  fill: "Fill",
  skins: "Motion",
};

/** What every variant carries. */
interface VariantBase {
  /** Unique within its system; the folder name (`variants/<system>/<name>/`). */
  name: string;
  /** What the viewer shows, visible (bake-offs are not blind). */
  label: string;
  /** One plain sentence on what the method does; may be empty. */
  about: string;
  /**
   * What to look for to judge it, in one short line, when the method has something of its own
   * to say; otherwise the viewer shows its system's (`LOOK_FOR`).
   */
  look?: string;
}

/**
 * What to look for to judge each system, shown under the pick's `about` (a variant's own
 * `look` replaces it): where in the app the difference between methods shows.
 */
export const LOOK_FOR: Record<VariantSystem, string> = {
  objects:
    "Click things: the first click should pick the whole object, the next its part. Ground types are listed under Ground & soil in the Objects panel.",
  fill: "Set Inferred to Highlight: generated splats turn purple. Orbit to check they look real from every side.",
  skins:
    "Turn wind up in Settings › Simulated wind and watch trees and shrubs; rigid things like the spool should not bend.",
};

export interface ObjectsVariant extends VariantBase {
  /** Its `instances.json`, relative to the measured tileset. */
  instances: string;
}

export interface FillVariant extends VariantBase {
  /** Its inferred layers, as `extras.inferredLayers` lists them. Empty: a "no fill" method. */
  inferredLayers: InferredLayerRef[];
}

export interface SkinsVariant extends VariantBase {
  /** Its `skin.json` (beside its `skin.bin`), relative to the measured tileset. */
  skin: string;
}

export interface ScanVariants {
  objects: ObjectsVariant[];
  fill: FillVariant[];
  skins: SkinsVariant[];
}

export type VariantOf<S extends VariantSystem> = ScanVariants[S][number];

/** A scan that declares none. */
export const NO_VARIANTS: ScanVariants = Object.freeze({ objects: [], fill: [], skins: [] });

/** A path field: a non-empty string, or `{ uri }` as `extras.instances` and `extras.skin`. */
function pathOf(value: unknown): string | null {
  if (typeof value === "string") return value.trim() === "" ? null : value;
  const uri = (value as { uri?: unknown } | null | undefined)?.uri;
  return typeof uri === "string" && uri.trim() !== "" ? uri : null;
}

function textOf(value: unknown): string | null {
  return typeof value === "string" && value.trim() !== "" ? value.trim() : null;
}

/** The name, label and about of an entry, or null when it has no usable name. */
function baseOf(entry: unknown): VariantBase | null {
  const e = entry as Record<string, unknown> | null | undefined;
  if (typeof e !== "object" || e === null) return null;
  const name = textOf(e.name);
  if (name === null) return null;
  const look = textOf(e.look);
  return {
    name,
    label: textOf(e.label) ?? name,
    about: textOf(e.about) ?? "",
    ...(look === null ? {} : { look }),
  };
}

/** What to look for with `variant` of `system` picked (Today: `null`). */
export function lookFor(system: VariantSystem, variant: { look?: string } | null): string {
  return variant?.look ?? LOOK_FOR[system];
}

/** Reads one system's list: entries `read` accepts, the first of each name. */
function listOf<T extends VariantBase>(
  value: unknown,
  read: (entry: Record<string, unknown>, base: VariantBase) => T | null,
): T[] {
  if (!Array.isArray(value)) return [];
  const out: T[] = [];
  const names = new Set<string>();
  for (const entry of value as unknown[]) {
    const base = baseOf(entry);
    if (base === null || names.has(base.name)) continue;
    const variant = read(entry as Record<string, unknown>, base);
    if (variant === null) continue;
    names.add(base.name);
    out.push(variant);
  }
  return out;
}

/**
 * The variants a measured tileset's root declares (`root.extras`): every well-formed entry of
 * each system, in the published order. An entry needs a `name` and its system's file
 * (`instances`, `skin`, or an `inferredLayers` list whose entries all read); a repeated name
 * keeps the first.
 */
export function variantsOf(extras: unknown): ScanVariants {
  const v = (extras as { variants?: Record<string, unknown> } | null | undefined)?.variants;
  if (typeof v !== "object" || v === null || Array.isArray(v)) return NO_VARIANTS;
  const objects = listOf<ObjectsVariant>(v.objects, (e, base) => {
    const instances = pathOf(e.instances);
    return instances === null ? null : { ...base, instances };
  });
  const fill = listOf<FillVariant>(v.fill, (e, base) => {
    if (!Array.isArray(e.inferredLayers)) return null;
    const inferredLayers = inferredLayersOf({ inferredLayers: e.inferredLayers });
    // A list with an entry that does not read is malformed: half a method is not the method.
    if (inferredLayers.length !== e.inferredLayers.length) return null;
    return { ...base, inferredLayers };
  });
  const skins = listOf<SkinsVariant>(v.skins, (e, base) => {
    const skin = pathOf(e.skin);
    return skin === null ? null : { ...base, skin };
  });
  if (objects.length + fill.length + skins.length === 0) return NO_VARIANTS;
  return { objects, fill, skins };
}

/** Per system, whether the scan publishes its own -- Today's -- files for it. */
export function todayOf(extras: unknown): Record<VariantSystem, boolean> {
  return {
    objects: instancesRefOf(extras) !== null,
    fill: inferredLayersOf(extras).length > 0,
    skins: skinRefOf(extras) !== null,
  };
}

/** Whether a scan offers any variant at all. */
export function hasVariants(variants: ScanVariants): boolean {
  return variants.objects.length + variants.fill.length + variants.skins.length > 0;
}

/** A variant file's URL beside the measured tileset's, keeping its query (a signed URL's). */
export function resolveVariantUrl(tilesetUrl: string, uri: string): string {
  return resolveLayerUrl(tilesetUrl, uri);
}

/** The variant named `name` of `system`, or null (Today, or a name no longer offered). */
export function findVariant<S extends VariantSystem>(
  variants: ScanVariants,
  system: S,
  name: string | null | undefined,
): VariantOf<S> | null {
  if (name === null || name === undefined) return null;
  return (variants[system] as VariantOf<S>[]).find((v) => v.name === name) ?? null;
}

/** The instances a scan draws with `picked` (Today when null): the variant's, or its own. */
export function instancesRefFor(
  extras: unknown,
  picked: ObjectsVariant | null,
): InstancesRef | null {
  return picked ? { uri: picked.instances, count: 0 } : instancesRefOf(extras);
}

/** The skin a scan moves by with `picked` (Today when null): the variant's, or its own. */
export function skinRefFor(extras: unknown, picked: SkinsVariant | null): SkinRef | null {
  return picked ? { uri: picked.skin, count: 0 } : skinRefOf(extras);
}

/** The inferred layers a scan draws with `picked` (Today when null). */
export function inferredLayersFor(extras: unknown, picked: FillVariant | null): InferredLayerRef[] {
  return picked ? picked.inferredLayers : inferredLayersOf(extras);
}
