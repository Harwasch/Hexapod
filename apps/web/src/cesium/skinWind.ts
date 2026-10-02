/**
 * The wind driver of scene-object skins (step C1, docs/SCENE_OBJECTS.md §4): per skinned
 * instance, the anchored modal model of `@twin/world`'s `skinWind.ts`, advanced on the scene
 * clock and written through the skin part's one driver interface,
 * `setInstanceHandles(instanceId, Z)` (`splatSkin.ts`).
 *
 * Which instances sway, and how, is a material per instance: its property prior
 * (`materialPrior`: behaviour `in-place` sways, `movable` does not; vegetation is soft, damped
 * and catches the wind) overridden field by field by `materials.json` (what the video teacher
 * fits, `lib/skinMaterials.ts`). In calm every driven skin is handed `null`: the scene is the
 * measured one, pixel for pixel, the same frame the wind drops.
 */

import {
  materialPrior,
  mergeMaterial,
  skinWindFromSettings,
  skinWindModel,
  SkinWindOscillator,
  type SkinMaterial,
  type SkinWindField,
  type WindSettings,
} from "@twin/world";

import type { Instance } from "@/lib/instances";
import type { SkinDoc, SkinEntry } from "@/lib/skin";
import type { MaterialTable } from "@/lib/skinMaterials";
import { useInstances } from "@/state/instances";

/** What the driver needs of a skin part (`SplatSkinning`). */
export interface SkinWindTarget {
  readonly doc: SkinDoc;
  readonly materials: MaterialTable;
  setInstanceHandles(instanceId: number, handles: ArrayLike<number> | null): boolean;
}

/** What the prior reads of an instance. */
export type InstanceTraits = Pick<Instance, "properties" | "behaviour">;

/** An instance's traits by id, or undefined while its table has not loaded. */
export type DescribeInstance = (instanceId: number) => InstanceTraits | undefined;

/** Instance traits from the viewer's store (`state/instances.ts`). */
export function describeFromStore(assetId: string): DescribeInstance {
  let table: readonly Instance[] | undefined;
  let byId = new Map<number, Instance>();
  return (id) => {
    const instances = useInstances.getState().assets[assetId]?.instances;
    if (instances !== table) {
      table = instances;
      byId = new Map((instances ?? []).map((i) => [i.id, i]));
    }
    return byId.get(id);
  };
}

interface Driven {
  readonly skin: SkinEntry;
  traits: InstanceTraits | undefined;
  material: SkinMaterial | undefined;
  oscillator: SkinWindOscillator | undefined;
  buffer: Float64Array;
  /** The part holds handles for it (not `null`). */
  displaced: boolean;
}

/** One frame's outcome. */
export interface SkinWindTick {
  /** Skins that moved this frame. */
  readonly moving: number;
  /** Skins the wind drives at all (material says so, and the skin carries its dynamics). */
  readonly driven: number;
  /** Something changed on screen: a render is owed. */
  readonly changed: boolean;
}

/** Drives every skin of one scan's skin part. */
export class SkinWindDriver {
  readonly target: SkinWindTarget;
  readonly #describe: DescribeInstance;
  readonly #field: SkinWindField;
  readonly #skins: Driven[];
  #materials: MaterialTable | undefined;

  constructor(target: SkinWindTarget, describe: DescribeInstance, field: SkinWindField) {
    this.target = target;
    this.#describe = describe;
    this.#field = field;
    this.#skins = target.doc.skins.map((skin) => ({
      skin,
      traits: undefined,
      material: undefined,
      oscillator: undefined,
      buffer: new Float64Array(skin.handles * 12),
      displaced: false,
    }));
  }

  /** The material instance `instanceId` sways with now, once its traits are known. */
  material(instanceId: number): SkinMaterial | undefined {
    return this.#skins.find((d) => d.skin.instance === instanceId)?.material;
  }

  /** Advances every driven skin to scene time `t` under `wind` and hands the part its handles. */
  tick(t: number, wind: WindSettings): SkinWindTick {
    const settings = skinWindFromSettings(wind);
    const calm = !(settings.speedMps > 0);
    const refresh = this.target.materials !== this.#materials;
    this.#materials = this.target.materials;
    let moving = 0;
    let driven = 0;
    let changed = false;
    for (const d of this.#skins) {
      const traits = this.#describe(d.skin.instance);
      if (refresh || traits !== d.traits || d.material === undefined) this.#resolve(d, traits);
      const oscillator = d.material?.wind ? d.oscillator : undefined;
      if (oscillator) driven += 1;
      if (calm || !oscillator) {
        d.oscillator?.reset();
        if (d.displaced) {
          this.target.setInstanceHandles(d.skin.instance, null);
          d.displaced = false;
          changed = true;
        }
        continue;
      }
      oscillator.advance(this.#field, settings, t);
      this.target.setInstanceHandles(d.skin.instance, oscillator.handles(t, d.buffer));
      d.displaced = true;
      moving += 1;
      changed = true;
    }
    return { moving, driven, changed };
  }

  /** Every skin back at rest (handles `null`), state dropped. */
  rest(): void {
    for (const d of this.#skins) {
      d.oscillator?.reset();
      if (d.displaced) this.target.setInstanceHandles(d.skin.instance, null);
      d.displaced = false;
    }
  }

  #resolve(d: Driven, traits: InstanceTraits | undefined): void {
    d.traits = traits;
    const record = this.target.materials.get(d.skin.instance);
    if (traits === undefined && record === undefined) {
      d.material = undefined;
      d.oscillator = undefined;
      return;
    }
    const material = mergeMaterial(materialPrior(traits?.properties, traits?.behaviour), record);
    const before = d.material;
    const same =
      before?.stiffness === material.stiffness &&
      before.damping === material.damping &&
      before.drag === material.drag;
    d.material = material;
    if (same && d.oscillator) return;
    const dynamics = d.skin.dynamics;
    const model = dynamics
      ? skinWindModel(
          {
            handles: d.skin.handles,
            origin: d.skin.origin,
            scale: d.skin.scale,
            eigenvalues: d.skin.eigenvalues,
            support: d.skin.support,
            mass: dynamics.mass,
            anchorGram: dynamics.anchorGram,
          },
          material,
        )
      : undefined;
    d.oscillator = model ? new SkinWindOscillator(model) : undefined;
  }
}
