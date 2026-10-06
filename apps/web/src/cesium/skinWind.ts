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
 *
 * An instance another driver has claimed (`motionClaims.ts`: telemetry moves it) is not
 * swayed and not written: its handles are that driver's.
 */

import {
  limbHandles,
  limbWindFromSettings,
  limbWindModel,
  materialPrior,
  mergeMaterial,
  skinWindFromSettings,
  skinWindModel,
  SkinWindOscillator,
  type LimbWindModel,
  type SkinMaterial,
  type SkinWindField,
  type WindSettings,
} from "@twin/world";

import type { Instance } from "@/lib/instances";
import { LIMBS_METHOD, skinFloats, type SkinDoc, type SkinEntry } from "@/lib/skin";
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
  /** What the skin itself carries of its object (a variant's skin), when the store has none. */
  readonly own: InstanceTraits | undefined;
  traits: InstanceTraits | undefined;
  material: SkinMaterial | undefined;
  oscillator: SkinWindOscillator | undefined;
  /**
   * A limbs skin's driver (`limbWind.ts`): the plant's own per-limb model, stateless, built
   * once. Its handles are limbs, never eigenmodes.
   */
  limbs: LimbWindModel | undefined;
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

/**
 * The traits a skin carries itself (`skin.json`'s `traits`, a variant's skin): what the prior
 * reads when the scan's `instances.json` does not list the instance (the Minnetonka tree has
 * none). The store's record wins whenever there is one.
 */
export function skinTraitsOf(skin: Pick<SkinEntry, "traits">): InstanceTraits | undefined {
  const t = skin.traits;
  if (!t) return undefined;
  return {
    properties: t.properties ?? {},
    behaviour: (t.behaviour ?? "in-place") as Instance["behaviour"],
  };
}

/** Drives every skin of one scan's skin part. */
export class SkinWindDriver {
  readonly target: SkinWindTarget;
  readonly #describe: DescribeInstance;
  readonly #field: SkinWindField;
  readonly #skins: Driven[];
  readonly #claimed: (instanceId: number) => boolean;
  #materials: MaterialTable | undefined;

  constructor(
    target: SkinWindTarget,
    describe: DescribeInstance,
    field: SkinWindField,
    claimed: (instanceId: number) => boolean = () => false,
  ) {
    this.target = target;
    this.#describe = describe;
    this.#field = field;
    this.#claimed = claimed;
    this.#skins = target.doc.skins.map((skin) => ({
      skin,
      own: skin.traits ? skinTraitsOf(skin) : undefined,
      traits: undefined,
      material: undefined,
      oscillator: undefined,
      limbs: skin.limbs ? limbWindModel(skin.limbs) : undefined,
      buffer: new Float64Array(skinFloats(skin)),
      displaced: false,
    }));
  }

  /** Whether another driver holds instance `instanceId` (the wind leaves it alone). */
  claimed(instanceId: number): boolean {
    return this.#claimed(instanceId);
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
      const traits = this.#describe(d.skin.instance) ?? d.own;
      if (refresh || traits !== d.traits || d.material === undefined) this.#resolve(d, traits);
      if (this.#claimed(d.skin.instance)) {
        // Another driver writes these handles now: drop the state, write nothing.
        d.oscillator?.reset();
        d.displaced = false;
        continue;
      }
      // A limbs skin sways by its plant's own model; any other by its eigenmodes.
      const limbs = d.material?.wind ? d.limbs : undefined;
      const oscillator = d.material?.wind && !limbs ? d.oscillator : undefined;
      if (oscillator || limbs) driven += 1;
      if (calm || (!oscillator && !limbs)) {
        d.oscillator?.reset();
        if (d.displaced) {
          this.target.setInstanceHandles(d.skin.instance, null);
          d.displaced = false;
          changed = true;
        }
        continue;
      }
      if (limbs) {
        // Stateless: the frame is a function of the scene time, as the rig's is.
        const living = limbWindFromSettings(wind, limbs.source);
        this.target.setInstanceHandles(d.skin.instance, limbHandles(limbs, t, living, d.buffer));
      } else if (oscillator) {
        oscillator.advance(this.#field, settings, t);
        this.target.setInstanceHandles(d.skin.instance, oscillator.handles(t, d.buffer));
      }
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
      if (d.displaced && !this.#claimed(d.skin.instance))
        this.target.setInstanceHandles(d.skin.instance, null);
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
    // A limbs skin's handles are limbs: its dynamics are the poke's, never an eigen-sway's.
    const dynamics =
      d.skin.limbs || this.target.doc.method === LIMBS_METHOD ? undefined : d.skin.dynamics;
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
