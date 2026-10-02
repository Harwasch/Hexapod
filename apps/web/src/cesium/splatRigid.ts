/**
 * Moves whole scene objects rigidly on the GPU, keyed by instance (step C3,
 * docs/SCENE_OBJECTS.md §4 "Telemetry on instances"): the motion of an object that has no
 * skin. A skinned object moves through its skin's constant handle instead (`splatSkin.ts`);
 * this part needs nothing but the per-splat instance ids the scan already carries
 * (`splatInstances.ts`), so any segmented object can be driven.
 *
 * Per driven instance a **slot** holds one rigid motion `x' = R x + t` (scan frame), folded
 * into the frame the shader sees as `foldHandle` folds a skin's handle (`A_b = L(R − I)L⁻¹`,
 * `t_b`), so the shader adds `A_b x_b + t_b`. Two textures of our own:
 *
 * - **slots** (RGBA32UI, 1024 texels a row, four ids a texel): the slot of every instance id,
 *   0 for none. Driving an instance writes its slot at it and at every instance below it
 *   (splats carry leaf ids); rewritten only when what is driven changes.
 * - **poses** (RGBA32F, 1024 texels a row, three a slot): the folded rows `[A_b | t_b]`.
 *
 * and the instance hooks' id texture, shared, read as `splatInstanceId` reads it.
 *
 * A motion chain part (`splatMotionChain.ts`): it composes with a skin and the Living
 * Survey's rig (each sees the rest position) and gives the chain its linear part, so the
 * splats' covariances turn with the object. Nothing driven costs nothing (`u_rigidActive` 0);
 * a splat of an undriven instance costs one id fetch and one slot fetch. If one driven
 * instance is below another, the deeper one's motion wins for its splats.
 *
 * **Sorted where drawn**: each splat's slot is its group for the sorter (`setSortMotion`), so
 * a driven object is ordered back to front at its moved place, not its rest place.
 */

import { RIGID_IDENTITY, type RigidMotion } from "@twin/world";

import { withDescendants, type InstancesDoc } from "@/lib/instances";
import { HANDLE_FLOATS, rigidHandle } from "@/lib/skin";

import { invertAffine } from "./splatFrames";
import type { MotionTextureFactory, OwnedTexture } from "./splatGpuMotion";
import type { SplatPrimitive, SplatShaderBuilder, SplatTilesetLike } from "./splatInternals";
import {
  addMotionPart,
  hasMotionPart,
  removeMotionPart,
  type SplatMotionPart,
} from "./splatMotionChain";
import { foldHandle } from "./splatSkin";
import { setSortMotion } from "./splatSorter";
import { sameMatrix } from "./splatTiles";

/** Texels a row of either texture holds. */
export const RIGID_TEXTURE_WIDTH = 1024;
/** Instance ids a slot texel holds. */
export const SLOTS_PER_TEXEL = 4;
/** Pose texels a slot takes. */
export const TEXELS_PER_SLOT = 3;
const FLOATS_PER_TEXEL = 4;

/** What the part needs of the instance hooks (`SplatInstances`). */
export interface InstanceIdSource {
  readonly doc: Pick<InstancesDoc, "instances" | "maxId">;
  /** RGBA32UI, four ids a texel (`splatInstanceId`'s layout). */
  readonly idTexture: unknown;
  readonly covered: number;
  readonly idsCurrent: boolean;
  /** The id of every splat index (CPU side), and a counter bumped when they change. */
  readonly ids: ArrayLike<number>;
  readonly idsVersion: number;
  /** Tileset local → the frame the shader sees. */
  readonly bake: readonly number[] | undefined;
}

/** The shader side: a displacement and its linear part. */
export function rigidGlsl(): string {
  const mask = String(RIGID_TEXTURE_WIDTH - 1);
  const shift = String(Math.log2(RIGID_TEXTURE_WIDTH));
  return `
mat3 splatRigidLinear = mat3(0.0);

uint splatRigidQuad(highp usampler2D quads, uint index) {
    uint q = index >> 2u;
    uvec4 t = texelFetch(quads, ivec2(int(q & ${mask}u), int(q >> ${shift}u)), 0);
    uint k = index & 3u;
    return k == 0u ? t.r : (k == 1u ? t.g : (k == 2u ? t.b : t.a));
}

vec3 splatRigidMotion(uint splatIndex, vec3 position) {
    splatRigidLinear = mat3(0.0);
    if (u_rigidActive < 0.5 || float(splatIndex) >= u_rigidCovered) {
        return vec3(0.0);
    }
    uint id = splatRigidQuad(u_rigidIds, splatIndex);
    if (id == 0u || float(id) > u_rigidMaxId) {
        return vec3(0.0);
    }
    uint slot = splatRigidQuad(u_rigidSlots, id);
    if (slot == 0u) {
        return vec3(0.0);
    }
    int at = int(slot) * ${String(TEXELS_PER_SLOT)};
    vec4 r0 = texelFetch(u_rigidPoses, ivec2(at & ${mask}, at >> ${shift}), 0);
    vec4 r1 = texelFetch(u_rigidPoses, ivec2((at + 1) & ${mask}, (at + 1) >> ${shift}), 0);
    vec4 r2 = texelFetch(u_rigidPoses, ivec2((at + 2) & ${mask}, (at + 2) >> ${shift}), 0);
    vec4 xh = vec4(position, 1.0);
    splatRigidLinear = mat3(r0.x, r1.x, r2.x, r0.y, r1.y, r2.y, r0.z, r1.z, r2.z);
    return vec3(dot(r0, xh), dot(r1, xh), dot(r2, xh));
}

mat3 splatRigidJacobian(uint splatIndex, vec3 position) {
    return splatRigidLinear;
}
`;
}

/** What the shader computes for one splat of slot `slot`, in float64. A reference for tests. */
export function evaluateRigidMotion(
  poses: Float32Array,
  slot: number,
  position: readonly [number, number, number],
): [number, number, number] {
  const out: [number, number, number] = [0, 0, 0];
  if (slot === 0) return out;
  for (let r = 0; r < 3; r += 1) {
    const o = (slot * TEXELS_PER_SLOT + r) * FLOATS_PER_TEXEL;
    out[r] =
      (poses[o] ?? 0) * position[0] +
      (poses[o + 1] ?? 0) * position[1] +
      (poses[o + 2] ?? 0) * position[2] +
      (poses[o + 3] ?? 0);
  }
  return out;
}

function rowsFor(count: number, perRow: number): number {
  return Math.max(1, Math.ceil(count / perRow));
}

/** The rigid part of one scan's motion chain. */
export class SplatRigidMotion implements SplatMotionPart {
  readonly motionFunction = "splatRigidMotion";
  readonly jacobianFunction = "splatRigidJacobian";
  /** After the skin: the order is immaterial (every part sees the rest position). */
  readonly motionOrder = 20;
  readonly #ids: InstanceIdSource;
  readonly #factory: MotionTextureFactory;
  readonly #tileset: SplatTilesetLike;
  #primitive: SplatPrimitive | undefined;
  #context: unknown;
  /** Slot by instance id (driven ones only). */
  readonly #slotOf = new Map<number, number>();
  /** Motion by slot, as last set (scan frame). */
  readonly #motions = new Map<number, RigidMotion>();
  /** Motion by driven instance id, as last set: what any renderer draws (`instanceMotions`). */
  readonly #byInstance = new Map<number, RigidMotion>();
  #motionVersion = 0;
  readonly #slots: Uint32Array;
  #poses: Float32Array;
  #slotTexture: OwnedTexture | undefined;
  #poseTexture: OwnedTexture | undefined;
  #emptyIds: OwnedTexture | undefined;
  #slotsDirty = true;
  #posesDirty = true;
  #poseRows = 0;
  #bake: readonly number[] | undefined;
  #inverse: number[] | undefined;
  #slotsVersion = 0;
  #groups: Uint16Array | undefined;
  #groupsKey = "";
  #sortDirty = true;
  /** The primitive the sorter was last told about. */
  #sorted: SplatPrimitive | undefined;
  /** Off draws every splat at rest; the part stays installed. */
  enabled = true;

  constructor(ids: InstanceIdSource, factory: MotionTextureFactory, tileset: SplatTilesetLike) {
    this.#ids = ids;
    this.#factory = factory;
    this.#tileset = tileset;
    this.#slots = new Uint32Array(
      rowsFor(ids.doc.maxId + 1, RIGID_TEXTURE_WIDTH * SLOTS_PER_TEXEL) *
        RIGID_TEXTURE_WIDTH *
        SLOTS_PER_TEXEL,
    );
    this.#poses = new Float32Array(RIGID_TEXTURE_WIDTH * FLOATS_PER_TEXEL);
  }

  /** The slot instance id `id`'s splats read (0: none). */
  slotOf(id: number): number {
    return this.#slots[id] ?? 0;
  }

  /** The pose texels as they would be uploaded, for tests. */
  get poseData(): Float32Array {
    return this.#poses;
  }

  /** Instances driven now. */
  get driven(): number[] {
    return [...this.#slotOf.keys()];
  }

  get moving(): boolean {
    return this.#motions.size > 0;
  }

  /**
   * Every driven instance's motion now (scan frame), by instance id: what a dedicated splat
   * renderer applies itself (`scanView/scanMotion.ts`). Everything below a driven instance
   * moves with it.
   */
  get instanceMotions(): ReadonlyMap<number, RigidMotion> {
    return this.#byInstance;
  }

  /** Bumped whenever a motion is set or cleared. */
  get motionVersion(): number {
    return this.#motionVersion;
  }

  /** Whether the shader would act this frame. */
  get active(): boolean {
    return (
      this.enabled &&
      this.#motions.size > 0 &&
      this.#ids.idsCurrent &&
      this.#bake !== undefined &&
      this.#poseTexture !== undefined &&
      this.#slotTexture !== undefined
    );
  }

  addToShader(
    shaderBuilder: SplatShaderBuilder,
    uniformMap: Record<string, () => unknown>,
    context: unknown,
  ): void {
    this.#context = context;
    this.#ensureTextures();
    const vertex = this.#factory.vertexDestination;
    shaderBuilder.addUniform("highp usampler2D", "u_rigidIds", vertex);
    shaderBuilder.addUniform("highp usampler2D", "u_rigidSlots", vertex);
    shaderBuilder.addUniform("highp sampler2D", "u_rigidPoses", vertex);
    shaderBuilder.addUniform("float", "u_rigidActive", vertex);
    shaderBuilder.addUniform("float", "u_rigidMaxId", vertex);
    shaderBuilder.addUniform("float", "u_rigidCovered", vertex);
    shaderBuilder.addVertexLines(rigidGlsl());
    uniformMap.u_rigidIds = () => this.#ids.idTexture ?? this.#emptyIds;
    uniformMap.u_rigidSlots = () => this.#slotTexture;
    uniformMap.u_rigidPoses = () => this.#poseTexture;
    uniformMap.u_rigidActive = () => (this.active && this.#ids.idTexture !== undefined ? 1 : 0);
    uniformMap.u_rigidMaxId = () => this.#ids.doc.maxId;
    uniformMap.u_rigidCovered = () => this.#ids.covered;
  }

  install(primitive: SplatPrimitive): boolean {
    if (this.#primitive && this.#primitive !== primitive) this.uninstall();
    this.#primitive = primitive;
    return hasMotionPart(primitive, this) || addMotionPart(primitive, this);
  }

  get installed(): boolean {
    return hasMotionPart(this.#primitive, this);
  }

  uninstall(): void {
    const primitive = this.#primitive;
    this.#primitive = undefined;
    if (primitive) removeMotionPart(primitive, this);
  }

  /**
   * Moves instance `id` (and everything below it) by `motion` (`x' = R x + t`, scan frame), or
   * puts it back at rest with `null`.
   */
  setInstanceMotion(id: number, motion: RigidMotion | null): void {
    if (id < 1 || id > this.#ids.doc.maxId) return;
    let slot = this.#slotOf.get(id);
    if (motion === null) {
      if (slot === undefined) return;
      this.#byInstance.delete(id);
      this.#motionVersion += 1;
      this.#motions.delete(slot);
      this.#slotOf.delete(id);
      this.#writePose(slot, RIGID_IDENTITY);
      this.#rewriteSlots();
      return;
    }
    if (slot === undefined) {
      slot = 1;
      const used = new Set(this.#slotOf.values());
      while (used.has(slot)) slot += 1;
      this.#slotOf.set(id, slot);
      this.#reservePoses(slot);
      this.#rewriteSlots();
    }
    this.#motions.set(slot, motion);
    this.#byInstance.set(id, motion);
    this.#motionVersion += 1;
    this.#writePose(slot, motion);
  }

  /** Every instance back at rest. */
  rest(): void {
    for (const id of [...this.#slotOf.keys()]) this.setInstanceMotion(id, null);
  }

  /**
   * Installs on the drawn primitive, refolds the poses when the bake changes, and uploads what
   * changed. Call once a frame, before the render; true when something was uploaded.
   */
  sync(): boolean {
    const primitive = this.#tileset.gaussianSplatPrimitive;
    if (primitive && (primitive !== this.#primitive || !this.installed)) this.install(primitive);
    const bake = this.#ids.bake;
    if (bake !== undefined && (this.#bake === undefined || !sameMatrix(this.#bake, bake))) {
      const inverse = invertAffine(bake);
      if (inverse !== undefined) {
        this.#bake = Array.from(bake);
        this.#inverse = inverse;
        for (const [slot, motion] of this.#motions) this.#writePose(slot, motion);
      }
    }
    this.#syncSort(primitive);
    return this.#ensureTextures();
  }

  destroy(): void {
    if (this.#sorted) setSortMotion(this.#sorted, undefined);
    this.#sorted = undefined;
    this.uninstall();
    for (const texture of [this.#slotTexture, this.#poseTexture, this.#emptyIds]) {
      if (texture && !texture.isDestroyed()) texture.destroy();
    }
    this.#slotTexture = undefined;
    this.#poseTexture = undefined;
    this.#emptyIds = undefined;
  }

  #rewriteSlots(): void {
    this.#slots.fill(0);
    // Shallow instances first, so a driven instance below another keeps its own slot.
    const levels = new Map(this.#ids.doc.instances.map((i) => [i.id, i.level] as const));
    const order = [...this.#slotOf].sort(([a], [b]) => (levels.get(a) ?? 0) - (levels.get(b) ?? 0));
    for (const [id, slot] of order) {
      for (const leaf of withDescendants(this.#ids.doc, [id])) {
        if (leaf >= 1 && leaf < this.#slots.length) this.#slots[leaf] = slot;
      }
    }
    this.#slotsDirty = true;
    this.#slotsVersion += 1;
  }

  #reservePoses(slot: number): void {
    const needed =
      rowsFor((slot + 1) * TEXELS_PER_SLOT, RIGID_TEXTURE_WIDTH) *
      RIGID_TEXTURE_WIDTH *
      FLOATS_PER_TEXEL;
    if (needed <= this.#poses.length) return;
    const grown = new Float32Array(needed);
    grown.set(this.#poses);
    this.#poses = grown;
  }

  #writePose(slot: number, motion: RigidMotion): void {
    const at = slot * TEXELS_PER_SLOT * FLOATS_PER_TEXEL;
    this.#poses.fill(0, at, at + TEXELS_PER_SLOT * FLOATS_PER_TEXEL);
    const bake = this.#bake;
    const inverse = this.#inverse;
    if (bake !== undefined && inverse !== undefined && this.#motions.has(slot)) {
      const z = rigidHandle(motion.rotation, motion.translation, new Float64Array(HANDLE_FLOATS));
      foldHandle(z, 0, [0, 0, 0], bake, inverse, this.#poses, at);
    }
    this.#posesDirty = true;
    this.#sortDirty = true;
  }

  /** Tells the sorter which splats move with which slot, and how. */
  #syncSort(primitive: SplatPrimitive | undefined): void {
    if (this.#sorted && this.#sorted !== primitive) {
      setSortMotion(this.#sorted, undefined);
      this.#sorted = undefined;
    }
    if (!primitive) return;
    if (this.#motions.size === 0 || !this.enabled) {
      if (this.#sorted) setSortMotion(primitive, undefined);
      this.#sorted = undefined;
      this.#groups = undefined;
      this.#groupsKey = "";
      return;
    }
    const key = `${String(this.#ids.idsVersion)}:${String(this.#slotsVersion)}`;
    if (key !== this.#groupsKey || !this.#groups) {
      const ids = this.#ids.ids;
      const count = Math.min(this.#ids.covered, ids.length);
      const groups = new Uint16Array(count);
      for (let i = 0; i < count; i += 1) groups[i] = this.#slots[ids[i] ?? 0] ?? 0;
      this.#groups = groups;
      this.#groupsKey = key;
      this.#sortDirty = true;
    }
    if (!this.#sortDirty && this.#sorted === primitive) return;
    const slots = this.#poses.length / (TEXELS_PER_SLOT * FLOATS_PER_TEXEL);
    const motions = new Float64Array(slots * 12);
    for (const slot of this.#motions.keys()) {
      for (let r = 0; r < 3; r += 1) {
        const o = (slot * TEXELS_PER_SLOT + r) * FLOATS_PER_TEXEL;
        for (let c = 0; c < 4; c += 1)
          motions[slot * 12 + r * 4 + c] = (this.#poses[o + c] ?? 0) + (c === r ? 1 : 0);
      }
    }
    setSortMotion(primitive, { groups: this.#groups, motions });
    this.#sorted = primitive;
    this.#sortDirty = false;
  }

  #ensureTextures(): boolean {
    const context = this.#context;
    if (context === undefined) return false;
    let uploaded = false;
    this.#emptyIds ??= this.#factory.createUintQuads(context, 1, 1, new Uint32Array(4));
    if (this.#slotTexture === undefined || this.#slotTexture.isDestroyed() || this.#slotsDirty) {
      // A few KB, rewritten only when what is driven changes.
      this.#slotTexture?.destroy();
      this.#slotTexture = this.#factory.createUintQuads(
        context,
        RIGID_TEXTURE_WIDTH,
        this.#slots.length / (RIGID_TEXTURE_WIDTH * SLOTS_PER_TEXEL),
        this.#slots,
      );
      this.#slotsDirty = false;
      uploaded = true;
    }
    const rows = this.#poses.length / (RIGID_TEXTURE_WIDTH * FLOATS_PER_TEXEL);
    if (
      this.#poseTexture === undefined ||
      this.#poseTexture.isDestroyed() ||
      this.#poseRows !== rows
    ) {
      this.#poseTexture?.destroy();
      this.#poseTexture = this.#factory.createFloat(
        context,
        RIGID_TEXTURE_WIDTH,
        rows,
        this.#poses,
      );
      this.#poseRows = rows;
      this.#posesDirty = false;
      uploaded = true;
    } else if (this.#posesDirty) {
      this.#poseTexture.copyFrom({
        source: { width: RIGID_TEXTURE_WIDTH, height: rows, arrayBufferView: this.#poses },
        xOffset: 0,
        yOffset: 0,
      });
      this.#posesDirty = false;
      uploaded = true;
    }
    return uploaded;
  }
}
