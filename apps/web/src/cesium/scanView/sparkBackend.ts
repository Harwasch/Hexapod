/**
 * The Spark back-end of the scan renderer (ScanRendererHost): three.js with Spark's splat
 * renderer drawing the tiles the streamer chose, as they are.
 *
 * Two things Spark does by default are off here:
 *
 * - **Its own level of detail.** `lod: true` had Spark build a merged tree over every tile and
 *   draw at most `lodSplatCount` splats from all of them, which put a second, coarser level of
 *   detail on top of the tileset's own: blocks of merged, grid-aligned splats where the
 *   tileset had sent the real ones, and a tree to build (one tile at a time, in Spark's
 *   worker) before a new tile could show. The tileset already is a level-of-detail hierarchy,
 *   and the streamer picks from it within the budget, so Spark draws what it is given.
 * - **16-bit positions.** Spark's packed splats keep centres as half floats, which are 6 cm
 *   apart at 64-128 m from the origin -- and a tile's splats are in the scan's frame, up to a
 *   hundred metres out: a visible lattice. `extSplats` keeps them as 32-bit floats.
 *
 * The scan's objects: each tile's SPZ centres are digested as it loads (`spzPickData`, the
 * checksum `instances.json` and `skin.json` key their per-splat data by), and once the scan
 * has instances or a skin every tile so listed gets a world modifier (a Spark dyno) that reads
 * its splat's instance id, skin id and weight row from per-tile textures and the shared
 * tables, and applies what CesiumJS and PlayCanvas apply: the motion (scanMotion.ts: skins and
 * rigid motions, the covariance through the motion's linear part) and the hide-and-highlight
 * rule (scanInstances.ts). A change has Spark regenerate only the tiles holding what changed.
 * Spark sorts what its generators output, so a moved splat is sorted where it is drawn.
 */

import { dyno, SparkRenderer, type SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";

import { checksumPositions } from "@twin/world";

import { tileInstanceIds, type InstancesDoc } from "@/lib/instances";
import { tileSkin, type SkinDoc } from "@/lib/skin";
import type { PickTile } from "@/lib/splatPick";
import { spzPickData } from "@/lib/spzPositions";
import { loadSplatTile } from "@/view/sparkStream";

import { INSTANCE_TEXTURE_WIDTH } from "../splatInstances";
import type { InstanceStyle } from "./scanInstances";
import { MOTION_TEXTURE_WIDTH, SCAN_MOTION_GLSL, type ScanMotion } from "./scanMotion";
import type { ScanBackend, ScanPose } from "./types";

/** Splats a row of a tile's per-splat textures holds (one a texel). */
const SPARK_SPLATS_WIDTH = 4096;

/**
 * The modifier as one function of Spark's dyno globals: its tables are parameters, so the
 * shared GLSL takes them by name. `hasIds` and `hasSkin` say which per-splat textures are real.
 */
export const SPARK_MOTION_GLSL = `
${SCAN_MOTION_GLSL}
void hexapodSparkModify(
    int index, inout vec3 center, inout vec3 scales, inout vec4 quaternion, inout vec4 rgba,
    bool hasIds, bool hasSkin, highp usampler2D ids, highp usampler2D skins,
    highp usampler2D weights, highp sampler2D uInstanceState, vec4 uInstanceParams,
    vec4 uInstanceTint, vec4 uInstanceDim, highp sampler2D handles, highp usampler2D slots,
    highp sampler2D poses, vec4 motion, vec4 extra) {
    ivec2 at = ivec2(index % ${String(SPARK_SPLATS_WIDTH)}, index / ${String(SPARK_SPLATS_WIDTH)});
    uint id = hasIds ? texelFetch(ids, at, 0).r : 0u;
    if (motion.x > 0.5 || motion.z > 0.5) {
        vec3 delta = vec3(0.0);
        mat3 linear = mat3(0.0);
        if (hasSkin) {
            hexapodSkinMotion(handles, texelFetch(skins, at, 0).r, texelFetch(weights, at, 0),
                              motion, extra.x, center, delta, linear);
        }
        if (hasIds) {
            hexapodRigidMotion(slots, poses, id, motion, center, delta, linear);
        }
        center += delta;
        if (extra.y > 0.5) {
            hexapodCovariance(mat3(1.0) + linear, quaternion, scales);
        }
    }
    if (uInstanceParams.x < 0.5) {
        return;
    }
    vec4 state = vec4(0.0);
    if (id != 0u && float(id) <= uInstanceParams.y) {
        int i = int(id);
        state = texelFetch(uInstanceState, ivec2(i & ${String(INSTANCE_TEXTURE_WIDTH - 1)}, i >> ${String(Math.log2(INSTANCE_TEXTURE_WIDTH))}), 0);
    }
    if (state.r > 0.5) {
        rgba = vec4(rgba.rgb, 0.0);
        return;
    }
    if (uInstanceParams.z < 0.5) {
        return;
    }
    if (state.g > 0.5) {
        rgba = vec4(mix(rgba.rgb, uInstanceTint.rgb, uInstanceTint.a) + 0.06, rgba.a);
        return;
    }
    rgba = vec4(rgba.rgb * uInstanceDim.x, rgba.a * uInstanceDim.y);
}
`;

/** One tile's binding to the scan's objects. */
interface SparkTile {
  checksum: string;
  count: number;
  /** The instances doc its ids were written from, whether it lists the tile, the ids. */
  doc: InstancesDoc | null;
  matched: boolean;
  ids: THREE.DataTexture | null;
  idSet: ReadonlySet<number>;
  /** The skin doc its skin textures were written from, and the skins it holds. */
  skinDoc: SkinDoc | null;
  skins: THREE.DataTexture | null;
  weights: THREE.DataTexture | null;
  skinSet: ReadonlySet<number>;
  /** Whether it has a modifier (built from its textures as they are now). */
  modified: boolean;
  /** The tile's splats for picking, in its own order (cesium/sceneSelect). */
  pick: PickTile;
  /** `pick` where `place` last put the tile (a split object at its pose), or null at rest. */
  placedPick: PickTile | null;
}

/** `pick` moved by `matrix` (a split object's placement in the scan frame). */
function placedPickTile(pick: PickTile, matrix: THREE.Matrix4): PickTile {
  const positions = new Float32Array(pick.positions.length);
  const point = new THREE.Vector3();
  for (let i = 0; i < pick.count; i++) {
    point.fromArray(pick.positions, i * 3).applyMatrix4(matrix);
    point.toArray(positions, i * 3);
  }
  return { ...pick, positions };
}

/** A per-tile texture: one splat a texel, `SPARK_SPLATS_WIDTH` a row. */
function splatTexture(count: number, channels: 1 | 4, data?: Uint32Array): THREE.DataTexture {
  const rows = Math.max(1, Math.ceil(count / SPARK_SPLATS_WIDTH));
  const words = new Uint32Array(SPARK_SPLATS_WIDTH * rows * channels);
  if (data) words.set(data.subarray(0, Math.min(data.length, words.length)));
  const texture = new THREE.DataTexture(
    words,
    SPARK_SPLATS_WIDTH,
    rows,
    channels === 1 ? THREE.RedIntegerFormat : THREE.RGBAIntegerFormat,
    THREE.UnsignedIntType,
  );
  texture.internalFormat = channels === 1 ? "R32UI" : "RGBA32UI";
  texture.needsUpdate = true;
  return texture;
}

/** A shared table: `MOTION_TEXTURE_WIDTH` texels a row, RGBA32F or RGBA32UI. */
function tableTexture(data: Float32Array | Uint32Array, rows: number): THREE.DataTexture {
  const float = data instanceof Float32Array;
  const texture = new THREE.DataTexture(
    data.slice(),
    MOTION_TEXTURE_WIDTH,
    rows,
    float ? THREE.RGBAFormat : THREE.RGBAIntegerFormat,
    float ? THREE.FloatType : THREE.UnsignedIntType,
  );
  texture.internalFormat = float ? "RGBA32F" : "RGBA32UI";
  texture.needsUpdate = true;
  return texture;
}

export interface BackendOptions {
  /** Keeps the drawn frame readable after it is shown (harnesses read pixels back). */
  preserveDrawingBuffer?: boolean;
}

export function createBackend(
  canvas: HTMLCanvasElement,
  _budget?: number,
  options: BackendOptions = {},
): Promise<ScanBackend<SplatMesh>> {
  const renderer = new THREE.WebGLRenderer({
    canvas,
    alpha: true,
    antialias: false,
    premultipliedAlpha: true,
    powerPreference: "high-performance",
    preserveDrawingBuffer: options.preserveDrawingBuffer === true,
  });
  renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  const spark = new SparkRenderer({ renderer, enableLod: false });
  scene.add(spark);
  const camera = new THREE.PerspectiveCamera();
  const target = new THREE.Vector3();
  const direction = new THREE.Vector3();
  const placement = new THREE.Matrix4();
  const placedScale = new THREE.Vector3();
  let size = { width: 0, height: 0, pixelRatio: 0 };

  // The shared tables and numbers, as dyno uniforms every tile's modifier reads.
  const tiles = new Map<SplatMesh, SparkTile>();
  /** Tiles on screen now (added, not removed). */
  const shown = new Set<SplatMesh>();
  let style: InstanceStyle | null = null;
  let motion: ScanMotion | null = null;
  let stateTexture: THREE.DataTexture | null = null;
  const emptyUint = splatTexture(1, 4);
  const emptyFloat = tableTexture(new Float32Array(MOTION_TEXTURE_WIDTH * 4), 1);
  const emptySplats = splatTexture(1, 1);
  const state = dyno.dynoSampler2D(new THREE.DataTexture(new Uint8Array(4), 1, 1));
  const params = dyno.dynoVec4(new THREE.Vector4());
  const tint = dyno.dynoVec4(new THREE.Vector4());
  const dim = dyno.dynoVec4(new THREE.Vector4(1, 1, 0, 0));
  const handles = dyno.dynoSampler2D(emptyFloat);
  const slots = dyno.dynoUsampler2D(emptyUint);
  const poses = dyno.dynoSampler2D(emptyFloat);
  const motionParams = dyno.dynoVec4(new THREE.Vector4());
  const motionExtra = dyno.dynoVec4(new THREE.Vector4(0, 1, 0, 0));
  let redrawn = 0;

  const modifierFor = (
    tile: SparkTile,
  ): dyno.Dyno<{ gsplat: typeof dyno.Gsplat }, { gsplat: typeof dyno.Gsplat }> => {
    const ids = dyno.dynoUsampler2D(tile.ids ?? emptySplats);
    const skins = dyno.dynoUsampler2D(tile.skins ?? emptySplats);
    const weights = dyno.dynoUsampler2D(tile.weights ?? emptyUint);
    const hasIds = tile.ids !== null;
    const hasSkin = tile.skins !== null;
    return dyno.dynoBlock({ gsplat: dyno.Gsplat }, { gsplat: dyno.Gsplat }, ({ gsplat }) => {
      if (!gsplat) throw new Error("No gsplat input");
      const rule = new dyno.Dyno({
        inTypes: {
          gsplat: dyno.Gsplat,
          ids: "usampler2D",
          skins: "usampler2D",
          weights: "usampler2D",
          state: "sampler2D",
          params: "vec4",
          tint: "vec4",
          dim: "vec4",
          handles: "sampler2D",
          slots: "usampler2D",
          poses: "sampler2D",
          motion: "vec4",
          extra: "vec4",
        },
        outTypes: { gsplat: dyno.Gsplat },
        globals: () => [SPARK_MOTION_GLSL],
        statements: ({ inputs, outputs }) => {
          const out = String(outputs.gsplat);
          const i = (name: keyof typeof inputs): string => String(inputs[name]);
          return [
            `${out} = ${i("gsplat")};`,
            `hexapodSparkModify(${i("gsplat")}.index, ${out}.center, ${out}.scales, ${out}.quaternion, ${out}.rgba, ${String(hasIds)}, ${String(hasSkin)}, ${i("ids")}, ${i("skins")}, ${i("weights")}, ${i("state")}, ${i("params")}, ${i("tint")}, ${i("dim")}, ${i("handles")}, ${i("slots")}, ${i("poses")}, ${i("motion")}, ${i("extra")});`,
          ];
        },
      });
      return {
        gsplat: rule.apply({
          gsplat,
          ids,
          skins,
          weights,
          state,
          params,
          tint,
          dim,
          handles,
          slots,
          poses,
          motion: motionParams,
          extra: motionExtra,
        }).gsplat,
      };
    });
  };

  /** The instances whose ids tiles carry: hide and highlight's, or the motion's. */
  const instancesDoc = (): InstancesDoc | null => style?.doc ?? motion?.instances ?? null;

  /**
   * Writes what the tile needs for the current style and motion -- a new per-splat texture
   * builds its modifier again -- and has Spark regenerate it.
   */
  const bind = (mesh: SplatMesh, tile: SparkTile): void => {
    let replaced = false;
    const doc = instancesDoc();
    if (doc && tile.doc !== doc) {
      const found = tileInstanceIds(doc, tile.checksum);
      const listed = found?.length === tile.count ? found : undefined;
      tile.ids?.dispose();
      tile.ids = listed ? splatTexture(tile.count, 1, listed) : null;
      tile.idSet = new Set(listed ?? []);
      tile.doc = doc;
      tile.matched = listed !== undefined;
      replaced = true;
    }
    const skinDoc = motion?.skin ?? null;
    if (skinDoc && tile.skinDoc !== skinDoc) {
      tile.skinDoc = skinDoc;
      const found = tileSkin(skinDoc, tile.checksum);
      const skinSet = new Set<number>();
      const fits = found?.skins.length === tile.count;
      if (fits) for (const id of found.skins) if (id !== 0) skinSet.add(id);
      tile.skins?.dispose();
      tile.weights?.dispose();
      const skinned = fits && skinSet.size > 0;
      tile.skins = skinned ? splatTexture(tile.count, 1, found.skins) : null;
      tile.weights = skinned ? splatTexture(tile.count, 4, found.words) : null;
      tile.skinSet = skinSet;
      replaced = true;
    }
    if (tile.doc === null && tile.skins === null) return;
    if (replaced || !tile.modified) {
      tile.modified = true;
      mesh.worldModifier = modifierFor(tile);
      mesh.updateGenerator();
    }
    mesh.updateVersion();
  };

  const setInstances = (next: InstanceStyle | null): void => {
    style = next;
    if (next) {
      if (stateTexture?.image.height !== next.rows) {
        stateTexture?.dispose();
        stateTexture = new THREE.DataTexture(
          new Uint8Array(next.state.length),
          INSTANCE_TEXTURE_WIDTH,
          next.rows,
          THREE.RGBAFormat,
          THREE.UnsignedByteType,
        );
        state.value = stateTexture;
      }
      (stateTexture.image.data as Uint8Array).set(next.state);
      stateTexture.needsUpdate = true;
      params.value.set(...next.params);
      tint.value.set(...next.tint);
      dim.value.set(...next.dim);
    } else {
      params.value.set(0, 0, 0, 0);
    }
    for (const [mesh, tile] of tiles) bind(mesh, tile);
  };

  /** Puts `data` in the shared table behind `uniform`, in a new texture when it grew. */
  const upload = (
    uniform: { value: THREE.DataTexture },
    data: Float32Array | Uint32Array,
    rows: number,
    empty: THREE.DataTexture,
  ): void => {
    const current = uniform.value;
    if (current !== empty && current.image.height === rows) {
      (current.image.data as typeof data).set(data);
      current.needsUpdate = true;
      return;
    }
    if (current !== empty) current.dispose();
    uniform.value = tableTexture(data, rows);
  };

  const setMotion = (next: ScanMotion | null): void => {
    const before = motion;
    motion = next;
    if (next) {
      upload(handles, next.handles, next.handleRows, emptyFloat);
      upload(slots, next.slots, next.slotRows, emptyUint);
      upload(poses, next.poses, next.poseRows, emptyFloat);
      motionParams.value.set(...next.params);
      motionExtra.value.set(...next.extra);
    } else {
      motionParams.value.set(0, 0, 0, 0);
    }
    const covarianceChanged = (before?.extra[1] ?? 1) !== (next?.extra[1] ?? 1);
    for (const [mesh, tile] of tiles) {
      // Motion switched on or off, or a new document: every tile binds again.
      const rebind =
        next === null ||
        before === null ||
        (next.skin !== null && tile.skinDoc !== next.skin) ||
        (next.instances !== null && tile.doc !== next.instances && style === null);
      const touched =
        rebind ||
        covarianceChanged ||
        [...next.changedSkins].some((id) => tile.skinSet.has(id)) ||
        [...next.changedIds].some((id) => tile.idSet.has(id));
      if (!touched) continue;
      if (rebind) bind(mesh, tile);
      else if (tile.modified) mesh.updateVersion();
      if (tile.modified) redrawn += 1;
    }
  };

  const backend: ScanBackend<SplatMesh> = {
    name: "spark",
    loadFactor: 1,
    load: async (tilesetUrl, tile, signal) => {
      const { mesh, bytes } = await loadSplatTile(tilesetUrl, tile, {
        lod: false,
        extSplats: true,
        signal,
      });
      const data = await spzPickData(bytes).catch(() => undefined);
      if (data) {
        const checksum = checksumPositions(data.positions);
        const count = data.positions.length / 3;
        const binding: SparkTile = {
          checksum,
          count,
          pick: { checksum, count, ...data },
          doc: null,
          matched: false,
          ids: null,
          idSet: new Set(),
          skinDoc: null,
          skins: null,
          weights: null,
          skinSet: new Set(),
          modified: false,
          placedPick: null,
        };
        tiles.set(mesh, binding);
        if (style || motion) bind(mesh, binding);
      }
      return mesh;
    },
    add: (mesh) => {
      scene.add(mesh);
      shown.add(mesh);
    },
    remove: (mesh) => {
      scene.remove(mesh);
      shown.delete(mesh);
    },
    dispose: (mesh) => {
      shown.delete(mesh);
      const tile = tiles.get(mesh);
      tile?.ids?.dispose();
      tile?.skins?.dispose();
      tile?.weights?.dispose();
      tiles.delete(mesh);
      mesh.dispose();
    },
    place: (mesh, matrix) => {
      const tile = tiles.get(mesh);
      if (matrix === null) {
        if (tile) tile.placedPick = null;
        mesh.position.set(0, 0, 0);
        mesh.quaternion.identity();
        return;
      }
      placement.fromArray(Array.from(matrix));
      if (tile) tile.placedPick = placedPickTile(tile.pick, placement);
      placement.decompose(mesh.position, mesh.quaternion, placedScale);
    },
    setInstances,
    pickTiles: () => {
      const out: PickTile[] = [];
      for (const mesh of shown) {
        const tile = tiles.get(mesh);
        if (tile) out.push(tile.placedPick ?? tile.pick);
      }
      return out;
    },
    // Spark generates into `current` and draws `display` until the sort of `current` lands
    // (asynchronously, at most every `minSortIntervalMs`): only then is the frame the state.
    settled: () =>
      !spark.sorting &&
      !spark.sortDirty &&
      spark.sortTimeoutId === -1 &&
      spark.display === spark.current,
    instanceTiles: () => {
      let matched = 0;
      for (const tile of tiles.values()) if (tile.matched) matched += 1;
      return { tiles: tiles.size, matched };
    },
    setMotion,
    motionTiles: () => {
      let skinned = 0;
      for (const tile of tiles.values()) if (tile.skins !== null) skinned += 1;
      return { skinned, redrawn };
    },
    // Spark shows a new mesh once a sort that includes it has run: until then the displayed
    // mapping holds none of its splats.
    isDrawn: (mesh) =>
      spark.display.mapping.some(
        (entry) => (entry.node === mesh || entry.node.parent === mesh) && entry.count > 0,
      ),
    fade: (mesh, alpha) => {
      mesh.opacity = alpha;
    },
    // What is drawn is what the streamer shows; its budget is the draw budget.
    setBudget: () => undefined,
    render: (pose: ScanPose) => {
      if (
        pose.width !== size.width ||
        pose.height !== size.height ||
        pose.pixelRatio !== size.pixelRatio
      ) {
        size = { width: pose.width, height: pose.height, pixelRatio: pose.pixelRatio };
        renderer.setPixelRatio(pose.pixelRatio);
        renderer.setSize(pose.width, pose.height, false);
      }
      camera.position.set(...pose.eye);
      camera.up.set(...pose.up);
      camera.lookAt(target.copy(camera.position).add(direction.set(...pose.direction)));
      camera.fov = THREE.MathUtils.radToDeg(pose.fovy);
      camera.aspect = pose.width / Math.max(1, pose.height);
      camera.near = pose.near;
      camera.far = pose.far;
      camera.updateProjectionMatrix();
      renderer.render(scene, camera);
    },
    destroy: () => {
      for (const tile of tiles.values()) {
        tile.ids?.dispose();
        tile.skins?.dispose();
        tile.weights?.dispose();
      }
      tiles.clear();
      stateTexture?.dispose();
      for (const uniform of [handles, slots, poses]) uniform.value.dispose();
      emptyUint.dispose();
      emptyFloat.dispose();
      emptySplats.dispose();
      renderer.setAnimationLoop(null);
      renderer.dispose();
    },
  };
  return Promise.resolve(backend);
}
