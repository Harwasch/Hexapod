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
 * The scan's objects (scanInstances.ts): each tile's SPZ centres are digested as it loads
 * (`spzPositions`, the checksum `instances.json` keys its ids by), and once the scan has
 * instances every tile gets an object modifier (a Spark dyno) that reads its splat's id from a
 * per-tile texture and the shared state table, and applies the same rule as CesiumJS and
 * PlayCanvas. A change of what is hidden or highlighted rewrites the table and has Spark
 * regenerate the tiles' splats.
 */

import { dyno, SparkRenderer, type SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";

import { checksumPositions } from "@twin/world";

import { tileInstanceIds, type InstancesDoc } from "@/lib/instances";
import { spzPositions } from "@/lib/spzPositions";
import { loadSplatTile } from "@/view/sparkStream";

import { INSTANCE_TEXTURE_WIDTH } from "../splatInstances";
import type { InstanceStyle } from "./scanInstances";
import type { ScanBackend, ScanPose } from "./types";

/** Splats a row of a tile's id texture holds (one a texel, R32UI). */
const SPARK_IDS_WIDTH = 4096;

/**
 * The rule as Spark's dyno globals: the shared GLSL takes its uniforms by these names, so they
 * are the parameters of the one function the modifier calls.
 */
const SPARK_INSTANCE_GLSL = `
vec4 hexapodSparkInstanceColor(
    int index, vec4 color, usampler2D ids, sampler2D uInstanceState,
    vec4 uInstanceParams, vec4 uInstanceTint, vec4 uInstanceDim) {
    if (uInstanceParams.x < 0.5) {
        return color;
    }
    uint id = texelFetch(ids, ivec2(index % ${String(SPARK_IDS_WIDTH)}, index / ${String(SPARK_IDS_WIDTH)}), 0).r;
    vec4 state = vec4(0.0);
    if (id != 0u && float(id) <= uInstanceParams.y) {
        int i = int(id);
        state = texelFetch(uInstanceState, ivec2(i & ${String(INSTANCE_TEXTURE_WIDTH - 1)}, i >> ${String(Math.log2(INSTANCE_TEXTURE_WIDTH))}), 0);
    }
    if (state.r > 0.5) {
        return vec4(color.rgb, 0.0);
    }
    if (uInstanceParams.z < 0.5) {
        return color;
    }
    if (state.g > 0.5) {
        return vec4(mix(color.rgb, uInstanceTint.rgb, uInstanceTint.a) + 0.06, color.a);
    }
    return vec4(color.rgb * uInstanceDim.x, color.a * uInstanceDim.y);
}
`;

/** One tile's binding to the scan's objects. */
interface SparkTile {
  checksum: string;
  count: number;
  doc: InstancesDoc | null;
  matched: boolean;
  ids: THREE.DataTexture | null;
}

export function createBackend(canvas: HTMLCanvasElement): Promise<ScanBackend<SplatMesh>> {
  const renderer = new THREE.WebGLRenderer({
    canvas,
    alpha: true,
    antialias: false,
    premultipliedAlpha: true,
    powerPreference: "high-performance",
  });
  renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  const spark = new SparkRenderer({ renderer, enableLod: false });
  scene.add(spark);
  const camera = new THREE.PerspectiveCamera();
  const target = new THREE.Vector3();
  const direction = new THREE.Vector3();
  let size = { width: 0, height: 0, pixelRatio: 0 };

  // The scan's objects: the shared state table and the rule's numbers, as dyno uniforms.
  const tiles = new Map<SplatMesh, SparkTile>();
  let style: InstanceStyle | null = null;
  let stateTexture: THREE.DataTexture | null = null;
  const state = dyno.dynoSampler2D(new THREE.DataTexture());
  const params = dyno.dynoVec4(new THREE.Vector4());
  const tint = dyno.dynoVec4(new THREE.Vector4());
  const dim = dyno.dynoVec4(new THREE.Vector4(1, 1, 0, 0));
  const modifierFor = (
    ids: THREE.DataTexture,
  ): dyno.Dyno<{ gsplat: typeof dyno.Gsplat }, { gsplat: typeof dyno.Gsplat }> => {
    const idsUniform = dyno.dynoUsampler2D(ids);
    return dyno.dynoBlock({ gsplat: dyno.Gsplat }, { gsplat: dyno.Gsplat }, ({ gsplat }) => {
      if (!gsplat) throw new Error("No gsplat input");
      const rule = new dyno.Dyno({
        inTypes: {
          gsplat: dyno.Gsplat,
          ids: "usampler2D",
          state: "sampler2D",
          params: "vec4",
          tint: "vec4",
          dim: "vec4",
        },
        outTypes: { gsplat: dyno.Gsplat },
        globals: () => [SPARK_INSTANCE_GLSL],
        statements: ({ inputs, outputs }) => [
          `${String(outputs.gsplat)} = ${String(inputs.gsplat)};`,
          `${String(outputs.gsplat)}.rgba = hexapodSparkInstanceColor(${String(inputs.gsplat)}.index, ${String(inputs.gsplat)}.rgba, ${String(inputs.ids)}, ${String(inputs.state)}, ${String(inputs.params)}, ${String(inputs.tint)}, ${String(inputs.dim)});`,
        ],
      });
      return {
        gsplat: rule.apply({ gsplat, ids: idsUniform, state, params, tint, dim }).gsplat,
      };
    });
  };
  const bind = (mesh: SplatMesh, tile: SparkTile): void => {
    if (style && tile.doc !== style.doc) {
      const found = tileInstanceIds(style.doc, tile.checksum);
      const listed = found?.length === tile.count ? found : undefined;
      const rows = Math.max(1, Math.ceil(tile.count / SPARK_IDS_WIDTH));
      const data = new Uint32Array(SPARK_IDS_WIDTH * rows);
      if (listed) data.set(listed);
      tile.ids?.dispose();
      const texture = new THREE.DataTexture(
        data,
        SPARK_IDS_WIDTH,
        rows,
        THREE.RedIntegerFormat,
        THREE.UnsignedIntType,
      );
      texture.internalFormat = "R32UI";
      texture.needsUpdate = true;
      tile.ids = texture;
      tile.doc = style.doc;
      tile.matched = listed !== undefined;
      mesh.objectModifier = modifierFor(texture);
      mesh.updateGenerator();
    }
    if (tile.doc !== null) mesh.updateVersion();
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

  const backend: ScanBackend<SplatMesh> = {
    name: "spark",
    loadFactor: 1,
    load: async (tilesetUrl, tile, signal) => {
      const { mesh, bytes } = await loadSplatTile(tilesetUrl, tile, {
        lod: false,
        extSplats: true,
        signal,
      });
      const positions = await spzPositions(bytes).catch(() => undefined);
      if (positions) {
        const binding: SparkTile = {
          checksum: checksumPositions(positions),
          count: positions.length / 3,
          doc: null,
          matched: false,
          ids: null,
        };
        tiles.set(mesh, binding);
        if (style) bind(mesh, binding);
      }
      return mesh;
    },
    add: (mesh) => scene.add(mesh),
    remove: (mesh) => scene.remove(mesh),
    dispose: (mesh) => {
      tiles.get(mesh)?.ids?.dispose();
      tiles.delete(mesh);
      mesh.dispose();
    },
    setInstances,
    instanceTiles: () => {
      let matched = 0;
      for (const tile of tiles.values()) if (tile.matched) matched += 1;
      return { tiles: tiles.size, matched };
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
      for (const tile of tiles.values()) tile.ids?.dispose();
      tiles.clear();
      stateTexture?.dispose();
      renderer.setAnimationLoop(null);
      renderer.dispose();
    },
  };
  return Promise.resolve(backend);
}
