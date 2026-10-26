import * as pc from "playcanvas";

import { useLivingLab } from "@/state/livingLab";
import { useAsCaptured } from "@/state/asCaptured";
import { effectiveLightArm, lightSun, useLivingLight } from "@/state/livingLight";

import { lightCopyGlsl, LIGHT_SOURCES, LIGHT_STREAMS, lightStreamData } from "./lightHook";

interface Manifest {
  version: number;
  bytesPerSplat: number;
  tiles: Record<string, { uri: string; count: number }>;
}
interface Binding {
  entity: pc.Entity;
  resource: pc.GSplatResource;
  order: Uint32Array;
  checksum: string;
}
/** Local research only. Exact checksums fail closed; async loads are cancelled on disposal. */
export class LightRuntime {
  stream = false;
  readonly bound = new WeakSet<pc.Entity>();
  readonly bindings = new Set<Binding>();
  readonly abort = new AbortController();
  readonly modifiers = new Map<string, { glsl: string }>();
  readonly off: (() => void)[];
  readonly started = performance.now();
  readonly sun = new Float32Array(4);
  readonly params = new Float32Array(4);
  cloud: pc.Texture | null = null;
  manifest: Promise<{ doc: Manifest; url: string }> | null = null;
  constructor(
    readonly app: pc.AppBase,
    readonly changed: (entity?: pc.Entity) => void,
  ) {
    const refresh = () => {
      if (effectiveLightArm() !== "off")
        for (const binding of this.bindings) void this.load(binding);
      changed();
    };
    this.off = [
      useLivingLight.subscribe(refresh),
      useAsCaptured.subscribe(refresh),
      useLivingLab.subscribe(refresh),
    ];
  }
  readonly pending = new WeakSet<pc.Entity>();
  register(binding: Binding): void {
    this.bindings.add(binding);
    if (effectiveLightArm() !== "off") void this.load(binding);
  }
  remove(entity: pc.Entity): void {
    for (const binding of this.bindings)
      if (binding.entity === entity) this.bindings.delete(binding);
  }
  async load(binding: Binding): Promise<void> {
    if (this.pending.has(binding.entity) || this.bound.has(binding.entity)) return;
    const url = useLivingLight.getState().manifest;
    if (!url) return;
    this.pending.add(binding.entity);
    try {
      this.manifest ??= fetch(url, { signal: this.abort.signal }).then(async (response) => {
        if (!response.ok) throw new Error(`Lighting manifest HTTP ${String(response.status)}`);
        const doc = (await response.json()) as Manifest;
        if (doc.version !== 1 || doc.bytesPerSplat !== 48 || !doc.tiles)
          throw new Error("Invalid lighting manifest");
        return { doc, url: response.url };
      });
      const loaded = await this.manifest;
      const entry = loaded.doc.tiles[binding.checksum];
      if (!entry || entry.count !== binding.order.length) return;
      const response = await fetch(new URL(entry.uri, loaded.url), { signal: this.abort.signal });
      if (!response.ok) throw new Error(`Lighting sidecar HTTP ${String(response.status)}`);
      const streams = lightStreamData(await response.arrayBuffer(), binding.order);
      if (this.abort.signal.aborted || !this.bindings.has(binding)) return;
      if (!this.stream) {
        this.app.scene.gsplat.format.addExtraStreams(
          LIGHT_STREAMS.map((name) => ({ name, format: pc.PIXELFORMAT_RGBA32U })),
        );
        this.stream = true;
        for (const item of this.bindings) this.changed(item.entity);
      }
      binding.resource.format.addExtraStreams(
        LIGHT_SOURCES.map((name) => ({ name, format: pc.PIXELFORMAT_RGBA32U })),
      );
      LIGHT_SOURCES.forEach((name, i) => {
        const texture = binding.resource.getTexture(name);
        const data = streams[i];
        if (texture && data) {
          (texture.lock() as Uint32Array).set(data);
          texture.unlock();
        }
      });
      this.bound.add(binding.entity);
      this.changed(binding.entity);
      useLivingLight.getState().set({ status: "Simulated · baked lighting loaded" });
    } catch (error) {
      if (!this.abort.signal.aborted) useLivingLight.getState().set({ status: String(error) });
    }
  }
  modifier(base: { glsl?: string } | null, entity: pc.Entity): { glsl: string } | null {
    if (!this.stream) return base as { glsl: string } | null;
    const key = `${String(this.bound.has(entity))}|${base?.glsl ?? ""}`;
    let value = this.modifiers.get(key);
    if (!value) {
      value = { glsl: lightCopyGlsl(base?.glsl, this.bound.has(entity)) };
      this.modifiers.set(key, value);
    }
    return value;
  }
  update(material: pc.ShaderMaterial): void {
    const state = useLivingLight.getState();
    const dir = lightSun(state.day);
    this.sun.set([...dir, state.diffuse ? 0 : Math.max(0, dir[2])]);
    this.params.set([
      Math.min(1, Math.max(0, state.day)) * 31,
      state.arm === "b" ? 1 : 0,
      state.cloud,
      state.seconds ?? (performance.now() - this.started) / 1000,
    ]);
    if (!this.cloud) {
      this.cloud = new pc.Texture(this.app.graphicsDevice, {
        name: "livingLightCloud",
        width: 64,
        height: 64,
        format: pc.PIXELFORMAT_RGBA8,
        mipmaps: false,
        minFilter: pc.FILTER_LINEAR,
        magFilter: pc.FILTER_LINEAR,
        addressU: pc.ADDRESS_REPEAT,
        addressV: pc.ADDRESS_REPEAT,
      });
      const pixels = this.cloud.lock() as Uint8Array;
      // Periodic low-frequency noise texture, with deterministic phases (no seams).
      for (let y = 0; y < 64; y++)
        for (let x = 0; x < 64; x++) {
          const u = (x / 64) * 2 * Math.PI,
            v = (y / 64) * 2 * Math.PI;
          const value =
            0.5 +
            0.19 * Math.sin(u + 0.7) * Math.cos(v) +
            0.13 * Math.sin(2 * u - v + 1.3) +
            0.08 * Math.cos(u + 3 * v);
          const index = (y * 64 + x) * 4;
          pixels.fill(Math.round(Math.min(1, Math.max(0, value)) * 255), index, index + 3);
          pixels[index + 3] = 255;
        }
      this.cloud.unlock();
    }
    material.setParameter("uLightSun", this.sun);
    material.setParameter("uLightParams", this.params);
    material.setParameter("uLightCloud", this.cloud);
  }
  destroy(): void {
    this.abort.abort();
    this.off.forEach((off) => off());
    this.cloud?.destroy();
    this.bindings.clear();
  }
}
