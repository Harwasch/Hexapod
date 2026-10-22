/**
 * Telemetry under As captured (cesium/telemetry.ts `attachTelemetry`, state/asCaptured.ts): the
 * attachment hands the driver the switch each tick, so every bound object is written at rest --
 * its measured frame -- while it is on, through the skin (the yard's shrub, 10) and the rigid part
 * (the building, 8) alike, and moves again from where its readings are now once it is off.
 * The scan's parts are stand-ins (the module's own skin and rigid parts are tested in
 * telemetry.test.ts); the attachment, the driver and the yard's telemetry.json are real.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import type { RigidMotion } from "@twin/world";
import { Event, Matrix4, type Cesium3DTileset } from "cesium";
import { afterEach, describe, expect, it, vi } from "vitest";

import { parseTelemetry } from "@/lib/telemetry";
import { useAsCaptured } from "@/state/asCaptured";

const YARD = resolve(process.cwd(), "../../data/tiles/synthetic-yard");
const read = (relative: string): unknown =>
  JSON.parse(readFileSync(resolve(YARD, relative), "utf8")) as unknown;

const parts = vi.hoisted(() => ({
  skin: new Map<number, Float64Array | null>(),
  rigid: new Map<number, RigidMotion | null>(),
}));

vi.mock("@/cesium/splatInstances", async (original) => {
  const doc = (await import("@/lib/instances")).parseInstances(
    JSON.parse(
      (await import("node:fs")).readFileSync(
        (await import("node:path")).resolve(
          process.cwd(),
          "../../data/tiles/synthetic-yard/instances/instances.json",
        ),
        "utf8",
      ),
    ),
  );
  return {
    ...(await original<Record<string, unknown>>()),
    instancesHookOf: () => ({ doc }),
  };
});

vi.mock("@/cesium/splatSkin", async (original) => {
  const fs = await import("node:fs");
  const path = await import("node:path");
  const dir = path.resolve(process.cwd(), "../../data/tiles/synthetic-yard/skin");
  const bytes = Uint8Array.from(fs.readFileSync(path.resolve(dir, "skin.bin")));
  const doc = (await import("@/lib/skin")).parseSkin(
    JSON.parse(fs.readFileSync(path.resolve(dir, "skin.json"), "utf8")),
    bytes.buffer,
  );
  return {
    ...(await original<Record<string, unknown>>()),
    skinningOf: () => ({
      doc,
      setInstanceHandles: (id: number, handles: ArrayLike<number> | null) => {
        parts.skin.set(id, handles === null ? null : Float64Array.from(handles));
        return true;
      },
    }),
  };
});

vi.mock("@/cesium/splatRigid", async (original) => ({
  ...(await original<Record<string, unknown>>()),
  SplatRigidMotion: class {
    setInstanceMotion(id: number, motion: RigidMotion | null): void {
      parts.rigid.set(id, motion);
    }
    sync(): boolean {
      return false;
    }
    destroy(): void {
      parts.rigid.clear();
    }
  },
}));

const { attachTelemetry } = await import("@/cesium/telemetry");

function yardRoot(): Matrix4 {
  const tileset = read("splat/tileset.json") as { root: { transform: number[] } };
  return Matrix4.fromArray(tileset.root.transform);
}

afterEach(() => {
  useAsCaptured.getState().set(false);
  parts.skin.clear();
  parts.rigid.clear();
});

describe("telemetry while As captured is on", () => {
  it("writes every bound object at rest, and moves it again from now when it is off", async () => {
    const doc = parseTelemetry(read("telemetry/telemetry.json"));
    if (!doc) throw new Error("no telemetry");
    const tileset = {
      root: {
        extras: { telemetry: { uri: "../telemetry/telemetry.json", count: doc.bindings.length } },
        computedTransform: yardRoot(),
      },
      resource: { url: "https://tiles.example/yard/splat/tileset.json" },
    } as unknown as Cesium3DTileset;
    const preUpdate = new Event();
    let now = 0;
    const dispose = attachTelemetry(
      tileset,
      { preUpdate, requestRender: () => undefined },
      "yard",
      { clock: () => now, factory: {} as never, load: () => Promise.resolve(doc) },
    );
    await new Promise((r) => setTimeout(r, 0));
    const tickTo = (until: number): void => {
      for (; now <= until; now += 100) preUpdate.raiseEvent();
    };
    tickTo(4000);
    expect(parts.rigid.get(8)).toBeTruthy();
    expect(parts.skin.get(10)).toBeTruthy();
    useAsCaptured.getState().set(true);
    tickTo(4200);
    expect(parts.rigid.get(8)).toBeNull();
    expect(parts.skin.get(10)).toBeNull();
    // Held for as long as it is on, the sources still read meanwhile.
    tickTo(6000);
    expect(parts.rigid.get(8)).toBeNull();
    expect(parts.skin.get(10)).toBeNull();
    useAsCaptured.getState().set(false);
    tickTo(6200);
    expect(parts.rigid.get(8)).toBeTruthy();
    expect(parts.skin.get(10)).toBeTruthy();
    dispose();
  });
});
