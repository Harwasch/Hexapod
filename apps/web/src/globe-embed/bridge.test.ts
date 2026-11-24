import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { resolve } from "node:path";
import { describe, expect, it, vi } from "vitest";
import { Event as CesiumEvent, type Cesium3DTileset } from "cesium";
import { decodeMessage, dispatchMessage } from "./bridge";
import { ScanLayer } from "./ScanLayer";
import { EMBED_TILE_OPTIONS } from "@/cesium/providers/tiles";
import fixtures from "../../../ios/Modules/Globe/Tests/GlobeTests/Fixtures/messages.json";

const root = resolve(__dirname, "../../../..");
describe("shared native bridge", () => {
  it("round-trips every shared wire fixture and rejects unknown versions", () => {
    for (const fixture of fixtures)
      expect(JSON.parse(JSON.stringify(decodeMessage(fixture)))).toEqual(fixture);
    expect(() => decodeMessage({ ...fixtures[0], version: 2 })).toThrow();
    expect(() => decodeMessage({ version: 1, type: "selectScan" })).toThrow();
    expect(() => decodeMessage({ version: 1, type: "mystery" })).toThrow();
    const bad = structuredClone(fixtures[0]);
    bad!.snapshot!.pins[0]!.geo.lat = 91;
    expect(() => decodeMessage(bad)).toThrow();
  });
  it("dispatches commands only, keeping native events out of JS actions", () => {
    const commands = { snapshot: vi.fn(), selectScan: vi.fn(), flyToScan: vi.fn() };
    for (const fixture of fixtures.slice(0, 3)) dispatchMessage(fixture, commands);
    expect(commands.snapshot).toHaveBeenCalledWith(fixtures[0]!.snapshot);
    expect(commands.selectScan).toHaveBeenCalledWith(fixtures[1]!.scanId);
    expect(commands.flyToScan).toHaveBeenCalledWith(fixtures[2]!.scanId);
    expect(() => dispatchMessage(fixtures[4], commands)).toThrow();
  });
  it("keeps both generated languages in sync with the one schema", () => {
    const result = spawnSync(
      process.execPath,
      [resolve(root, "apps/web/scripts/generate-globe-bridge.mjs"), "--check"],
      { encoding: "utf8" },
    );
    expect(result.status, result.stderr).toBe(0);
  });
  it("uses existing KHR Gaussian tiles and an explicit mobile cache", () => {
    const tile = JSON.parse(
      readFileSync(resolve(root, "apps/web/public/fixtures/globe-splat/tileset.json"), "utf8"),
    ) as { extensions: { "3DTILES_content_gltf": { extensionsRequired: string[] } } };
    expect(tile.extensions["3DTILES_content_gltf"].extensionsRequired).toContain(
      "KHR_gaussian_splatting",
    );
    expect(EMBED_TILE_OPTIONS).toMatchObject({
      maximumScreenSpaceError: 8,
      cacheBytes: 64 * 1024 * 1024,
      maximumCacheOverflowBytes: 16 * 1024 * 1024,
      preloadWhenHidden: false,
    });
  });
});
it("destroys stale loads, replaces the visible scan, and ignores late failures", async () => {
  const tile = (): Cesium3DTileset =>
    ({ destroy: vi.fn(), tileFailed: new CesiumEvent() }) as unknown as Cesium3DTileset;
  let resolveFirst!: (tile: Cesium3DTileset) => void;
  const first = new Promise<Cesium3DTileset>((resolve) => {
    resolveFirst = resolve;
  });
  const load = vi.fn().mockReturnValueOnce(first).mockResolvedValueOnce(tile());
  const add = vi.fn(),
    remove = vi.fn(),
    changed = vi.fn(),
    failed = vi.fn();
  const layer = new ScanLayer(load, add, remove, changed, failed);
  const pending = layer.select("a");
  await layer.select("b");
  const stale = tile();
  const destroyed = vi.spyOn(stale, "destroy");
  resolveFirst(stale);
  await pending;
  expect(destroyed).toHaveBeenCalledOnce();
  expect(add).toHaveBeenCalledOnce();
  layer.dispose();
  expect(remove).toHaveBeenCalledOnce();
  expect(failed).not.toHaveBeenCalled();
});

it("forwards failures of loaded tiles and detaches retired tiles", async () => {
  const tile = { tileFailed: new CesiumEvent(), destroy: vi.fn() };
  const failed = vi.fn();
  const layer = new ScanLayer(
    () => Promise.resolve(tile as unknown as Cesium3DTileset),
    vi.fn(),
    vi.fn(),
    vi.fn(),
    failed,
  );
  await layer.select("a");
  tile.tileFailed.raiseEvent({ message: "blocked" });
  expect(failed).toHaveBeenCalledOnce();
  layer.dispose();
  tile.tileFailed.raiseEvent({ message: "late" });
  expect(failed).toHaveBeenCalledOnce();
  expect(tile.tileFailed.numberOfListeners).toBe(0);
});
