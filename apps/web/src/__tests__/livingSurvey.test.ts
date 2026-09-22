/**
 * `LivingSurveyManager` against a fake scene: the wiring, without a GPU.
 *
 * The deformer's arithmetic is proved in `splatDeformer.test.ts` and the whole path through a
 * real engine is proved in `e2e/livingSurvey.spec.ts`. What is left for here is what the manager
 * alone decides — when a frame is asked for, when the performance ladder is told the scene is
 * animating, what happens when a site leaves, and that `destroy()` puts everything back.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";

import { checksumPositions, WIND_CALM, type WindSettings } from "@twin/world";

import { LIVING_EPOCH_ISO, LivingSurveyManager, sceneSeconds } from "@/cesium/LivingSurveyManager";
import { rigUrlFor } from "@/cesium/livingRigs";
import type { PerformanceManager } from "@/cesium/PerformanceManager";
import type { LoadedSiteAsset, SiteManager } from "@/cesium/SiteManager";
import {
  clearSplatCaptures,
  digestSplatPositions,
  recordSplatCapture,
} from "@/cesium/splatCaptureRegistry";
import type { SceneEvents } from "@/cesium/types";
import { Emitter } from "@/lib/emitter";
import { DEFAULT_WIND_STRENGTH, useLiving } from "@/state/living";

import {
  bakeFixture,
  canonicalPositions,
  FakeSplatPrimitive,
  FakeSplatTileset,
  fixtureRig,
  FIXTURE_SPLATS,
  packedBufferFor,
} from "./splatFixture";

import { JulianDate, type Viewer } from "cesium";

const BREEZE: WindSettings = { strength: 0.6, bearingDeg: 250 };
const ASSET_ID = "asset-tree";
const SITE_ID = "site-tree";
const RIG_URL = "http://localhost/api/v1/tiles/synthetic-tree/source/rig.json";

/** A Cesium `Event` stand-in: records listeners so removal can be asserted. */
class FakeEvent {
  readonly listeners = new Set<() => void>();

  readonly addEventListener = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  raise(): void {
    for (const listener of [...this.listeners]) listener();
  }

  get numberOfListeners(): number {
    return this.listeners.size;
  }
}

function createHarness() {
  const baked = bakeFixture(canonicalPositions);
  const primitive = new FakeSplatPrimitive(baked);
  const tileset = new FakeSplatTileset(primitive);
  recordSplatCapture({
    count: baked.length / 3,
    digest: digestSplatPositions(baked, baked.length / 3),
    data: packedBufferFor(baked),
    at: 0,
  });

  const preUpdate = new FakeEvent();
  const requestRender = vi.fn();
  // Seconds of scene time are all the manager reads from the clock, so a plain holder that
  // `JulianDate.secondsDifference` understands is enough.
  const clock = { currentTime: sceneTime(0) };
  const viewer = { scene: { preUpdate, requestRender }, clock };

  let loaded: LoadedSiteAsset[] = [
    {
      siteId: SITE_ID,
      siteSlug: "synthetic-tree",
      assetId: ASSET_ID,
      assetName: "Gaussian splat (procedural)",
      representation: "gaussian-splat",
      sourceUrl: "http://localhost/api/v1/tiles/synthetic-tree/splat/tileset.json",
      rigPath: "../source/rig.json",
      shown: true,
      tileset: tileset as unknown as LoadedSiteAsset["tileset"],
    },
  ];
  const sites = {
    loadedAssets: () => loaded,
    tilesetFor: () => (loaded.length > 0 ? loaded[0]?.tileset : null),
  };
  const setAnimating = vi.fn();
  const performance = { setAnimating };
  const events = new Emitter<SceneEvents>();
  const toasts: SceneEvents["toast"][] = [];
  events.on("toast", (toast) => toasts.push(toast));

  const manager = new LivingSurveyManager(
    viewer as unknown as Viewer,
    events,
    sites as unknown as SiteManager,
    performance as unknown as PerformanceManager,
  );

  return {
    manager,
    events,
    toasts,
    preUpdate,
    requestRender,
    setAnimating,
    primitive,
    tileset,
    /** Moves the scene clock to `t` seconds after the epoch, then raises one tick. */
    tick(t: number): void {
      clock.currentTime = sceneTime(t);
      preUpdate.raise();
    },
    unload(): void {
      loaded = [];
      events.emit("asset", { id: ASSET_ID, patch: { loadState: "idle" } });
    },
  };
}

/** Scene time `t` seconds after the manager's epoch — the only clock the manager reads. */
function sceneTime(t: number): JulianDate {
  return JulianDate.addSeconds(JulianDate.fromIso8601(LIVING_EPOCH_ISO), t, new JulianDate());
}

beforeEach(() => {
  clearSplatCaptures();
  useLiving.getState().reset();
});

describe("the scene clock is the only clock", () => {
  it("measures seconds from a fixed epoch, so a frozen clock gives a fixed frame", () => {
    const at = (t: number) => sceneSeconds(sceneTime(t));
    expect(at(0)).toBe(0);
    expect(at(4.25)).toBeCloseTo(4.25, 9);
    expect(at(-30)).toBeCloseTo(-30, 9);
    // The epoch is a constant, not the moment of construction: the same scene time always
    // means the same `t`, in this session and in the next one.
    expect(at(4.25)).toBe(at(4.25));
  });

  it("treats an unusable clock as the epoch rather than producing NaN motion", () => {
    expect(sceneSeconds(new JulianDate(Number.NaN, 0))).toBe(0);
  });
});

describe("attaching", () => {
  it("finds the rig beside the tileset, and only for a rigged splat capture", () => {
    const rig = "../source/rig.json";
    const url = "http://host/api/v1/tiles/synthetic-tree/splat/tileset.json";
    expect(rigUrlFor(rig, "gaussian-splat", url)).toBe(
      "http://host/api/v1/tiles/synthetic-tree/source/rig.json",
    );
    // The same claim resolves against a bucket, because it is relative to the tileset and
    // the rig travels with the tiles. This is the A9 case: the catalog says the same thing
    // and only the host changed.
    expect(
      rigUrlFor(
        rig,
        "gaussian-splat",
        "https://cdn.example.com/sites/synthetic-tree/splat/tileset.json",
      ),
    ).toBe("https://cdn.example.com/sites/synthetic-tree/source/rig.json");
    // A capture the catalog makes no rig claim about, a representation that cannot deform,
    // and an ion asset with no URL to resolve against.
    expect(rigUrlFor(null, "gaussian-splat", url)).toBeNull();
    expect(rigUrlFor("", "gaussian-splat", url)).toBeNull();
    expect(rigUrlFor(rig, "mesh", url)).toBeNull();
    expect(rigUrlFor(rig, "gaussian-splat", null)).toBeNull();
  });

  it("attaches to a loaded rigged site and reports it", async () => {
    const harness = await attached();
    const status = harness.manager.status;
    expect(status.sites).toHaveLength(1);
    expect(status.sites[0]?.phase).toBe("ready");
    expect(status.sites[0]?.numSplats).toBe(FIXTURE_SPLATS);
    expect(status.sites[0]?.rigSourceNote).toBe(fixtureRig.sourceNote);
    harness.manager.destroy();
  });
});

describe("idle stays idle", () => {
  it("asks for a frame while displaced, once more to restore, then never again", async () => {
    const harness = await attached();
    harness.requestRender.mockClear();

    // Calm: the deformer writes nothing, so nothing is worth rendering.
    harness.tick(1);
    harness.tick(2);
    expect(harness.requestRender).not.toHaveBeenCalled();
    // Not one upload, not even on the frame it attached: at rest there is nothing to write.
    expect(harness.primitive.texture.uploads).toHaveLength(0);

    harness.manager.setWind(BREEZE);
    harness.requestRender.mockClear();
    for (const t of [3, 4, 5]) harness.tick(t);
    expect(harness.requestRender).toHaveBeenCalledTimes(3);
    expect(harness.manager.status.sites[0]?.phase).toBe("ready");

    // Back to calm. One restoring write and one last frame to show it, then silence.
    harness.manager.setWind(WIND_CALM);
    harness.requestRender.mockClear();
    const uploadsBefore = harness.primitive.texture.uploads.length;
    harness.tick(6);
    expect(harness.requestRender).toHaveBeenCalledTimes(1);
    expect(harness.primitive.texture.uploads.length).toBe(uploadsBefore + 1);

    harness.requestRender.mockClear();
    for (const t of [7, 8, 9, 10]) harness.tick(t);
    expect(harness.requestRender).not.toHaveBeenCalled();
    expect(harness.primitive.texture.uploads.length).toBe(uploadsBefore + 1);
    harness.manager.destroy();
  });

  it("restores the exact measured bytes, not approximately", async () => {
    const harness = await attached();
    harness.manager.setWind(BREEZE);
    harness.tick(3);
    harness.manager.setWind(WIND_CALM);
    harness.tick(4);

    const restored = harness.primitive.texture.uploads.at(-1);
    expect(restored).toBeDefined();
    const floats = new Float32Array(restored?.words.buffer ?? new ArrayBuffer(0));
    const positions = new Float32Array(2000 * 3);
    for (let i = 0; i < 2000; i += 1) {
      positions[i * 3] = floats[i * 8] ?? 0;
      positions[i * 3 + 1] = floats[i * 8 + 1] ?? 0;
      positions[i * 3 + 2] = floats[i * 8 + 2] ?? 0;
    }
    expect(checksumPositions(positions)).toBe(
      checksumPositions(bakeFixture(canonicalPositions).slice(0, 6000)),
    );
    harness.manager.destroy();
  });

  it("holds the measured pose for a snapshot, and gives it back on release", async () => {
    const harness = await attached();
    harness.manager.setWind(BREEZE);
    harness.tick(3);
    expect(harness.primitive.texture.uploads.at(-1)?.words).toBeDefined();

    const release = harness.manager.holdMeasuredPose();
    harness.tick(4);
    const held = harness.primitive.texture.uploads.at(-1);
    const measured = bakeFixture(canonicalPositions);
    const floats = new Float32Array(held?.words.buffer ?? new ArrayBuffer(0));
    expect(floats[0]).toBe(measured[0]);
    expect(floats[1]).toBe(measured[1]);
    expect(floats[2]).toBe(measured[2]);
    // The wind itself never changed — a snapshot is a photograph, not a change in the weather.
    expect(harness.manager.wind).toEqual(BREEZE);
    expect(harness.manager.status.animating).toBe(true);

    release();
    harness.tick(5);
    expect(harness.manager.status.sites[0]?.phase).toBe("ready");
    harness.manager.destroy();
  });
});

describe("consecutive frames are different frames", () => {
  /**
   * The guard that keeps two indistinguishable failures apart.
   *
   * A tree that bends once and then sits there has exactly two possible causes: a motion model
   * with no motion in it, or a `viewer.clock.currentTime` that has stopped. They look identical
   * on screen and they were confused once already. This asserts the manager's half — one frame
   * of scene time apart, the words that reach the GPU are different words — so a regression in
   * the model can never again be mistaken for a frozen clock, or the reverse.
   */
  const FRAME = 1 / 60;

  it("uploads different splat positions one frame apart at the default wind", async () => {
    const harness = await attached();
    harness.manager.setWind({ strength: DEFAULT_WIND_STRENGTH, bearingDeg: 250 });
    harness.tick(10);
    harness.tick(10 + FRAME);
    const uploads = harness.primitive.texture.uploads;
    expect(uploads.length).toBeGreaterThanOrEqual(2);
    const before = uploads.at(-2);
    const after = uploads.at(-1);
    expect(before).toBeDefined();
    expect(after).toBeDefined();
    if (!before || !after) return;
    expect(after.words.length).toBe(before.words.length);
    let differing = 0;
    for (let i = 0; i < before.words.length; i += 1) {
      if (before.words[i] !== after.words[i]) differing += 1;
    }
    // Not "some bit somewhere moved": nearly the whole tree has to have moved. Of the words in
    // an upload only three per splat are position — the rest are covariance and colour, which
    // the deformer never touches — so the count to measure against is `3 · numSplats`, and one
    // frame at the default wind moves the tip about 1.6 mm, which is many float32 ulps
    // everywhere the tree is not pinned to the ground.
    const positionWords = 3 * (harness.manager.status.sites[0]?.numSplats ?? 0);
    expect(positionWords).toBe(FIXTURE_SPLATS * 3);
    expect(differing).toBeGreaterThan(0.8 * positionWords);
    expect(differing).toBeLessThanOrEqual(positionWords);
    harness.manager.destroy();
  });

  it("uploads identical positions when the clock does not advance", async () => {
    // The other explanation's signature, pinned so the two can be told apart from the outside.
    // `deform` is a pure function of scene time, so a stopped clock gives the same frame for
    // ever — which is correct behaviour for a stopped clock and a bug in whatever stopped it.
    const harness = await attached();
    harness.manager.setWind({ strength: DEFAULT_WIND_STRENGTH, bearingDeg: 250 });
    harness.tick(10);
    harness.tick(10);
    const uploads = harness.primitive.texture.uploads;
    const before = uploads.at(-2);
    const after = uploads.at(-1);
    expect(before).toBeDefined();
    expect(after).toBeDefined();
    if (!before || !after) return;
    expect([...after.words]).toEqual([...before.words]);
    harness.manager.destroy();
  });

  it("keeps moving over a whole second of frames, not just the first two", async () => {
    // A model whose only time-varying term is a 0.22-units-per-second gust envelope passes the
    // two-frame test and still reads as static. Sixty frames is a second of wall time, which is
    // the span a person judges "is this moving?" over.
    const harness = await attached();
    harness.manager.setWind({ strength: DEFAULT_WIND_STRENGTH, bearingDeg: 250 });
    const uploads = harness.primitive.texture.uploads;
    const start = uploads.length;
    for (let frame = 0; frame <= 60; frame += 1) harness.tick(20 + frame * FRAME);
    const frames = uploads.slice(start);
    expect(frames.length).toBe(61);
    let changed = 0;
    for (let i = 1; i < frames.length; i += 1) {
      const a = frames[i - 1];
      const b = frames[i];
      if (!a || !b) continue;
      if (a.words.some((word, index) => word !== b.words[index])) changed += 1;
    }
    expect(changed).toBe(frames.length - 1);
    harness.manager.destroy();
  });
});

describe("telling the performance ladder what is going on", () => {
  it("switches animation on with the wind and off with everything that ends it", async () => {
    const harness = await attached();
    harness.setAnimating.mockClear();

    harness.manager.setWind(BREEZE);
    expect(harness.setAnimating).toHaveBeenLastCalledWith(true);

    harness.setAnimating.mockClear();
    harness.tick(3);
    harness.tick(4);
    // Never per frame: the ladder is told about transitions, not about frames.
    expect(harness.setAnimating).not.toHaveBeenCalled();

    harness.manager.setWind(WIND_CALM);
    expect(harness.setAnimating).toHaveBeenLastCalledWith(false);

    harness.manager.setWind(BREEZE);
    expect(harness.setAnimating).toHaveBeenLastCalledWith(true);
    harness.unload();
    expect(harness.setAnimating).toHaveBeenLastCalledWith(false);
    harness.manager.destroy();
  });

  it("stops animating on destroy", async () => {
    const harness = await attached();
    harness.manager.setWind(BREEZE);
    harness.setAnimating.mockClear();
    harness.manager.destroy();
    expect(harness.setAnimating).toHaveBeenLastCalledWith(false);
  });
});

describe("a site that leaves", () => {
  it("drops the deformer in the same turn, so nothing writes into freed GL objects", async () => {
    const harness = await attached();
    harness.manager.setWind(BREEZE);
    harness.tick(3);
    const uploads = harness.primitive.texture.uploads.length;

    harness.unload();
    expect(harness.manager.status.sites).toHaveLength(0);
    // The tick is not merely idle — it is gone.
    expect(harness.preUpdate.numberOfListeners).toBe(0);
    harness.preUpdate.raise();
    expect(harness.primitive.texture.uploads.length).toBe(uploads);
    harness.manager.destroy();
  });
});

describe("refusals", () => {
  it("toasts a permanent refusal, retires the deformer and never retries", async () => {
    const harness = await attached();
    // A snapshot that now aggregates two tiles: splat indices are no longer stable.
    harness.primitive.selectedTileLength = 2;
    harness.primitive._snapshot = { generation: 2 };
    harness.manager.setWind(BREEZE);
    harness.tick(3);

    expect(harness.manager.status.sites).toHaveLength(0);
    expect(harness.toasts).toHaveLength(1);
    expect(harness.toasts[0]?.tone).toBe("warning");
    expect(harness.toasts[0]?.body).toContain("exactly as it was measured");
    // Retired, so the ladder is told the scene is still again.
    expect(harness.setAnimating).toHaveBeenLastCalledWith(false);

    // An asset event must not start it over.
    harness.events.emit("asset", { id: ASSET_ID, patch: { loadState: "ready" } });
    await Promise.resolve();
    expect(harness.manager.status.sites).toHaveLength(0);
    harness.manager.destroy();
  });

  it("never toasts while it is merely waiting for the engine", async () => {
    const harness = await attached();
    harness.primitive.gaussianSplatTexture = undefined;
    harness.manager.setWind(BREEZE);
    harness.tick(3);
    expect(harness.manager.status.sites[0]?.reason).toBe("no-texture");
    expect(harness.toasts).toHaveLength(0);
    harness.manager.destroy();
  });
});

describe("destroy", () => {
  it("leaves no listeners behind", async () => {
    const harness = await attached();
    expect(harness.preUpdate.numberOfListeners).toBe(1);
    harness.manager.destroy();
    expect(harness.preUpdate.numberOfListeners).toBe(0);

    harness.requestRender.mockClear();
    harness.events.emit("asset", { id: ASSET_ID, patch: { loadState: "ready" } });
    await Promise.resolve();
    harness.preUpdate.raise();
    expect(harness.requestRender).not.toHaveBeenCalled();
    expect(harness.manager.status.sites).toHaveLength(0);
  });
});

describe("the wind store", () => {
  it("starts calm, and clamps whatever it is handed into the model's own domain", () => {
    expect(useLiving.getState().wind).toEqual(WIND_CALM);
    useLiving.getState().setWind({ strength: DEFAULT_WIND_STRENGTH });
    expect(useLiving.getState().wind.strength).toBe(DEFAULT_WIND_STRENGTH);
    useLiving.getState().setWind({ strength: 4, bearingDeg: -30 });
    expect(useLiving.getState().wind).toEqual({ strength: 1, bearingDeg: 330 });
    useLiving.getState().setWind({ strength: Number.NaN });
    expect(useLiving.getState().wind.strength).toBe(0);
  });

  it("defaults well below full strength, from the sort-staleness bound", () => {
    // Full strength is 72 splat radii of stale draw order on a real capture's gaussians; the
    // default has to sit near the threshold the model hypothesises, not near the maximum.
    expect(DEFAULT_WIND_STRENGTH).toBeGreaterThan(0);
    expect(DEFAULT_WIND_STRENGTH).toBeLessThan(0.2);
  });
});

/** A manager with the fixture attached and one tick run, so the deformer is `ready`. */
async function attached(): Promise<ReturnType<typeof createHarness>> {
  const rigText = JSON.stringify(fixtureRig);
  const fetchMock = vi.fn((input: unknown) =>
    Promise.resolve({
      ok: String(input) === RIG_URL,
      text: () => Promise.resolve(rigText),
    } as Response),
  );
  vi.stubGlobal("fetch", fetchMock);
  const harness = createHarness();
  // The rig is fetched; let the attach promise settle before the first tick.
  await vi.waitFor(() => expect(harness.manager.status.sites).toHaveLength(1));
  harness.tick(0);
  vi.unstubAllGlobals();
  return harness;
}
