/**
 * The change drape in the app itself (features/change, ADR 0018): Compare › Dates offers the
 * change map between two dated captures of the yard once both measured a height surface; the
 * map is drawn on the ground under CesiumJS and under PlayCanvas alike; its legend counts each
 * class and says it is derived, uncalibrated, the least change it can show and which date is
 * not survey-grade; and the write token it is behind is asked for, not a blank drape.
 *
 * The `/change` answer is mocked, in the real API's shape, from what the repository commits:
 * the cells P5's reference change lists for `data/tiles/synthetic-yard-surfaces` (e1 -> e2 at
 * 1 m: a raised block, a lowered block, and a weak raised block), the cells it counts as not
 * comparable (one date measured them, the other did not: read here from the two surface.bin
 * planes, as the API compares them), its counts, and A5's evidence for the rest (minDetectableM
 * as the least threshold, the uncertainties, the appearance stub, as ADR 0018 records them). Areas are each class's
 * 4-connected regions, split by confidence as the API splits them, outlined as row runs of
 * cells in the yard's frame (whose origin is the yard tileset's root transform). Unchanged
 * cells are counted, never outlined, as the API now answers.
 *
 * Headless GL is SwiftShader: the drape is asserted through `window.__twin` (the ground
 * primitives it tags) and, by pixels, as the drape's colours appearing on the ground from
 * above where they were not before.
 */
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { gunzipSync } from "node:zlib";

import type { Page, Route } from "@playwright/test";

import { expect, mockApi, test } from "./fixtures";

const TILES = resolve(process.cwd(), "../../data/tiles");
const EVIDENCE = process.env.W5C_EVIDENCE_DIR;
const YARD = "/fixture-tiles/synthetic-yard/splat/tileset.json";
const SITE_ID = "99999999-9999-4999-8999-999999999999";
const E1 = "99999999-9999-4999-8999-000000000001";
const E2 = "99999999-9999-4999-8999-000000000002";
const A1 = "99999999-9999-4999-8999-0000000000a1";
const A2 = "99999999-9999-4999-8999-0000000000a2";
const TOKEN = "pilot-token";

/** The yard's root transform's origin (synthetic_yard.py); the yard is 32 x 24 m east/north. */
const LON = -123.88;
const LAT = 46.1335;
const east = (m: number) => m / (111_320 * Math.cos((LAT * Math.PI) / 180));
const north = (m: number) => m / 111_320;
const MID = { lon: LON + east(16), lat: LAT + north(12) };
const at = (x: number, y: number): [number, number] => [LON + east(x), LAT + north(y)];

const ring = (half: number): [number, number][] => [
  [MID.lon - east(half), MID.lat - north(half)],
  [MID.lon + east(half), MID.lat - north(half)],
  [MID.lon + east(half), MID.lat + north(half)],
  [MID.lon - east(half), MID.lat + north(half)],
  [MID.lon - east(half), MID.lat - north(half)],
];

const CELLS = Array.from({ length: 16 }, (_, i) => {
  const r = 14 * Math.sqrt((i + 0.5) / 16);
  const a = i * 2.399963;
  return { lon: MID.lon + east(r * Math.cos(a)), lat: MID.lat + north(r * Math.sin(a)), height: 0 };
});

const footprint = { type: "MultiPolygon", coordinates: [[ring(40)]] };

const surfaceOf = (assetId: string) => ({
  status: "measured",
  assetId,
  url: `https://tiles.example/${assetId}/surface.bin`,
  cellM: 1,
  level: 2,
  policy: "surface-v1",
  calibration: "uncalibrated",
  cells: 1280,
  surfacedCells: 1272,
  medianSigmaM: 0.02,
  originHeightM: 0,
  groundWindowM: 5,
});

function epoch(
  id: string,
  ordinal: number,
  observedAt: string,
  platform: string,
  asset: string,
  surface: boolean,
) {
  return {
    id,
    siteId: SITE_ID,
    ordinal,
    captureId: null,
    jobId: null,
    observedAt,
    observedAtSource: "captured-at",
    footprint,
    platform,
    status: "published",
    joinMethod: ordinal === 1 ? "new-site" : "spatial",
    // The phone date aligned visually to the drone date, ±0.062 m (A5's evidence): coarse.
    alignment:
      ordinal === 1
        ? null
        : {
            verdict: "coarse",
            uncertaintyM: 0.062,
            label: "Phone video, aligned visually, ±0.062 m",
          },
    quality: null,
    // A map is answered only for dates with a confidence grid too (else 409
    // confidence-not-measured): each of these has one, the phone date capped by ±0.062 m.
    confidence: {
      status: "measured",
      assetId: asset,
      url: `https://tiles.example/${asset}/confidence.bin`,
      cellM: 1,
      level: 2,
      policy: "weak-v1",
      weakShare: 0.02,
      weakCells: 23,
      scoredCells: 1272,
      unseenCells: 0,
      reasons: {},
      alignmentCap: {
        uncertaintyM: ordinal === 1 ? 0 : 0.062,
        factor: 1,
        priorEpochs: ordinal - 1,
        source: ordinal === 1 ? "site-frame" : "epoch_alignment.json",
      },
    },
    surface: surface ? surfaceOf(asset) : null,
    currentAssetId: asset,
    createdAt: observedAt,
    updatedAt: observedAt,
  };
}

function asset(id: string, epochId: string, url: string, createdAt: string) {
  return {
    id,
    siteId: SITE_ID,
    provider: "3d-tiles-url",
    name: "Gaussian splat",
    representation: "gaussian-splat",
    source: { type: "3d-tiles-url", url },
    footprint: null,
    observedAt: createdAt,
    validFrom: null,
    validTo: null,
    resolution: null,
    crs: null,
    license: null,
    attribution: [],
    provenance: null,
    epochId,
    sourceKind: "observed",
    sidecarFlags: [],
    renderConfig: {
      maximumScreenSpaceError: 16,
      pointCloudShading: null,
      clipsWorld: true,
      clipFootprint: "catalog",
      heightOffsetM: 0,
      clampToGround: true,
      groundSamples: CELLS,
    },
    defaultVisible: true,
    createdAt,
    updatedAt: createdAt,
  };
}

function site(surfaces: boolean) {
  return {
    id: SITE_ID,
    slug: "yard-change",
    name: "Yard, what changed",
    description: null,
    boundary: { type: "Polygon", coordinates: [ring(40)] },
    centroid: { longitude: MID.lon, latitude: MID.lat, height: null },
    areaM2: 6_400,
    thumbnailUrl: null,
    metadata: {},
    attribution: [],
    license: null,
    assets: [
      asset(A1, E1, YARD, "2026-03-02T00:00:00Z"),
      asset(A2, E2, `${YARD}?epoch=2`, "2026-06-02T00:00:00Z"),
    ],
    epochs: [
      epoch(E1, 1, "2026-03-01T15:00:00Z", "drone", A1, surfaces),
      epoch(E2, 2, "2026-06-01T15:00:00Z", "phone-video", A2, surfaces),
    ],
    frame: null,
    cameraBookmarks: [],
    createdAt: "2026-03-02T00:00:00Z",
    updatedAt: "2026-06-02T00:00:00Z",
  };
}

const summary = {
  id: SITE_ID,
  slug: "yard-change",
  name: "Yard, what changed",
  description: null,
  centroid: { longitude: MID.lon, latitude: MID.lat, height: null },
  areaM2: 6_400,
  thumbnailUrl: null,
  representations: ["gaussian-splat"],
  latestObservedAt: "2026-06-01T15:00:00Z",
  quality: null,
  createdAt: "2026-03-02T00:00:00Z",
  updatedAt: "2026-06-02T00:00:00Z",
};

// ---- The change map, from the committed surfaces fixture ----------------------------------

interface Level {
  cellM: number;
  i0: number;
  j0: number;
  ni: number;
  nj: number;
  offset: number;
}
interface SurfacesFixture {
  epochs: Record<"e1" | "e2", { descriptor: { levels: Level[] }; file: string }>;
  change: {
    cellM: number;
    counts: { raised: number; lowered: number; unchanged: number; insufficient: number };
    changed: {
      i: number;
      j: number;
      deltaM: number;
      thresholdM: number;
      class: string;
      weak: boolean;
    }[];
  };
}

const SURFACES = resolve(TILES, "synthetic-yard-surfaces");

/** The cells a surface measured (dsm not noData), at its own level, as "i,j". */
function measuredCells(
  fixture: SurfacesFixture,
  key: "e1" | "e2",
): { cellM: number; cells: Set<string> } {
  const { descriptor, file } = fixture.epochs[key];
  const level = descriptor.levels[0];
  if (!level) throw new Error(`${key}: no level`);
  const planes = gunzipSync(readFileSync(resolve(SURFACES, file)));
  const cells = new Set<string>();
  // Channels dsm i2, ground i2, sigma u1, support u1: six bytes a cell, little-endian.
  for (let j = 0; j < level.nj; j++)
    for (let i = 0; i < level.ni; i++)
      if (planes.readInt16LE(level.offset + (j * level.ni + i) * 6) !== -32768)
        cells.add(`${level.i0 + i},${level.j0 + j}`);
  return { cellM: level.cellM, cells };
}

/** Cells at the coarser level `cellM` either measured but no child of which both did. */
function insufficientCells(fixture: SurfacesFixture, cellM: number): [number, number][] {
  const up = (side: { cellM: number; cells: Set<string> }) => {
    const f = Math.round(cellM / side.cellM);
    return new Set(
      [...side.cells].map((key) => {
        const [i = 0, j = 0] = key.split(",").map(Number);
        return `${Math.floor(i / f)},${Math.floor(j / f)}`;
      }),
    );
  };
  const [a, b] = [up(measuredCells(fixture, "e1")), up(measuredCells(fixture, "e2"))];
  return [...new Set([...a, ...b])]
    .filter((key) => !(a.has(key) && b.has(key)))
    .map((key) => key.split(",").map(Number) as [number, number]);
}

/** 4-connected regions of cells. */
function regions(cells: [number, number][]): [number, number][][] {
  const left = new Set(cells.map(([i, j]) => `${i},${j}`));
  const out: [number, number][][] = [];
  for (const [i, j] of cells) {
    if (!left.delete(`${i},${j}`)) continue;
    const region: [number, number][] = [[i, j]];
    // Cells pushed while walking are walked too.
    for (const [x, y] of region) {
      for (const [dx, dy] of [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
      ] as const) {
        if (left.delete(`${x + dx},${y + dy}`)) region.push([x + dx, y + dy]);
      }
    }
    out.push(region);
  }
  return out;
}

/** A region outlined as one rectangle per run of cells along a row, in the yard's frame. */
function outline(region: [number, number][], cellM: number) {
  const rows = new Map<number, number[]>();
  for (const [i, j] of region) rows.set(j, [...(rows.get(j) ?? []), i]);
  const polygons: [number, number][][][] = [];
  for (const [j, is] of rows) {
    is.sort((p, q) => p - q);
    let start = is[0] ?? 0;
    for (let k = 1; k <= is.length; k++) {
      const i = is[k];
      if (i !== undefined && i === (is[k - 1] ?? 0) + 1) continue;
      const end = (is[k - 1] ?? 0) + 1;
      const [x0, x1, y0, y1] = [start * cellM, end * cellM, j * cellM, (j + 1) * cellM];
      polygons.push([[at(x0, y0), at(x1, y0), at(x1, y1), at(x0, y1), at(x0, y0)]]);
      if (i !== undefined) start = i;
    }
  }
  return { type: "MultiPolygon", coordinates: polygons };
}

function changeMap() {
  const fixture = JSON.parse(
    readFileSync(resolve(SURFACES, "surfaces.json"), "utf8"),
  ) as SurfacesFixture;
  const { cellM, counts, changed } = fixture.change;
  // What the fixture does not carry, as A5's evidence answered it (ADR 0018 § API Evidence).
  const evidence = {
    appearance: {
      status: "not-computed",
      reason:
        "needs the new photos compared with the old epoch at fixed geometry; never splat pixels",
    },
    uncertainty: { fromM: 0, toM: 0.062 },
  };
  const areas: Record<string, unknown>[] = [];
  for (const cls of ["raised", "lowered"]) {
    for (const weak of [false, true]) {
      const cells = changed.filter((c) => c.class === cls && c.weak === weak);
      for (const region of regions(cells.map((c) => [c.i, c.j]))) {
        const deltas = region.map(
          ([i, j]) => cells.find((c) => c.i === i && c.j === j)?.deltaM ?? 0,
        );
        areas.push({
          class: cls,
          footprint: outline(region, cellM),
          cells: region.length,
          areaM2: region.length * cellM * cellM,
          lowConfidence: weak,
          lowConfidenceCells: weak ? region.length : 0,
          meanDeltaM: deltas.reduce((s, d) => s + d, 0) / deltas.length,
          maxAbsDeltaM: Math.max(...deltas.map(Math.abs)),
          meanGroundDeltaM: 0,
        });
      }
    }
  }
  const insufficient = insufficientCells(fixture, cellM);
  expect(insufficient).toHaveLength(counts.insufficient);
  for (const region of regions(insufficient))
    areas.push({
      class: "insufficient",
      footprint: outline(region, cellM),
      cells: region.length,
      areaM2: region.length * cellM * cellM,
      lowConfidence: false,
      lowConfidenceCells: 0,
      meanDeltaM: null,
      maxAbsDeltaM: null,
      meanGroundDeltaM: null,
    });
  return {
    policy: "change-v1",
    derived: true,
    label: "Derived from height surfaces, not observed",
    calibration: "uncalibrated",
    siteId: SITE_ID,
    fromEpochId: E1,
    toEpochId: E2,
    fromAssetId: A1,
    toAssetId: A2,
    frame: "tileset",
    level: 2,
    nativeLevel: 2,
    cellM,
    thresholdSigma: 3,
    minDetectableM: Math.min(...changed.map((c) => c.thresholdM)),
    uncertainty: evidence.uncertainty,
    counts,
    lowConfidence: {
      // The fixture lists only changed cells' confidence; unchanged ones are not known here.
      unchanged: 0,
      raised: changed.filter((c) => c.class === "raised" && c.weak).length,
      lowered: changed.filter((c) => c.class === "lowered" && c.weak).length,
    },
    appearance: evidence.appearance,
    areaLevel: 2,
    areaCellM: cellM,
    areas,
  };
}

// ---- The app ---------------------------------------------------------------------------------

/** E2's tileset: the yard's own root transform around the synthetic tree's one tile. */
function secondDate(): string {
  const yard = JSON.parse(
    readFileSync(resolve(TILES, "synthetic-yard/splat/tileset.json"), "utf8"),
  ) as { root: { transform: number[] } };
  const tree = JSON.parse(
    readFileSync(resolve(TILES, "synthetic-tree/splat/tileset.json"), "utf8"),
  ) as {
    root: {
      content: { uri: string };
      extras: { viewCones: { uri: string }; collision: { uri: string } };
    } & Record<string, unknown>;
  } & Record<string, unknown>;
  const base = "/fixture-tiles/synthetic-tree/splat/";
  const extras = tree.root.extras;
  return JSON.stringify({
    ...tree,
    root: {
      ...tree.root,
      transform: yard.root.transform,
      content: { uri: base + tree.root.content.uri },
      extras: {
        ...extras,
        viewCones: { ...extras.viewCones, uri: base + extras.viewCones.uri },
        collision: { ...extras.collision, uri: base + extras.collision.uri },
      },
    },
  });
}

interface ChangeCalls {
  asked: { from: string | null; to: string | null; auth: string | null }[];
}

async function route(
  page: Page,
  options: { surfaces?: boolean; token?: boolean } = {},
): Promise<ChangeCalls> {
  const calls: ChangeCalls = { asked: [] };
  const map = changeMap();
  await mockApi(page);
  await page.route("**/api/v1/sites", (r) =>
    r.request().method() === "GET" ? r.fulfill({ status: 200, json: [summary] }) : r.fallback(),
  );
  await page.route(`**/api/v1/sites/${SITE_ID}`, (r) =>
    r.fulfill({ status: 200, json: site(options.surfaces ?? true) }),
  );
  await page.route(`**/api/v1/sites/${SITE_ID}/change?*`, (r: Route) => {
    const url = new URL(r.request().url());
    const auth = r.request().headers().authorization ?? null;
    calls.asked.push({ from: url.searchParams.get("from"), to: url.searchParams.get("to"), auth });
    if (options.token && auth !== `Bearer ${TOKEN}`)
      return r.fulfill({
        status: 401,
        json: { type: "about:blank", title: "Unauthorized", status: 401, detail: "write token" },
      });
    return r.fulfill({ status: 200, json: map });
  });
  const second = secondDate();
  await page.route("**/fixture-tiles/**", (r) => {
    const url = new URL(r.request().url());
    const relative = url.pathname.replace(/^.*\/fixture-tiles\//, "");
    if (relative.includes("..")) return r.abort();
    if (relative === "synthetic-yard/splat/tileset.json" && url.searchParams.get("epoch") === "2")
      return r.fulfill({ status: 200, contentType: "application/json", body: second });
    const file = resolve(TILES, relative);
    if (!existsSync(file)) return r.fulfill({ status: 404, body: "" });
    return r.fulfill({
      status: 200,
      contentType: relative.endsWith(".json") ? "application/json" : "application/octet-stream",
      body: readFileSync(file),
    });
  });
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (r) => r.abort());
  return calls;
}

async function arrive(page: Page, renderer: string): Promise<void> {
  await page.setViewportSize({ width: 1280, height: 760 });
  await page.goto(`/?renderer=${renderer}`);
  await page.waitForFunction(() => "__twin" in window && Boolean(window.__twin), undefined, {
    timeout: 120_000,
  });
  await page.getByTestId("onboarding-explore").click();
  await page.waitForFunction(`window.__twin.sites.summaries.length === 1`, undefined, {
    timeout: 60_000,
  });
  await page.evaluate(`void window.__twin.sites.flyTo("${SITE_ID}")`);
  await page.waitForFunction(
    `window.__twin.sites.flight?.state === "landed" && !window.__twin.camera.gliding && window.__twin.sites.activeSite?.id === "${SITE_ID}"`,
    undefined,
    { timeout: 240_000 },
  );
}

/** Straight down over the yard from `heightM`, north up: the whole of the drape in view. */
async function lookDown(page: Page, heightM: number): Promise<void> {
  const [a, e2] = [6_378_137, 6.69437999014e-3];
  const [phi, lambda] = [(MID.lat * Math.PI) / 180, (MID.lon * Math.PI) / 180];
  const n = a / Math.sqrt(1 - e2 * Math.sin(phi) ** 2);
  const [x, y, z] = [
    (n + heightM) * Math.cos(phi) * Math.cos(lambda),
    (n + heightM) * Math.cos(phi) * Math.sin(lambda),
    (n * (1 - e2) + heightM) * Math.sin(phi),
  ];
  await page.evaluate(`(() => {
    const twin = window.__twin;
    const camera = twin.viewer.camera;
    const Cartesian3 = camera.position.constructor;
    camera.setView({
      destination: new Cartesian3(${x}, ${y}, ${z}),
      orientation: { heading: 0, pitch: -Math.PI / 2, roll: 0 },
    });
    twin.scene.requestRender();
  })()`);
}

interface DrapeTag {
  cls: string;
  lowConfidence: boolean;
  polygons: number;
  hatched: boolean;
  ready: boolean;
  show: boolean;
}

/** The ground primitives the drape drew, as it tagged them. */
const drawn = (page: Page): Promise<DrapeTag[]> =>
  page.evaluate(`(() => {
    const ground = window.__twin.scene.groundPrimitives;
    const out = [];
    for (let i = 0; i < ground.length; i++) {
      const p = ground.get(i);
      if (p && p.changeDrape) out.push({ ...p.changeDrape, ready: Boolean(p.ready), show: p.show });
    }
    return out;
  })()`);

/** A screenshot once the scene has drawn a few frames at rest; saved as evidence when named. */
async function settledShot(page: Page, name?: string): Promise<Buffer> {
  await page.evaluate(`window.__twin.scene.requestRender()`);
  await page.waitForTimeout(3_000);
  await page.evaluate(`window.__twin.scene.requestRender()`);
  await page.waitForTimeout(1_000);
  if (!name) return page.screenshot();
  const path = EVIDENCE ? `${EVIDENCE}/${name}.png` : test.info().outputPath(`${name}.png`);
  const shot = await page.screenshot({ path });
  await test.info().attach(name, { path, contentType: "image/png" });
  return shot;
}

/**
 * The share of the view's pixels near each drape colour (summed RGB distance under 90),
 * over the middle of the view, clear of the HUD: raised (orange) and lowered (blue).
 */
async function drapePixels(page: Page, shot: Buffer): Promise<{ raised: number; lowered: number }> {
  return page.evaluate(async (data) => {
    const image = new Image();
    image.src = `data:image/png;base64,${data}`;
    await image.decode();
    const canvas = document.createElement("canvas");
    canvas.width = image.width;
    canvas.height = image.height;
    const context = canvas.getContext("2d");
    if (!context) return { raised: 0, lowered: 0 };
    context.drawImage(image, 0, 0);
    const [x, y] = [Math.floor(image.width * 0.1), Math.floor(image.height * 0.1)];
    const [w, h] = [Math.floor(image.width * 0.55), Math.floor(image.height * 0.65)];
    const px = context.getImageData(x, y, w, h).data;
    const near = (r: number, g: number, b: number, i: number) =>
      Math.abs((px[i] ?? 0) - r) + Math.abs((px[i + 1] ?? 0) - g) + Math.abs((px[i + 2] ?? 0) - b) <
      90;
    let raised = 0;
    let lowered = 0;
    for (let i = 0; i < px.length; i += 4) {
      // Orange #f08c3a and blue #3a9cf0 over the ground, at the drape's opacity.
      if (px[i]! > px[i + 2]! + 60 && near(220, 140, 80, i)) raised += 1;
      if (px[i + 2]! > px[i]! + 60 && near(80, 150, 220, i)) lowered += 1;
    }
    const total = px.length / 4;
    return { raised: raised / total, lowered: lowered / total };
  }, shot.toString("base64"));
}

async function railUnchanged(page: Page): Promise<void> {
  await expect(page.getByRole("toolbar", { name: "Tools" }).getByRole("button")).toHaveText([
    ...(process.env.VITE_ENABLE_LAND_EXPLORATION === "true" ? ["Land"] : []),
    "Layers",
    "Measure",
    "Add",
    "Settings",
  ]);
}

async function openCompare(page: Page) {
  await page.getByTestId("tool-layers").click();
  await page.getByTestId("layers-mode").getByRole("radio", { name: "Compare" }).click();
  const dates = page.getByTestId("compare-dates");
  await expect(dates).toBeVisible({ timeout: 30_000 });
  return dates;
}

async function closePanel(page: Page) {
  await page.getByTestId("layers-panel").getByRole("button", { name: "Close panel" }).click();
  await expect(page.getByTestId("layers-panel")).toHaveCount(0);
}

const poll = { timeout: 120_000, intervals: [250, 500, 1000] };

const EXPECTED = [
  { cls: "raised", lowConfidence: false, hatched: false },
  { cls: "raised", lowConfidence: true, hatched: true },
  { cls: "lowered", lowConfidence: false, hatched: false },
  { cls: "insufficient", lowConfidence: false, hatched: false },
];

for (const renderer of ["cesium", "playcanvas", "spark"] as const) {
  test(`under ${renderer}, Compare › Dates drapes the change between two dates and keys it`, async ({
    page,
  }) => {
    test.setTimeout(420_000);
    const calls = await route(page);
    await arrive(page, renderer);
    // The drone date: the yard, its ground covered in splats, which the drape lies under.
    await page
      .getByTestId("timeline")
      .getByRole("radio", { name: /Mar 1, 2026/ })
      .click();
    await expect
      .poll(() => page.evaluate(`window.__twin.sites.shownAsset()?.id ?? null`), poll)
      .toBe(A1);
    await lookDown(page, 70);
    const before = await settledShot(page, `${renderer}-before`);

    const dates = await openCompare(page);
    const change = dates.getByRole("switch", { name: "Change map" });
    await expect(change).toBeEnabled();
    await change.click();
    await expect(change).toHaveAttribute("aria-checked", "true");
    await expect.poll(() => calls.asked.at(-1), poll).toMatchObject({ from: E1, to: E2 });

    // On the ground: one primitive per class and confidence, ready, whatever draws the splats.
    await expect
      .poll(
        async () =>
          (await drawn(page)).map(({ cls, lowConfidence, hatched }) => ({
            cls,
            lowConfidence,
            hatched,
          })),
        poll,
      )
      .toEqual(EXPECTED);
    await expect
      .poll(async () => (await drawn(page)).every((d) => d.ready && d.show), poll)
      .toBe(true);

    // The legend: counts from the fixture, low confidence hatched, derived, the floor,
    // uncalibrated, and the phone date named as not survey-grade.
    const legend = page.getByTestId("change-drape");
    await expect(legend).toBeVisible();
    await expect(page.getByTestId("change-drape-label")).toHaveText(
      "Derived from height surfaces — not observed · changes under ±0.23 m cannot be seen between these dates",
    );
    await expect(page.getByTestId("change-row-raised")).toContainText("47 cells");
    await expect(page.getByTestId("change-row-raised")).toContainText("23 low confidence, hatched");
    await expect(page.getByTestId("change-row-lowered")).toContainText("24 cells");
    await expect(page.getByTestId("change-row-insufficient")).toContainText("72 cells");
    await expect(page.getByTestId("change-row-unchanged")).toContainText(
      "1,137 cells · counted, not drawn",
    );
    await expect(legend).toContainText("Uncalibrated: the 3σ rule and the 5 m ground window");
    await expect(page.getByTestId("change-drape-placement")).toHaveText(
      "Jun 1, 2026: Phone video, aligned visually, ±0.062 m — not survey-grade",
    );

    // It outlives the panel, and shows on the ground from above.
    await closePanel(page);
    await expect(legend).toBeVisible();
    await lookDown(page, 70);
    const after = await settledShot(page, `${renderer}-drape`);
    const [off, on] = [await drapePixels(page, before), await drapePixels(page, after)];
    test
      .info()
      .annotations.push({ type: "drape pixels", description: JSON.stringify({ off, on }) });
    // The drape lies under the splats: dense ground splats (the yard's grass, thicker under
    // PlayCanvas) cover most of it. The raised block shows through the grass at the yard's
    // south-west corner under both; the lowered block lies under the shed and a tree, so it is
    // recorded above but not asserted.
    expect(on.raised).toBeGreaterThan(off.raised + 0.001);

    // Hide: off the ground, and the legend gone.
    await legend.getByRole("button", { name: "Hide change map" }).click();
    await expect(legend).toHaveCount(0);
    await expect.poll(async () => (await drawn(page)).length, poll).toBe(0);
    await railUnchanged(page);
  });
}

test("the change map asks for the pilot's write token, then draws once it is given", async ({
  page,
}) => {
  test.setTimeout(360_000);
  const calls = await route(page, { token: true });
  await arrive(page, "cesium");
  const dates = await openCompare(page);
  await dates.getByRole("switch", { name: "Change map" }).click();
  const error = page.getByTestId("change-drape-error");
  await expect(error).toContainText("needs the pilot's write token");
  await expect(page.getByTestId("compare-change-refused")).toContainText(
    "needs the pilot's write token",
  );
  expect(calls.asked.at(-1)?.auth).toBeNull();
  expect(await drawn(page)).toEqual([]);
  await settledShot(page, "needs-token");

  await error.getByRole("button", { name: "Open Settings" }).click();
  await page.getByTestId("settings-write-token").fill(TOKEN);
  await expect.poll(() => calls.asked.at(-1)?.auth ?? null, poll).toBe(`Bearer ${TOKEN}`);
  await expect(page.getByTestId("change-drape-label")).toBeVisible({ timeout: 30_000 });
  await expect(error).toHaveCount(0);
  await expect.poll(async () => (await drawn(page)).length, poll).toBe(EXPECTED.length);
  await railUnchanged(page);
});

test("dates with no height surface: not offered, and Compare says why", async ({ page }) => {
  test.setTimeout(300_000);
  const calls = await route(page, { surfaces: false });
  await arrive(page, "cesium");
  const dates = await openCompare(page);
  await expect(dates.getByRole("switch", { name: "Change map" })).toBeDisabled();
  await expect(page.getByTestId("compare-change-not-offered")).toHaveText(
    "Not offered: neither Mar 1, 2026 nor Jun 1, 2026 has a height surface measured",
  );
  await settledShot(page, "not-offered");
  expect(calls.asked).toEqual([]);
  await railUnchanged(page);
});
