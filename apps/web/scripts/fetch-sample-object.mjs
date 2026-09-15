#!/usr/bin/env node
// Fetches a CC0 photoscanned object from Poly Haven, packs it into a single-tile 3D Tiles
// tileset under apps/web/public/samples/<slug>/ (served by the Vite dev server), and
// registers it as a site in the catalog so it shows up in the Sites panel.
//
//   node scripts/fetch-sample-object.mjs --slug rock_09 --lon -122.1385 --lat 47.645
//
// Options: --res 1k|2k|4k|8k (default 4k), --api http://127.0.0.1:8000, --web
// http://localhost:5173, --name "Sample object", --heading 0. The object is clamped to the
// terrain by the viewer, so no height is needed. Nothing here is committed: the samples
// directory is gitignored and the script is re-runnable (it updates an existing site).

import { mkdir, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { Cartesian3, Math as CesiumMath, Matrix3, Matrix4, Transforms } from "cesium";
import { NodeIO, getBounds } from "@gltf-transform/core";

const args = parseArgs(process.argv.slice(2));
const slug = args.slug ?? "rock_09";
const res = args.res ?? "4k";
const lon = Number(args.lon ?? -122.1385);
const lat = Number(args.lat ?? 47.645);
const heading = Number(args.heading ?? 0);
const api = (args.api ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const web = (args.web ?? "http://localhost:5173").replace(/\/$/, "");

const here = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.resolve(here, "..", "public", "samples", slug);
const srcDir = path.join(outDir, "src");
await mkdir(srcDir, { recursive: true });

// 1. Manifest and metadata from the Poly Haven API (CC0 assets, attribution appreciated).
const info = await json(`https://api.polyhaven.com/info/${slug}`);
const files = await json(`https://api.polyhaven.com/files/${slug}`);
const entry = files.gltf?.[res]?.gltf;
if (!entry)
  throw new Error(`no glTF at ${res} for ${slug}; available: ${Object.keys(files.gltf ?? {})}`);
const authors = Object.keys(info.authors ?? {}).join(", ") || "Poly Haven";
console.info(
  `${info.name} by ${authors} (CC0) — ${res} glTF, ${mb(entry.size + sum(entry.include))} MB`,
);

// 2. Download the .gltf and every file it references.
const gltfPath = path.join(srcDir, path.basename(entry.url));
await download(entry.url, gltfPath);
for (const [relative, file] of Object.entries(entry.include ?? {})) {
  await download(file.url, path.join(srcDir, relative));
}

// 3. Pack into one GLB and measure it (glTF is Y-up; 3D Tiles content is converted to Z-up).
const io = new NodeIO();
const document = await io.read(gltfPath);
const scene = document.getRoot().getDefaultScene() ?? document.getRoot().listScenes()[0];
const { min, max } = getBounds(scene);
const glbPath = path.join(outDir, "model.glb");
await writeFile(glbPath, await io.writeBinary(document));
const size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
console.info(`model.glb written; size ${size.map((v) => (v * 1000).toFixed(0)).join(" × ")} mm`);

// 4. Single-tile 3D Tiles 1.1 tileset placed on the globe with an east-north-up frame.
//    Bounding box is in the tile frame: Y-up glTF (x, y, z) → Z-up tile (x, -z, y).
const center = [(min[0] + max[0]) / 2, -(min[2] + max[2]) / 2, (min[1] + max[1]) / 2];
const half = [size[0] / 2, size[2] / 2, size[1] / 2];
const enu = Transforms.eastNorthUpToFixedFrame(Cartesian3.fromDegrees(lon, lat, 0));
// Heading turns the model about local up (clockwise from north, like a compass).
const spin = Matrix4.fromRotationTranslation(Matrix3.fromRotationZ(CesiumMath.toRadians(-heading)));
const transform = Matrix4.multiply(enu, spin, new Matrix4());
const tileset = {
  asset: { version: "1.1", generator: "hexapod fetch-sample-object" },
  geometricError: 1000,
  root: {
    transform: Matrix4.toArray(transform),
    boundingVolume: {
      box: [center[0], center[1], center[2], half[0], 0, 0, 0, half[1], 0, 0, 0, half[2]],
    },
    geometricError: 0,
    refine: "REPLACE",
    content: { uri: "model.glb" },
  },
};
await writeFile(path.join(outDir, "tileset.json"), JSON.stringify(tileset, null, 2));
const url = `${web}/samples/${slug}/tileset.json`;
console.info(`tileset: ${url}`);

// 5. Register (or update) the site through the catalog API.
const radiusDeg = 0.0006;
const boundary = {
  type: "Polygon",
  coordinates: [
    [
      [lon - radiusDeg, lat - radiusDeg],
      [lon + radiusDeg, lat - radiusDeg],
      [lon + radiusDeg, lat + radiusDeg],
      [lon - radiusDeg, lat + radiusDeg],
      [lon - radiusDeg, lat - radiusDeg],
    ],
  ],
};
const siteSlug = `sample-object-${slug.replace(/_/g, "-")}`;
const attribution = [
  {
    text: `${info.name} by ${authors}, Poly Haven (CC0)`,
    organization: "Poly Haven",
    url: `https://polyhaven.com/a/${slug}`,
  },
];
const site = {
  slug: siteSlug,
  name: args.name ?? `Sample object: ${info.name}`,
  description:
    `Photoscanned object from Poly Haven (${authors}, CC0), ${size.map((v) => (v * 100).toFixed(0)).join(" × ")} cm, ` +
    `${res} textures. Served from the dev server and clamped to the terrain by the viewer; a stand-in for a phone scan ` +
    `so millimetre-scale zoom can be tested before capturing anything.`,
  boundary,
  centroid: { longitude: lon, latitude: lat, height: 0 },
  metadata: {
    comparison: true,
    origin: "polyhaven",
    quality: { resolutionDescription: `${res} photoscan textures` },
  },
  attribution,
  license: {
    name: "CC0 1.0",
    spdxId: "CC0-1.0",
    url: "https://creativecommons.org/publicdomain/zero/1.0/",
    requiresAttribution: false,
  },
  assets: [
    {
      name: `${info.name} (${res} photoscan mesh)`,
      representation: "mesh",
      source: { type: "3d-tiles-url", url },
      footprint: boundary,
      attribution,
      provenance: {
        sourceOrganization: "Poly Haven",
        sourceUrl: `https://polyhaven.com/a/${slug}`,
        notes: "Packed into a single-tile 3D Tiles tileset by fetch-sample-object.mjs.",
      },
      renderConfig: { clipsWorld: false, clampToGround: true, maximumScreenSpaceError: 1 },
      defaultVisible: true,
    },
  ],
  cameraBookmarks: [],
};
const existing = await json(`${api}/api/v1/sites`).catch(() => []);
const found = Array.isArray(existing) ? existing.find((s) => s.slug === siteSlug) : undefined;
if (found) {
  await request(`${api}/api/v1/sites/${found.id}`, "DELETE").catch(() => undefined);
}
const created = await request(`${api}/api/v1/sites`, "POST", site);
console.info(
  `site registered: ${created.name} (${created.id}) — open the Sites panel and fly to it`,
);

// ---------------------------------------------------------------------------------------
function parseArgs(list) {
  const out = {};
  for (let i = 0; i < list.length; i++) {
    const a = list[i];
    if (a.startsWith("--"))
      out[a.slice(2)] = list[i + 1] && !list[i + 1].startsWith("--") ? list[++i] : "true";
  }
  return out;
}
async function json(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}
async function request(url, method, body) {
  const r = await fetch(url, {
    method,
    headers: { "content-type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(`${method} ${url}: ${r.status} ${await r.text()}`);
  return r.status === 204 ? null : r.json();
}
async function download(url, to) {
  if (existsSync(to)) return;
  await mkdir(path.dirname(to), { recursive: true });
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  await writeFile(to, Buffer.from(await r.arrayBuffer()));
  console.info(`  fetched ${path.basename(to)}`);
}
function sum(include) {
  return Object.values(include ?? {}).reduce((a, f) => a + f.size, 0);
}
function mb(bytes) {
  return (bytes / 1048576).toFixed(1);
}
