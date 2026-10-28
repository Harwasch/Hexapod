// Build a dedicated research entry and serve only its static output. No API or writes.
import { cp, mkdir, readFile, writeFile, stat } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { createServer } from "node:http";
import { homedir } from "node:os";
import { resolve, extname, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { build, mergeConfig } from "vite";
import config from "../../vite.config.ts";

const web = fileURLToPath(new URL("../../", import.meta.url));
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const i = args.indexOf(name);
  return i < 0 ? fallback : args[i + 1];
};
const host = option("--host", "127.0.0.1");
const port = Number(option("--port", "5720"));
if (!host || ["0.0.0.0", "::"].includes(host)) throw new Error("Use a specific interface address");
if (!Number.isInteger(port) || port < 5720 || port > 5729) throw new Error("Ports 5720–5729 only");
const cache =
  process.env.LIVING_WEATHER_CACHE ?? resolve(homedir(), ".cache/hexapod-living-weather");
const out = resolve(cache, "bench");
const source = resolve(
  option("--source", resolve(homedir(), ".cache/hexapod-living-breeze/source")),
);
const bake = resolve(option("--bake", resolve(homedir(), ".cache/hexapod-living-breeze/bake")));
config.build.rolldownOptions.input = {};
// App-specific manual groups create a CJS-helper cycle for this standalone entry.
config.build.rolldownOptions.output = {};
await build(
  mergeConfig(config, {
    configFile: false,
    root: web,
    publicDir: false,
    build: {
      outDir: out,
      emptyOutDir: true,
      sourcemap: false,
      rolldownOptions: { input: { bench: resolve(web, "scripts/living-lab/bench.html") } },
    },
  }),
);
await mkdir(`${out}/data`, { recursive: true });
await cp(`${source}/splat`, `${out}/data/tree`, { recursive: true });
await cp(bake, `${out}/data/bake`, { recursive: true });
await cp(resolve(option("--forest", `${cache}/forest`)), `${out}/data/forest`, { recursive: true });
try {
  await cp(`${cache}/scans/phone-capture-2026-09-26-20-41-utc/splat`, `${out}/data/spool`, {
    recursive: true,
  });
} catch (e) {
  if (e.code !== "ENOENT") throw e;
}
// The existing packets are not leaf instances; the weather arm disables their flutter.
const headerPath = `${out}/data/bake/articulation.json`;
const header = JSON.parse(await readFile(headerPath, "utf8"));
header.detail = {
  vegetation: true,
  fittedRig: true,
  separableCrownShare: 0,
  attachedCrownShare: 0,
  smallSplatShare: 0,
  leafBindings: false,
};
const detailPath = option("--detail", `${cache}/minnetonka-detail.json`);
try {
  Object.assign(header.detail, JSON.parse(await readFile(detailPath, "utf8")));
} catch (e) {
  if (e.code !== "ENOENT") throw e;
}
await writeFile(headerPath, JSON.stringify(header));
// The same depth-two LOD as round 3, with canonical GLB bytes preserved.
const manifest = JSON.parse(await readFile(`${source}/splat/tileset.json`, "utf8"));
manifest.root.extras ??= {};
manifest.root.extras.nativeLod = false;
function prune(tile, depth) {
  if (depth >= 2) delete tile.children;
  tile.geometricError = tile.children?.length ? 1000 : 0;
  for (const child of tile.children ?? []) prune(child, depth + 1);
}
prune(manifest.root, 0);
await writeFile(`${out}/data/tree/bench-tileset.json`, JSON.stringify(manifest));
const mime = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".wasm": "application/wasm",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".ktx2": "image/ktx2",
};
createServer(async (req, res) => {
  try {
    const path = decodeURIComponent(new URL(req.url, "http://local").pathname);
    const file = resolve(out, "." + (path === "/" ? "/scripts/living-lab/bench.html" : path));
    if (!file.startsWith(out + sep) || !(await stat(file)).isFile()) {
      res.writeHead(404).end();
      return;
    }
    res.setHeader("Content-Type", mime[extname(file)] ?? "application/octet-stream");
    createReadStream(file).pipe(res);
  } catch {
    res.writeHead(404).end();
  }
}).listen(port, host, () =>
  console.info(`Simulated weather/forest bench: http://${host}:${port}/`),
);
