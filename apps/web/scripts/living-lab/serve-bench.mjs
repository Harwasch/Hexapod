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
const port = Number(option("--port", "5710"));
if (!host || ["0.0.0.0", "::"].includes(host)) throw new Error("Use a specific interface address");
if (!Number.isInteger(port) || port < 5710 || port > 5719) throw new Error("Ports 5710–5719 only");
const cache = process.env.LIVING_BREEZE_CACHE ?? resolve(homedir(), ".cache/hexapod-living-breeze");
const out = resolve(cache, "bench");
const source = resolve(option("--source", `${cache}/source`));
const bake = resolve(option("--bake", `${cache}/bake`));
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
}).listen(port, host, () => console.info(`Simulated breeze bench: http://${host}:${port}/`));
