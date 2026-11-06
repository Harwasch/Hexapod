// Isolated CPU evaluation, no renderer or per-splat evaluation. Run via heavy.
import { build } from "vite";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { homedir } from "node:os";
import { pathToFileURL } from "node:url";
const root = resolve(import.meta.dirname, "../../../.."),
  cache = resolve(homedir(), ".cache/hexapod-living-weather");
const out = resolve(process.env.LAB_OUT ?? `${cache}/artifacts`);
await mkdir(out, { recursive: true });
const entry = `${cache}/cpu-entry.ts`;
await writeFile(
  entry,
  `export * from ${JSON.stringify(resolve(root, "apps/web/src/cesium/scanView/livingArticulation.ts"))};\nexport { prepareWeatherWind } from ${JSON.stringify(resolve(root, "packages/world/src/weatherWind.ts"))};\n`,
);
await build({
  configFile: false,
  resolve: { alias: { "@twin/world": resolve(root, "packages/world/src/index.ts") } },
  ssr: { noExternal: true },
  build: {
    ssr: entry,
    target: "node22",
    outDir: `${cache}/cpu`,
    emptyOutDir: true,
    minify: false,
    rolldownOptions: { output: { entryFileNames: "model.mjs" } },
  },
});
const m = await import(pathToFileURL(`${cache}/cpu/model.mjs`));
const json = async (p) => JSON.parse(await readFile(p, "utf8"));
const fixture = await json(
  resolve(root, "apps/web/src/lib/weather/fixtures/minnetonka-hours.json"),
);
const summary = [];
for (const scene of ["tree", "forest"]) {
  const folder =
    scene === "tree" ? resolve(homedir(), ".cache/hexapod-living-breeze/bake") : `${cache}/forest`;
  const h = await json(`${folder}/articulation.json`);
  const buffers = await Promise.all(
    [h.motionBinary, h.leafBinary, h.bindingBinary].map(async (p) => {
      const b = await readFile(`${folder}/${p}`);
      return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
    }),
  );
  const a = m.decodeArticulation(h, ...buffers);
  a.breeze = m.prepareBreeze(a, "windy");
  a.tier = "B";
  const p = m.articulationPixels(h.nodes.length),
    l = m.articulationPixels(h.leafCount);
  for (const i of [0, 3]) {
    const mean = fixture.hourly.wind_speed_10m[i],
      gust = fixture.hourly.wind_gusts_10m[i],
      direction = fixture.hourly.wind_direction_10m[i];
    a.weather = m.prepareWeatherWind({
      meanMps: mean,
      gustMps: gust,
      directionDeg: direction,
      latitude: 44.944403,
      longitude: -93.425903,
    });
    const samples = [];
    for (let k = 0; k < 250; k++) {
      const before = performance.now();
      m.evaluateArticulation(a, k / 24, 1, 90, p, l);
      const ms = performance.now() - before;
      if (k >= 50) samples.push(ms);
    }
    const s = [...samples].sort((a, b) => a - b);
    summary.push({
      scene,
      hour: fixture.labels[i],
      nodes: h.nodes.length,
      leafPacketsEvaluated: 0,
      medianMs: s[100],
      p95Ms: s[190],
      samples,
      poseUploadBytes: p.byteLength,
      allocatedLeafTextureBytes: l.byteLength,
    });
  }
}
await writeFile(`${out}/cpu-bench.json`, JSON.stringify(summary, null, 2));
console.info(summary.map(({ samples: _samples, ...s }) => s));
