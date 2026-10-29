// Crown-top probe: centimetres, actual hierarchical poses, no renderer. Run via heavy.
import { build } from "vite";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { homedir } from "node:os";
import { pathToFileURL } from "node:url";
const root = resolve(import.meta.dirname, "../../../.."),
  cache = resolve(homedir(), ".cache/hexapod-living-weather");
const out = process.env.LAB_OUT ?? `${cache}/artifacts`;
await mkdir(out, { recursive: true });
const entry = `${cache}/amplitude-entry.ts`;
await writeFile(
  entry,
  `export * from ${JSON.stringify(resolve(root, "apps/web/src/cesium/scanView/livingArticulation.ts"))};\nexport * from ${JSON.stringify(resolve(root, "packages/world/src/weatherWind.ts"))};\nexport * from ${JSON.stringify(resolve(root, "packages/world/src/weatherCalibration.ts"))};`,
);
await build({
  configFile: false,
  resolve: { alias: { "@twin/world": resolve(root, "packages/world/src/index.ts") } },
  ssr: { noExternal: true },
  build: {
    ssr: entry,
    target: "node22",
    outDir: `${cache}/amplitude`,
    emptyOutDir: true,
    minify: false,
    rolldownOptions: { output: { entryFileNames: "model.mjs" } },
  },
});
const m = await import(pathToFileURL(`${cache}/amplitude/model.mjs`));
const forest = process.argv[3] === "forest";
const folder = forest ? `${cache}/forest` : resolve(homedir(), ".cache/hexapod-living-breeze/bake");
const h = JSON.parse(await readFile(`${folder}/articulation.json`, "utf8"));
const buffers = await Promise.all(
  [h.motionBinary, h.leafBinary, h.bindingBinary].map(async (name) => {
    const b = await readFile(`${folder}/${name}`);
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  }),
);
const a = m.decodeArticulation(h, ...buffers);
if (forest) {
  a.header = { ...h, nodes: h.nodes.slice(0, 12), leafCount: 0, tiles: {} };
  a.coefficients = a.coefficients.slice(0, 72);
  a.bindings = new Uint32Array(0);
}
a.breeze = m.prepareBreeze(a, "windy");
a.tier = "B";
const top = a.header.nodes.reduce(
  (best, n, i) => (n.position[2] > a.header.nodes[best].position[2] ? i : best),
  0,
);
const point = [...a.header.nodes[top].position];
point[2] = 6.224609375 * (forest ? h.stand.copies[0].scale : 1);
const p = m.articulationPixels(h.nodes.length),
  l = m.articulationPixels(0);
const fixture = JSON.parse(
  await readFile(resolve(root, "apps/web/src/lib/weather/fixtures/minnetonka-hours.json"), "utf8"),
);
function measure(mean, gust, direction = 270) {
  a.weather = m.prepareWeatherWind({
    meanMps: mean,
    gustMps: gust,
    directionDeg: direction,
    latitude: 44.944403,
    longitude: -93.425903,
  });
  const displacements = [];
  let peak = 0;
  const along = [];
  for (let i = 0; i < 6000; i++) {
    m.evaluateArticulation(a, i / 10, 1, 90, p, l);
    const o = (top + 1) * 8;
    const q = Array.from(p.slice(o, o + 4));
    const v = m.rotatePoint(q, point).map((v, c) => v + (p[o + 4 + c] ?? 0) - point[c]);
    displacements.push(v);
    along.push(v[0] * a.weather.east + v[1] * a.weather.north);
    peak = Math.max(peak, Math.hypot(v[0], v[1]));
  }
  const avg = along.reduce((a, b) => a + b, 0) / along.length;
  const sigma = Math.sqrt(along.reduce((a, b) => a + (b - avg) ** 2, 0) / along.length);
  return {
    meanMps: mean,
    gustMps: gust,
    sigmaCm: 100 * sigma,
    meanAlongCm: 100 * avg,
    peakHorizontalCm: 100 * peak,
  };
}
const hours = fixture.labels.map((label, i) => ({
  label,
  targetSigmaCm:
    100 * m.weatherGustSigma(fixture.hourly.wind_speed_10m[i], fixture.hourly.wind_gusts_10m[i]),
  ...measure(
    fixture.hourly.wind_speed_10m[i],
    fixture.hourly.wind_gusts_10m[i],
    fixture.hourly.wind_direction_10m[i],
  ),
}));
const reference = [6, 9.17, 12, 15].map((u) => ({
  ...measure(u, u * 1.216),
  targetSigmaCm: 100 * 0.000997 * u * u,
}));
// Check the actual field records too. Their inverse-fit residual reaches 58.47%:
// a 60% raw-record envelope follows observed scatter, not machine arithmetic.
const fit = m.fitMagnoliaSway();
const sourcePoints = m.MAGNOLIA_SWAY.map(([u, sigmaPixels]) => ({
  ...measure(u, u * 1.216),
  measuredSigmaCm: sigmaPixels * fit.metresPerPixel * 100,
  fittedSigmaCm: fit.sigmaMetresPerMpsSquared * u * u * 100,
}));
const sourceScatterMaximum = Math.max(
  ...sourcePoints.map((p) => Math.abs(p.fittedSigmaCm / p.measuredSigmaCm - 1)),
);
const result = {
  scene: forest ? "forest-exposed-copy" : "tree",
  topNode: top,
  probeM: point,
  seconds: 600,
  samples: 6000,
  hours,
  reference,
  sourcePoints,
  sourceScatterMaximum,
};
await writeFile(
  `${out}/amplitudes-${process.argv[2] ?? "current"}${forest ? "-forest" : ""}.json`,
  JSON.stringify(result, null, 2),
);
console.info(result);
if (process.argv[2] === "calibrated") {
  // 20% transfer tolerance stated before the final fit: source height/projection,
  // nonlinear hierarchy and bounded extrapolation; not an arithmetic tolerance.
  for (const r of reference) {
    const error = Math.abs(r.sigmaCm / r.targetSigmaCm - 1);
    if (error > 0.2) throw new Error(`Crown calibration outside 20% at ${r.meanMps}: ${error}`);
  }
  for (const r of hours) {
    const error = Math.abs(r.sigmaCm / r.targetSigmaCm - 1);
    if (error > 0.2)
      throw new Error(`Hour outside 20% gust-adjusted field curve at ${r.label}: ${error}`);
  }
}

if (process.argv[2] === "calibrated") {
  for (const r of sourcePoints) {
    const error = Math.abs(r.sigmaCm / r.measuredSigmaCm - 1);
    if (error > 0.6)
      throw new Error(`Outside measured 60% field-scatter envelope at ${r.meanMps}: ${error}`);
  }
}
