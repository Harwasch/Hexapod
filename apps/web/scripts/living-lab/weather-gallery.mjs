// Encode only recorded fixed-clock PNGs. No generated/interpolated frames.
import { spawnSync } from "node:child_process";
import { readFile, writeFile, stat } from "node:fs/promises";
import { resolve } from "node:path";
import { homedir } from "node:os";
const cache = resolve(homedir(), ".cache/hexapod-living-weather"),
  out = resolve(process.env.LAB_OUT ?? `${cache}/artifacts`);
const ffmpeg =
  process.env.LIVING_FFMPEG ??
  resolve(homedir(), ".cache/hexapod-living-breeze/ffmpeg/ffmpeg-7.0.2-amd64-static/ffmpeg");
const run = (args) => {
  const r = spawnSync(ffmpeg, ["-hide_banner", "-loglevel", "error", "-y", ...args], {
    stdio: "inherit",
  });
  if (r.status !== 0) throw new Error(`ffmpeg failed (${r.status})`);
};
const fixture = JSON.parse(
  await readFile(
    new URL("../../src/lib/weather/fixtures/minnetonka-hours.json", import.meta.url),
    "utf8",
  ),
);
const names = fixture.labels;
for (const scene of ["forest", "crown"]) {
  for (const name of names) {
    const folder = `${out}/${scene}-${name}-frames`;
    for (let i = 0; i < 192; i++) await stat(`${folder}/${String(i).padStart(3, "0")}.png`);
    run([
      "-framerate",
      "24",
      "-i",
      `${folder}/%03d.png`,
      "-frames:v",
      "192",
      "-c:v",
      "libvpx-vp9",
      "-crf",
      "28",
      "-b:v",
      "0",
      "-threads",
      "4",
      "-row-mt",
      "1",
      "-pix_fmt",
      "yuv420p",
      `${out}/${scene}-${name}.webm`,
    ]);
  }
  run([
    ...names.flatMap((n) => ["-i", `${out}/${scene}-${n}.webm`]),
    "-filter_complex",
    "[0:v][1:v][2:v][3:v]hstack=inputs=4[out]",
    "-map",
    "[out]",
    "-frames:v",
    "192",
    "-c:v",
    "libvpx-vp9",
    "-crf",
    "28",
    "-b:v",
    "0",
    "-threads",
    "4",
    "-row-mt",
    "1",
    `${out}/${scene}-four-hours.webm`,
  ]);
}
const inputs = ["forest", "crown"].flatMap((scene) =>
  names.flatMap((n) => ["-i", `${out}/${scene}-${n}-frames/096.png`]),
);
run([
  ...inputs,
  "-filter_complex",
  "[0:v]scale=512:320[a];[1:v]scale=512:320[b];[2:v]scale=512:320[c];[3:v]scale=512:320[d];[4:v]scale=512:320[e];[5:v]scale=512:320[f];[6:v]scale=512:320[g];[7:v]scale=512:320[h];[a][b][c][d]hstack=inputs=4[top];[e][f][g][h]hstack=inputs=4[bottom];[top][bottom]vstack=inputs=2[out]",
  "-map",
  "[out]",
  "-frames:v",
  "1",
  "-threads",
  "4",
  `${out}/contact-sheet.png`,
]);
const weather = (i) =>
  `${fixture.hourly.time[i]}Z · mean ${fixture.hourly.wind_speed_10m[i]} / gust ${fixture.hourly.wind_gusts_10m[i]} m/s · from ${fixture.hourly.wind_direction_10m[i]}°`;
const gallery = `<!doctype html><html lang="en"><meta charset="utf-8"><title>Living weather wind · Simulated</title><style>body{margin:32px auto;max-width:1440px;padding:0 20px;background:#12212d;color:#eaf2f6;font:16px/1.5 system-ui}video,img{max-width:100%;display:block}figure{margin:24px 0}a{color:#8bd5ff}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}figcaption{margin:8px 0}h1{line-height:1.2}</style><h1>Simulated from weather at Minnetonka</h1><p>Four Open-Meteo archive hours drive the same captured crown and a 36-copy, 1.44-million-splat stand. Motion is simulated, not an observed replay. All clips are 8 s, 24 fps, 192 fixed-clock frames, 1024×640 per view. No interpolation. Winter wind uses the same leaf-on geometry.</p><p><a href="results.md">Results, timings, gates and limitations</a> · <a href="public-scan-detail.json">Every public scan's detail score</a> · <a href="cpu-bench.json">CPU samples</a></p><figure><a href="contact-sheet.png"><img src="contact-sheet.png" alt="Forest and crown at all four weather hours, left to right calm, light, fresh, storm"></a><figcaption>At t=4 s. Columns: calm, light, fresh, storm; rows: forest, crown. Open the image for full size.</figcaption></figure>${["forest", "crown"].map((scene) => `<h2>${scene === "forest" ? "Forest · shared field sweeping downwind" : "Crown · original round-4 camera"}</h2><figure><video controls preload="metadata" src="${scene}-four-hours.webm"></video><figcaption>Four hours side by side (4096×640). Left to right: ${names.join(", ")}.</figcaption></figure><div class="grid">${names.map((n, i) => `<figure><video controls preload="metadata" poster="${scene}-${n}-frames/096.png" src="${scene}-${n}.webm"></video><figcaption><strong>${n}</strong> · ${weather(i)}</figcaption></figure>`).join("")}</div>`).join("")}<p>Scan: Matthew Guertin / Matt1Up, <a href="https://github.com/Matt1Up/tree-photogrammetry-dataset">tree-photogrammetry-dataset</a>, <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Weather: <a href="https://open-meteo.com/">Open-Meteo</a>, reanalysis, <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Field scales, compliance and saturation remain visual priors. The heatmap shows 10 m wind; plant wind is reduced at canopy height.</p></html>`;
await writeFile(`${out}/index.html`, gallery);
