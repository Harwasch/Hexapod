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
  const r = spawnSync(
    ffmpeg,
    [
      "-hide_banner",
      "-loglevel",
      "error",
      "-y",
      "-filter_threads",
      "4",
      "-filter_complex_threads",
      "4",
      ...args,
    ],
    {
      stdio: "inherit",
    },
  );
  if (r.status !== 0) throw new Error(`ffmpeg failed (${r.status})`);
};
const fixture = JSON.parse(
  await readFile(
    new URL("../../src/lib/weather/fixtures/minnetonka-hours.json", import.meta.url),
    "utf8",
  ),
);
const names = fixture.labels;
for (const scene of ["forest-oblique", "forest-edge", "crown"]) {
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
// One overhead storm clip, with the 10 m inset.
run([
  "-framerate",
  "24",
  "-i",
  `${out}/forest-overhead-storm-frames/%03d.png`,
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
  `${out}/forest-overhead-storm.webm`,
]);
const scenes = ["forest-oblique", "forest-edge", "crown"];
const inputs = scenes.flatMap((scene) =>
  names.flatMap((n) => ["-i", `${out}/${scene}-${n}-frames/096.png`]),
);
const scaled = Array.from({ length: 12 }, (_, i) => `[${i}:v]scale=512:320[s${i}]`).join(";");
const rows = [0, 1, 2]
  .map(
    (r) =>
      Array.from({ length: 4 }, (_, i) => `[s${r * 4 + i}]`).join("") + `hstack=inputs=4[r${r}]`,
  )
  .join(";");
run([
  ...inputs,
  "-filter_complex",
  `${scaled};${rows};[r0][r1][r2]vstack=inputs=3[out]`,
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
const titles = {
  "forest-oblique": "Dense grove · 38 m oblique",
  "forest-edge": "Dense grove · 2 m eye height",
  crown: "Calibrated crown · round-4 camera",
};
const gallery = `<!doctype html><html lang="en"><meta charset="utf-8"><title>Living weather look · Simulated</title><style>body{margin:32px auto;max-width:1440px;padding:0 20px;background:#12212d;color:#eaf2f6;font:16px/1.5 system-ui}video,img{max-width:100%;display:block}figure{margin:24px 0}a{color:#8bd5ff}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}figcaption{margin:8px 0}</style><h1>Weather-driven wind · one more look</h1><p>Simulated. 80 crown-spaced copies, 2.4 million drawn splats. Crown diameter 3.89 m; nominal spacing 3.50 m. Per-copy yaw, ±10% scale, reflection and ±5% brightness. Canopy shelter is a cheap approximation. Sway is constrained by published magnolia field data, with uncertain species/architecture transfer.</p><p>Every clip: 8 s, 24 fps, 192 fixed-clock frames at i/24, 1024×640 per panel. No temporal interpolation. All four real archive hours use the same leaf-on geometry.</p><p><a href="results.md">Measurements, calibration, controls, gates and honest visual read</a></p><figure><a href="contact-sheet.png"><img src="contact-sheet.png" alt="Four weather hours: oblique grove, low grove and crown"></a><figcaption>t=4 s. Columns calm, light, fresh, storm; rows oblique grove, low grove, crown.</figcaption></figure>${scenes.map((scene) => `<h2>${titles[scene]}</h2><figure><video controls preload="metadata" src="${scene}-four-hours.webm"></video><figcaption>Four-panel comparison: ${names.map((n, i) => `${n}: ${weather(i)}`).join("; ")}.</figcaption></figure><div class="grid">${names.map((n, i) => `<figure><video controls preload="metadata" poster="${scene}-${n}-frames/096.png" src="${scene}-${n}.webm"></video><figcaption>${n} · ${weather(i)}</figcaption></figure>`).join("")}</div>`).join("")}<h2>Overhead storm · shared wind field</h2><video controls preload="metadata" poster="forest-overhead-storm-frames/096.png" src="forest-overhead-storm.webm"></video><p>${weather(3)}. North-up inset spans 45 m and shows the shared 10 m field. Trees additionally receive canopy-height and stand-shelter reductions.</p><p>Scan: Matthew Guertin / Matt1Up, tree-photogrammetry-dataset, CC BY 4.0. Weather: Open-Meteo reanalysis, CC BY 4.0. Calibration: <a href="https://doi.org/10.1017/flo.2021.15">Cardona &amp; Dabiri 2022</a>. Shelter coefficient: <a href="https://doi.org/10.1016/j.agrformet.2013.11.013">Kim et al. 2014</a> (assumed there too).</p></html>`;
await writeFile(`${out}/index.html`, gallery);
