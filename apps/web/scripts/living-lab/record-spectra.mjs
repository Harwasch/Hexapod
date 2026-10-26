// Reproducible CPU/SwiftShader comparison. Start vite :5741 and driver :5742 first.
// Run from repo root: node apps/web/scripts/living-lab/record-spectra.mjs [playcanvas|survey]
// Data and output stay untracked. No API writes or external browser requests are needed.
import { mkdir, readFile, writeFile, appendFile, access } from "node:fs/promises";
import { resolve } from "node:path";
import { request } from "node:http";

const root = process.cwd();
const out = resolve(process.env.LAB_OUT ?? resolve(root, ".lab-out/living-spectra"));
const data = resolve(root, ".lab-data/minnetonka-tree");
const driver = process.env.DRIVER_URL ?? "http://localhost:5742";
await mkdir(out, { recursive: true });
const manifest = JSON.parse(await readFile(`${data}/splat/tileset.json`, "utf8"));
manifest.root.extras.nativeLod = false; // mirror contains GLBs, not SOG; preserve every tile
await writeFile(`${data}/splat/lab-tileset.json`, JSON.stringify(manifest));
const tiles = `/@fs${data}/splat/lab-tileset.json`;
const spectra = `/@fs${root}/.lab-out/living-spectra/bake/spectra.json`;
function call(route, body = "") {
  // Rendering may take many minutes under SwiftShader: no five-minute fetch header timeout.
  return new Promise((resolve, reject) => {
    const req = request(`${driver}/${route}`, { method: "POST" }, (response) => {
      let text = "";
      response.setEncoding("utf8");
      response.on("data", (chunk) => {
        text += chunk;
      });
      response.on("end", () => {
        if (response.statusCode !== 200) reject(new Error(text));
        else {
          try {
            resolve(JSON.parse(text));
          } catch (error) {
            reject(error);
          }
        }
      });
    });
    req.on("error", reject);
    req.end(body);
  });
}
const ev = (expression) => call("eval", expression);
const shot = (name) => call(`canvas?path=${encodeURIComponent(`${out}/${name}`)}`);
const median = (a) => [...a].sort((a, b) => a - b)[Math.floor(a.length / 2)];
async function benchLine(arm, round, samples, status) {
  const line = JSON.stringify({ arm, round, medianMs: median(samples), samples, status });
  await appendFile(`${out}/bench-raw.jsonl`, `${line}\n`);
  console.log(line);
}
async function captured(name) {
  if (process.env.RESUME !== "1") return false;
  try {
    await access(`${out}/${name}`);
    return true;
  } catch {
    return false;
  }
}
const survey = process.argv[2] === "survey";
await call("fresh");
await call(
  "goto",
  `/scripts/living-lab/${survey ? "survey" : "harness"}.html?living=1&tiles=${encodeURIComponent(tiles)}&rig=${encodeURIComponent(`/@fs${data}/source/rig.json`)}`,
);
if (!survey) {
  await ev(`(async()=>{while(!window.__lab) await new Promise(r=>setTimeout(r,500));
    __lab.viewBounds(35,-12,15.4); LL.getState().set({motion:"off",clock:0,bench:true});
    while(__lab.status().loading || __lab.status().gaussians<1500000) await new Promise(r=>setTimeout(r,1000));
    return __lab.status();})()`);
  console.log("PlayCanvas loaded", await ev(`__lab.status()`));
  // Warm up spectra and both shader configurations before recording/timing.
  await ev(`(async()=>{LL.getState().set({motion:"spectra",spectraUrl:${JSON.stringify(spectra)},clock:0});
    while(!LL.getState().spectraStatus.startsWith("Simulated")) await new Promise(r=>setTimeout(r,500));
    return LL.getState().spectraStatus;})()`);
  for (const [name, patch] of [
    ["tree-static", { motion: "off", strength: 1 }],
    ["tree-calm", { motion: "spectra", strength: 0 }],
  ]) {
    await ev(`(async()=>{LL.getState().set(${JSON.stringify(patch)});
      const n=__lab.status().frames; while(__lab.status().frames<n+2) await new Promise(r=>setTimeout(r,100)); return true;})()`);
    await shot(`${name}.png`);
  }
  for (const view of ["tree", "crown"]) {
    if (view === "crown") await ev(`__lab.view([0,0,4.6],35,-12,7.8)`);
    for (const arm of ["gust", "spectra"]) {
      await mkdir(`${out}/${view}-${arm}-frames`, { recursive: true });
      for (let i = 0; i < 24; i++) {
        if (await captured(`${view}-${arm}-frames/${String(i).padStart(3, "0")}.png`)) continue;
        await ev(`(async()=>{LL.getState().set({motion:${JSON.stringify(arm)},strength:1,windBearingDeg:60,clock:${i / 12}});
          const n=__lab.status().frames; while(__lab.status().frames<n+2) await new Promise(r=>setTimeout(r,100)); return __lab.status().frames;})()`);
        await shot(`${view}-${arm}-frames/${String(i).padStart(3, "0")}.png`);
        console.log(`${view} ${arm} ${i + 1}/24`);
      }
    }
  }
  await ev(`__lab.viewBounds(35,-12,15.4)`);
  // Interleaved controls; animated clocks include the spectral sum every measured frame.
  for (let round = 0; round < 3; round++) {
    for (const arm of ["off", "gust", "off", "spectra"]) {
      const samples =
        await ev(`(async()=>{LL.getState().set({motion:${JSON.stringify(arm)},clock:null,bench:true});
        window.__livingBench=[]; while(__livingBench.length<12) await new Promise(r=>setTimeout(r,100));
        return __livingBench.slice(4,12);})()`);
      await benchLine(arm, round, samples, await ev(`__lab.status()`));
    }
  }
  await ev(`LL.getState().set({motion:"off",bench:false})`);
} else {
  const wind = `{strength:0.1,bearingDeg:60}`;
  await ev(`(async()=>{while(!window.__survey) await new Promise(r=>setTimeout(r,500));
    await __survey.waitUntilReady(600000); const ready = await __survey.view(15.4,0,${wind},600000);
    __survey.manualFrames(); return ready;})()`);
  for (const view of ["tree", "crown"]) {
    if (view === "crown") await ev(`__survey.viewAt([0,0,4.6],35,-12,7.8)`);
    await mkdir(`${out}/${view}-today-frames`, { recursive: true });
    for (let i = 0; i < 24; i++) {
      await ev(`__survey.step(${i / 12},${wind})`);
      if (await captured(`${view}-today-frames/${String(i).padStart(3, "0")}.png`)) continue;
      await shot(`${view}-today-frames/${String(i).padStart(3, "0")}.png`);
      console.log(`${view} today ${i + 1}/24`);
    }
  }
  await ev(`__survey.view(15.4,0,${wind},600000)`);
  for (let round = 0; round < 3; round++) {
    for (const strength of [0, 0.1]) {
      const samples = await ev(`(async()=>{const samples=[];
        for(let i=0;i<12;i++) samples.push(await __survey.benchFrame(${round * 3}+i/12,{strength:${strength},bearingDeg:60}));
        return samples.slice(4);})()`);
      await benchLine(
        strength ? "today" : "today-off",
        round,
        samples,
        await ev(`__survey.status()`),
      );
    }
  }
}
await writeFile(
  `${out}/${survey ? "survey" : "playcanvas"}-console.json`,
  JSON.stringify(await call("logs"), null, 2),
);
console.log("Recording and bench complete");
