// One browser, fixed clocks, paired static controls. Start the lab driver on :5742 first.
import { mkdir, readFile, writeFile, appendFile } from "node:fs/promises";
import { resolve } from "node:path";
import { request } from "node:http";

const root = process.cwd();
const scene = process.argv[2] ?? "spool";
if (!["tree", "spool"].includes(scene)) throw new Error("Expected tree or spool");
const preview = process.argv.includes("--preview");
const resume = process.argv.includes("--resume");
const out = resolve(process.env.LAB_OUT ?? ".lab-out/round3");
const driver = process.env.DRIVER_URL ?? "http://localhost:5742";
const bake = `${root}/.lab-out/round3/${scene}-bake`;
const source = scene === "tree" ? `${root}/.lab-data/minnetonka-tree/splat` : `${bake}/scene/splat`;
await mkdir(out, { recursive: true });
const manifest = JSON.parse(await readFile(`${source}/tileset.json`, "utf8"));
manifest.root.extras ??= {};
manifest.root.extras.nativeLod = false;
await writeFile(`${source}/lab-tileset.json`, JSON.stringify(manifest));
const call = (route, body = "") =>
  new Promise((resolve, reject) => {
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
const ev = (text) => call("eval", text);
const shot = (name) => call(`canvas?path=${encodeURIComponent(`${out}/${name}.png`)}`);
const settle = (patch) =>
  ev(`(async()=>{LL.getState().set(${JSON.stringify(patch)}); const n=__lab.status().frames;
  while(__lab.status().frames<n+2) await new Promise(r=>setTimeout(r,100)); return __lab.status();})()`);
const exists = async (path) => {
  try {
    return await readFile(path);
  } catch (error) {
    if (error.code === "ENOENT") return null;
    throw error;
  }
};
await call("fresh");
await call(
  "goto",
  `/scripts/living-lab/harness.html?living=1&tiles=${encodeURIComponent(`/@fs${source}/lab-tileset.json`)}`,
);
await ev(`(async()=>{while(!window.__lab) await new Promise(r=>setTimeout(r,500));
  ${scene === "tree" ? "__lab.viewBounds(35,-12,15.4)" : "__lab.view([1.1,.7,.42],35,-18,1.4)"};
  LL.getState().set({motion:'off',clock:0,bench:true,flatGround:${scene === "tree" ? "0" : ".29"}});
  while(__lab.status().loading || __lab.status().gaussians<${scene === "tree" ? "1500000" : "30000"} || !__lab.status().settled)
    await new Promise(r=>setTimeout(r,500)); return __lab.status();})()`);
console.info(scene, "loaded", await ev("__lab.status()"));
await ev(
  `LL.getState().set({motion:'articulated',articulationUrl:${JSON.stringify(`/@fs${bake}/articulation.json`)}})`,
);
await ev(`(async()=>{while(!LL.getState().articulationStatus.startsWith('Simulated')) {
  if(LL.getState().articulationStatus.startsWith('Error')) throw new Error(LL.getState().articulationStatus);
  await new Promise(r=>setTimeout(r,500)); } return LL.getState().articulationStatus;})()`);
if (scene === "tree") {
  await ev(
    `LL.getState().set({motion:'spectra',spectraUrl:${JSON.stringify(`/@fs${root}/.lab-out/living-spectra/bake/spectra.json`)}})`,
  );
  await ev(
    `(async()=>{while(!LL.getState().spectraStatus.startsWith('Simulated')) await new Promise(r=>setTimeout(r,500)); return true;})()`,
  );
}
const viewport = await ev("[innerWidth,innerHeight]");
const priorTiming = resume ? await exists(`${out}/bench-raw.jsonl`) : null;
const completedBlocks = priorTiming
  ? priorTiming
      .toString()
      .trim()
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line))
      .filter((row) => row.scene === scene)
  : [];
let blockIndex = 0;
for (const [name, patch] of [
  ["static", { motion: "off", strength: 1 }],
  ["calm", { motion: "articulated", strength: 0 }],
]) {
  await settle(patch);
  await shot(`${scene}-${name}`);
}
const arms = scene === "tree" ? ["spectra", "articulated"] : ["gust", "articulated"];
const views = scene === "tree" ? ["tree", "crown"] : ["spool"];
for (const view of views) {
  if (view === "crown") await ev("__lab.view([0,0,4.6],35,-12,7.8)");
  for (const arm of arms) {
    await mkdir(`${out}/${view}-${arm}-frames`, { recursive: true });
    for (const i of preview ? [0, 8, 16, 23] : Array.from({ length: 24 }, (_, i) => i)) {
      const frameName = `${view}-${arm}-frames/${String(i).padStart(3, "0")}`;
      const previous = resume ? await exists(`${out}/${frameName}.png`) : null;
      if (previous) {
        if (previous.readUInt32BE(16) !== viewport[0] || previous.readUInt32BE(20) !== viewport[1])
          throw new Error(`Resume viewport differs for ${frameName}`);
        console.info(`Resume: ${frameName}`);
        continue;
      }
      await settle({ motion: arm, strength: 1, windBearingDeg: 60, clock: i / 12 });
      await shot(frameName);
      console.info(`${view} ${arm} ${i + 1}/24`);
    }
  }
}
if (!preview) {
  if (scene === "tree") await ev("__lab.viewBounds(35,-12,15.4)");
  for (let round = 0; round < 3; round++)
    for (const arm of ["off", arms[0], "off", "articulated"]) {
      const prior = completedBlocks[blockIndex++];
      if (prior) {
        if (prior.round !== round || prior.arm !== arm)
          throw new Error("Invalid benchmark resume order");
        continue;
      }
      const samples =
        await ev(`(async()=>{LL.getState().set({motion:${JSON.stringify(arm)},strength:1,clock:null,bench:true});
      window.__livingBench=[]; while(__livingBench.length<12) await new Promise(r=>setTimeout(r,100));
      return __livingBench.slice(4,12);})()`);
      const row = {
        scene,
        viewport,
        round,
        arm,
        samples,
        medianMs: [...samples].sort((a, b) => a - b)[4],
        status: await ev("__lab.status()"),
      };
      await appendFile(`${out}/bench-raw.jsonl`, JSON.stringify(row) + "\n");
      console.info(row);
    }
}
await ev('LL.getState().set({motion:"off",bench:false})');
await writeFile(`${out}/${scene}-console.json`, JSON.stringify(await call("logs"), null, 2));
await writeFile(
  `${out}/${scene}-status.json`,
  JSON.stringify(await ev("({status:__lab.status(),lab:LL.getState()})"), null, 2),
);
console.info("Complete", scene, preview ? "preview" : "recording and benchmark");
