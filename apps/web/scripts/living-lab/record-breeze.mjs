// Run after serve-bench and driver. Fixed-clock frames; one browser, serial arms.
import { mkdir, writeFile, appendFile, stat } from "node:fs/promises";
import { resolve } from "node:path";
const out =
  process.env.LAB_OUT ??
  resolve(import.meta.dirname, "../../../../.claude/ops/notes/living-breeze");
const driver = process.env.DRIVER_URL ?? "http://127.0.0.1:5711";
async function call(route, body = "") {
  const r = await fetch(`${driver}/${route}`, { method: "POST", body });
  const t = await r.text();
  if (!r.ok) throw new Error(t);
  return JSON.parse(t);
}
const ev = (s) => call("eval", s);
await mkdir(out, { recursive: true });
await ev(
  `(()=>{document.querySelector('main').style.display='none';const v=document.getElementById('viewer');v.style.width='512px';v.style.height='320px';document.querySelector('.cesium-widget').style.height='100%';__lab.widget.resolutionScale=1;LL.getState().set({clock:0,bench:true});return true})()`,
);
const settle = async (patch) =>
  ev(
    `(async()=>{LL.getState().set(${JSON.stringify(patch)});const n=__lab.status().frames;while(__lab.status().frames<n+3 || __lab.status().loading || !__lab.status().settled || __lab.status().gaussians!==687226)await new Promise(r=>setTimeout(r,50));return __lab.status()})()`,
  );
const configure = async (arm) => {
  await settle({
    motion: arm === "off" ? "off" : "articulated",
    articulationUrl:
      arm === "round3"
        ? "/data/round3/articulation.json"
        : `/data/bake/articulation.json?breeze=${arm}`,
    strength: 1,
    windBearingDeg: 60,
  });
  if (arm !== "off")
    await ev(
      `(async()=>{while(!LL.getState().articulationStatus.startsWith('Simulated')){if(LL.getState().articulationStatus.startsWith('Error'))throw new Error(LL.getState().articulationStatus);await new Promise(r=>setTimeout(r,100))}return LL.getState().articulationStatus})()`,
    );
};
async function shot(path) {
  const png = await ev(
    `(()=>{const c=document.createElement('canvas');c.width=512;c.height=320;const x=c.getContext('2d');x.fillStyle='#9fb3c8';x.fillRect(0,0,512,320);for(const source of document.querySelectorAll('#viewer canvas')){const gl=source.getContext('webgl2');if(gl?.getContextAttributes()?.preserveDrawingBuffer)x.drawImage(source,0,0,512,320)}return c.toDataURL('image/png').split(',')[1]})()`,
  );
  await writeFile(path, Buffer.from(png, "base64"));
}
// Same framing and clock for all arms; four seconds, 24 fps, no interpolation.
for (const view of ["tree", "crown"]) {
  await ev(view === "tree" ? "__lab.viewBounds(35,-12,15.4)" : "__lab.view([0,0,4.6],35,-12,7.8)");
  for (const arm of ["round3", "calm", "breezy", "windy"]) {
    await configure(arm);
    await mkdir(`${out}/${view}-${arm}`, { recursive: true });
    for (let i = 0; i < 96; i++) {
      if (
        process.argv.includes("--resume") &&
        (await stat(`${out}/${view}-${arm}/${String(i).padStart(3, "0")}.png`).then(
          () => true,
          () => false,
        ))
      )
        continue;
      await settle({ clock: i / 24 });
      await shot(`${out}/${view}-${arm}/${String(i).padStart(3, "0")}.png`);
      if (i % 24 === 0) console.info(view, arm, i);
    }
  }
}
await ev("__lab.viewBounds(35,-12,15.4)");
await configure("off");
await settle({ clock: 0 });
await shot(`${out}/pixel-off.png`);
await configure("calm");
await settle({ strength: 0 });
await shot(`${out}/pixel-zero.png`);
await settle({ strength: 1 });
await ev("captured.getState().set(true)");
await settle({ clock: 1 });
await shot(`${out}/pixel-captured.png`);
await ev("captured.getState().set(false)");
for (let round = 0; round < 3; round++)
  for (const arm of ["off", "round3", "off", "calm", "off", "breezy", "off", "windy"]) {
    await configure(arm);
    const samples = await ev(
      `(async()=>{LL.getState().set({clock:null,bench:true,strength:1});window.__livingBench=[];while(__livingBench.length<12)await new Promise(r=>setTimeout(r,100));return __livingBench.slice(4,12)})()`,
    );
    const row = { round, arm, samples, status: await ev("__lab.status()") };
    await appendFile(`${out}/bench-512-raw.jsonl`, JSON.stringify(row) + "\n");
    console.info("bench", round, arm);
  }
await settle({ motion: "off", bench: true });
await ev("LL.getState().set({bench:false})");
await writeFile(`${out}/console.json`, JSON.stringify(await call("logs"), null, 2));
console.info("Recordings and interleaved benchmark complete");
