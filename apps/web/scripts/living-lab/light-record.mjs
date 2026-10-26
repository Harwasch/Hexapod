// Deterministic four-arm recording + interleaved completed-frame timings.
// Requires driver.mjs, light-cache.mjs and Vite on distinct 5600–5699 ports.
import { request } from "node:http";
import { existsSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
const out = resolve(process.env.LIGHT_OUTPUT);
const driver = process.env.LIGHT_DRIVER ?? "http://localhost:5601";
const cache = process.env.LIGHT_CACHE_URL ?? "http://localhost:5602";
// Node fetch imposes a five-minute headers timeout. Software-rendered screenshots
// can be paused by a shared host; use HTTP without that unrelated transport deadline.
const call = (route, body) =>
  new Promise((resolve, reject) => {
    const req = request(driver + route, { method: "POST" }, (response) => {
      let value = "";
      response.on("data", (chunk) => {
        value += chunk;
      });
      response.on("end", () => {
        if (response.statusCode !== 200) {
          reject(new Error(value));
          return;
        }
        try {
          resolve(JSON.parse(value));
        } catch (error) {
          reject(error);
        }
      });
    });
    req.on("error", reject);
    req.end(body);
  });
const evaluate = (code) => call("/eval", code);
const shot = (name) => call(`/shot?path=${encodeURIComponent(resolve(out, name))}`);
const frames = async (n) =>
  evaluate(`new Promise(resolve => {
  const start=__lab.status().frames;
  const tick=()=> __lab.status().frames >= start+${n} ? resolve(true) : requestAnimationFrame(tick);
  requestAnimationFrame(tick);
})`);
const set = async (arm, day = 0.5, seconds = 0, diffuse = false) => {
  await evaluate(`(() => {
    __lab.captured.getState().set(${arm === "captured"});
    __lab.light.getState().set({arm:${JSON.stringify(arm === "a" || arm === "b" ? arm : "off")},day:${day},seconds:${seconds},diffuse:${diffuse}});
    __lab.today(${day},${arm === "today"});
    return true;
  })()`);
  await frames(3);
};
await mkdir(out, { recursive: true });
await call("/fresh");
await call(
  "/goto",
  `/scripts/living-lab/harness.html?living=1&tiles=${cache}/scan/tileset.json&lightManifest=${cache}/sidecars/manifest.json`,
);
await evaluate(`new Promise(resolve => {
  const tick=()=>window.__lab && __lab.status().gaussians===978455 && __lab.status().loading===0 ? resolve(true) : setTimeout(tick,100);
  tick();
})`);
await evaluate(
  `(() => { __lab.view([0,0,0.1],135,-30,5); LL.getState().set({bench:true,clock:0}); return true; })()`,
);
await frames(6);
await set("captured");
await shot("captured-before.png");
const timings = [];
const measure = async (arm, round) => {
  await set(arm);
  await evaluate("(() => {window.__livingBench=[];return true;})()");
  await frames(12);
  const ms = await evaluate("window.__livingBench.slice(0,12)");
  timings.push({ arm, round, ms });
  await writeFile(resolve(out, "bench-partial.json"), JSON.stringify(timings, null, 2));
};
await measure("captured", "before-streams");
await set("b");
await evaluate(`new Promise(resolve => {
  const tick=()=>__lab.light.getState().status.includes("loaded") ? resolve(true) : setTimeout(tick,100);
  tick();
})`);
await frames(8);
await set("captured");
await shot("captured-after.png");
for (const arm of ["captured", "today", "a", "b"]) {
  await mkdir(resolve(out, arm), { recursive: true });
  for (let i = 0; i < 24; i++) {
    if (existsSync(resolve(out, arm, `${String(i).padStart(3, "0")}.png`))) continue;
    await set(arm, i / 23, i * 0.5);
    await shot(`${arm}/${String(i).padStart(3, "0")}.png`);
  }
  console.info("recorded", arm);
}
await set("a", 0.5, 0, true);
await shot("diffuse-a.png");
await set("b", 0.5, 0, true);
await shot("diffuse-b.png");
await set("today", 0.15);
await shot("double-shadow-today.png");
await set("b", 0.15);
await shot("double-shadow-b.png");
// Fresh batches in paired, reversed order to reduce slow thermal/load drift bias.
for (let round = 0; round < 3; round++)
  for (const arm of ["today", "a", "b"]) {
    for (const candidate of round % 2 ? [arm, "captured"] : ["captured", arm])
      await measure(candidate, round);
  }
await set("b", 0.9);
await set("a", 0.1);
await set("captured");
await shot("captured-after-toggles.png");
const status = await evaluate("__lab.status()");
const renderer = await evaluate(`(() => {
  const canvas=[...document.querySelectorAll("canvas")].at(-1);
  const gl=canvas.getContext("webgl2"); const ext=gl.getExtension("WEBGL_debug_renderer_info");
  return {renderer:ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):"unknown",userAgent:navigator.userAgent};
})()`);
const logs = await (await fetch(driver + "/logs")).json();
await writeFile(
  resolve(out, "bench.json"),
  JSON.stringify({ timings, status, renderer, logs, width: 720, height: 600 }, null, 2),
);
console.info("recordings and bench complete", out);
