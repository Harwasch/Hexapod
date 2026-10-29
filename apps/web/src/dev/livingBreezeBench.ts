import { armOrder, summarizeWeatherBench, type WeatherBenchBlock } from "./livingWeatherBench";
import { prepareWeatherWind, sampleWeatherWind, type WeatherWind } from "@twin/world";
import { startLivingLabHarness } from "./livingLabHarness";
import { labWeatherClock } from "@/cesium/scanView/livingHook";
import { useAsCaptured } from "@/state/asCaptured";
import { setLabWeather } from "@/state/livingWeather";
import { parseWeather, weatherFeed } from "@/lib/weather/openMeteo";
import fixture from "@/lib/weather/fixtures/minnetonka-hours.json";

const el = (id: string) => {
  const e = document.getElementById(id);
  if (!e) throw new Error(id);
  return e;
};
const params = new URLSearchParams(location.search),
  forest = params.get("living-forest") === "1",
  spool = params.get("living-scan") === "spool";
const hours = parseWeather(fixture, 44.944403, -93.425903);
const names = ["calm", "light", "fresh", "storm"];
const run = el("run") as HTMLButtonElement,
  download = el("download") as HTMLButtonElement;
const lab = await startLivingLabHarness(
  el("viewer"),
  forest
    ? "/data/forest/tileset.json"
    : spool
      ? "/data/spool/tileset.json"
      : "/data/tree/bench-tileset.json",
  { forest },
);
Object.assign(window, { __lab: lab, LL: lab.lab, captured: useAsCaptured });
lab.widget.useBrowserRecommendedResolution = true;
lab.widget.resolutionScale = 1;
if (forest) {
  const view = params.get("view") ?? "oblique";
  if (view === "overhead") lab.view([0, 0, 2], 0, -89.9, 48);
  else if (view === "edge") lab.view([0, -7, 3.4], 10, 6, 13.4);
  else lab.view([0, 0, 3], 25, -25, 38);
} else if (spool) lab.viewBounds(35, -25, 7);
else lab.viewBounds(35, -12, 15.4);
const store = lab.lab,
  bake = forest
    ? "/data/forest/articulation.json"
    : spool
      ? "/data/spool/instances.json?tier-a=1"
      : "/data/bake/articulation.json";
const expected = forest ? 2400000 : spool ? 153566 : 687226;
const delay = () => new Promise<void>((r) => setTimeout(r, 75));
const firstHour = hours[0];
if (!firstHour) throw new Error("No recorded weather hours");
let wind: WeatherWind = firstHour,
  label = "",
  requestNumber = 0,
  applyNumber = 0;
const field = el("field") as HTMLCanvasElement;
field.style.display = forest && params.get("view") === "overhead" ? "block" : "none";
const context = field.getContext("2d");
if (!context) throw new Error("Canvas 2D unavailable");
const ctx = context;
function overlay() {
  const f = prepareWeatherWind(wind),
    t = store.getState().clock ?? labWeatherClock();
  ctx.fillStyle = "#142333";
  ctx.fillRect(0, 0, 224, 248);
  const active = !useAsCaptured.getState().on && store.getState().motion !== "off";
  for (let y = 0; y < 20; y++)
    for (let x = 0; x < 20; x++) {
      const v = active
        ? sampleWeatherWind(f, (x / 19 - 0.5) * 45, (0.5 - y / 19) * 45, t)
        : { alongMps: 0 };
      ctx.fillStyle = `hsl(${220 - 210 * Math.min(1, v.alongMps / 25)} 70% 45%)`;
      ctx.fillRect(12 + x * 10, 25 + y * 10, 10, 10);
    }
  ctx.fillStyle = "white";
  ctx.font = "11px system-ui";
  ctx.fillText("Simulated 10 m field · N ↑ · 45 m", 10, 15);
  ctx.fillText("0           10           20      25 m/s", 12, 240);
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 1.5;
  for (let y = 0; y < 4; y++)
    for (let x = 0; x < 4; x++) {
      const px = 38 + x * 50,
        py = 50 + y * 50;
      const v = sampleWeatherWind(f, ((px - 112) / 200) * 45, ((125 - py) / 200) * 45, t);
      const norm = Math.max(1, Math.hypot(v.eastMps, v.northMps));
      const dx = (v.eastMps / norm) * 12,
        dy = (-v.northMps / norm) * 12;
      ctx.beginPath();
      ctx.moveTo(px - dx, py - dy);
      ctx.lineTo(px + dx, py + dy);
      ctx.lineTo(px + dx - dx * 0.45 + dy * 0.35, py + dy - dy * 0.45 - dx * 0.35);
      ctx.moveTo(px + dx, py + dy);
      ctx.lineTo(px + dx - dx * 0.45 - dy * 0.35, py + dy - dy * 0.45 + dx * 0.35);
      ctx.stroke();
    }
  el("caption").textContent = active
    ? label
    : forest
      ? "Simulated forest stand · As captured tree geometry · motion off"
      : "As captured · motion off";
}
lab.widget.scene.postRender.addEventListener(overlay);
async function ready() {
  const deadline = performance.now() + 240000;
  while (lab.status().loading || !lab.status().settled || lab.status().gaussians !== expected) {
    if (performance.now() > deadline)
      throw new Error("Scene loading timed out: " + JSON.stringify(lab.status()));
    await delay();
  }
}
function controls() {
  for (const [id, value] of [
    ["mean", wind.meanMps],
    ["gust", wind.gustMps ?? wind.meanMps],
    ["direction", wind.directionDeg],
  ] as const) {
    (el(id) as HTMLInputElement).value = String(value);
    el(id + "-value").textContent = String(value);
  }
}
async function apply(next: WeatherWind | null, note: string) {
  const generation = ++applyNumber;
  if (next) wind = next;
  label = note;
  controls();
  store.getState().set({ bench: true, strength: 1, water: false });
  const before = lab.status().frames;
  setLabWeather(next, bake);
  while (lab.status().frames <= before) await delay();
  if (next && !useAsCaptured.getState().on) {
    const deadline = performance.now() + 180000;
    while (!store.getState().articulationStatus.startsWith("Simulated")) {
      if (performance.now() > deadline) throw new Error(store.getState().articulationStatus);
      await delay();
    }
  }
  await ready();
  if (generation === applyNumber) el("status").textContent = note;
}
async function preset(arm: string) {
  requestNumber++;
  (el("source") as HTMLSelectElement).value = "manual";
  if (arm === "off") return apply(null, "As captured · plain scan");
  const h = hours[names.indexOf(arm)];
  if (!h) throw new Error("Unknown weather hour");
  (el("date") as HTMLInputElement).value = h.time.slice(0, 16);
  return apply(
    h,
    `Simulated from weather at ${spool ? "Minnetonka weather applied to Spool (manual transfer)" : "Minnetonka"}, ${h.time} · mean ${h.meanMps} / gust ${h.gustMps} m/s · from ${h.directionDeg}° · Tier ${spool ? "off (rigid/ground ancestry)" : "B"}${forest ? " · 80 copies" : ""}`,
  );
}
Object.assign(window, { __weather: { hours, names, preset, apply, overlay, forest, expected } });
(el("captured") as HTMLInputElement).checked = useAsCaptured.getState().on;
el("captured").addEventListener("change", (e) =>
  useAsCaptured.getState().set((e.target as HTMLInputElement).checked),
);
el("preset").addEventListener("change", (e) => {
  void preset((e.target as HTMLSelectElement).value);
});
for (const id of ["mean", "gust", "direction"])
  el(id).addEventListener("change", () => {
    requestNumber++;
    (el("source") as HTMLSelectElement).value = "manual";
    const next = {
      ...wind,
      meanMps: Number((el("mean") as HTMLInputElement).value),
      gustMps: Number((el("gust") as HTMLInputElement).value),
      directionDeg: Number((el("direction") as HTMLInputElement).value),
    };
    next.gustMps = Math.max(next.gustMps, next.meanMps);
    void apply(
      next,
      `Simulated · manual mean ${next.meanMps} / gust ${next.gustMps} m/s · from ${next.directionDeg}°`,
    );
  });
async function live() {
  if ((el("source") as HTMLSelectElement).value !== "live") return;
  const generation = ++requestNumber,
    selectedDate = (el("date") as HTMLInputElement).value;
  const result = await weatherFeed(
    wind.latitude,
    wind.longitude,
    new Date((el("date") as HTMLInputElement).value + "Z"),
  );
  if (
    generation !== requestNumber ||
    (el("source") as HTMLSelectElement).value !== "live" ||
    (el("date") as HTMLInputElement).value !== selectedDate
  )
    return;
  await apply(result.wind, result.note);
}
el("source").addEventListener("change", () => {
  void live();
});
el("date").addEventListener("change", () => {
  void live();
});
interface BenchWindow {
  __livingBench?: number[];
}
const bw = window as unknown as BenchWindow;
let result: unknown;
run.onclick = () => {
  void (async () => {
    run.disabled = true;
    download.disabled = true;
    if (useAsCaptured.getState().on)
      throw new Error("Turn off As captured to measure simulated motion");
    const rounds = Number(params.get("rounds") ?? 8),
      samples = Number(params.get("samples") ?? 32);
    if (
      !Number.isInteger(rounds) ||
      rounds < 1 ||
      rounds > 20 ||
      !Number.isInteger(samples) ||
      samples < 2 ||
      samples > 240
    )
      throw new Error("Invalid benchmark length");
    const seed = crypto.getRandomValues(new Uint32Array(1))[0] ?? 0;
    const blocks: WeatherBenchBlock[] = [];
    for (let round = 0; round < rounds; round++)
      for (const arm of armOrder(round, seed)) {
        await preset(arm);
        el("status").textContent = `Round ${round + 1}/${rounds}: ${arm}`;
        store.getState().set({ clock: null });
        bw.__livingBench = [];
        while ((bw.__livingBench?.length ?? 0) < samples + 8) await delay();
        blocks.push({ round, arm, samples: (bw.__livingBench ?? []).slice(8, samples + 8) });
      }
    const statistics = summarizeWeatherBench(blocks);
    const { summary } = statistics;
    const canvas = Array.from(el("viewer").querySelectorAll("canvas")).filter((c) => c !== field),
      gl = canvas.map((c) => c.getContext("webgl2")).find(Boolean),
      debug = gl?.getExtension("WEBGL_debug_renderer_info");
    if (canvas.some((c) => c.width !== 1920 || c.height !== 1080))
      throw new Error("Drawing buffer changed: expected 1920 × 1080");
    result = {
      time: new Date().toISOString(),
      userAgent: navigator.userAgent,
      renderer: debug ? String(gl?.getParameter(debug.UNMASKED_RENDERER_WEBGL)) : "unavailable",
      buffers: canvas.map((c) => [c.width, c.height]),
      forest,
      status: lab.status(),
      hours,
      method:
        "randomized bracketed arms, eight warmups, one-pixel readPixels fence, CPU+upload+draw",
      seed,
      ...statistics,
      blocks,
    };
    el("results").innerHTML = summary
      .map(
        (s) =>
          `<tr><td>${s.arm}</td><td>${s.medianMs.toFixed(3)}</td><td>${s.p95Ms.toFixed(3)}</td><td>${s.addedMs.toFixed(3)} [${s.addedP10Ms.toFixed(3)}, ${s.addedP90Ms.toFixed(3)}]</td><td>${s.ratio.toFixed(3)}</td></tr>`,
      )
      .join("");
    Object.assign(window, { __breezeResult: result });
    el("info").textContent = JSON.stringify({ seed, ...statistics }, null, 2);
    el("status").textContent =
      `Complete · Simulated. ${statistics.verdict} Download JSON for per-round pairs.`;
    store.getState().set({ motion: "off", bench: false });
    download.disabled = false;
    run.disabled = false;
  })().catch((e: unknown) => {
    el("status").textContent = String(e);
    run.disabled = false;
  });
};
download.onclick = () => {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = "living-weather-bench.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
try {
  await ready();
  const initial = params.get("preset") ?? "calm";
  (el("preset") as HTMLSelectElement).value = initial;
  await preset(initial);
  run.disabled = false;
  if (params.get("autobench") === "1") run.click();
  el("info").textContent = JSON.stringify({ forest, status: lab.status() }, null, 2);
} catch (e) {
  el("status").textContent = String(e);
}
