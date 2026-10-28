import { startLivingLabHarness } from "./livingLabHarness";
import { useAsCaptured } from "@/state/asCaptured";

const el = (id: string) => {
  const e = document.getElementById(id);
  if (!e) throw new Error(id);
  return e;
};
const run = el("run") as HTMLButtonElement,
  download = el("download") as HTMLButtonElement;
const params = new URLSearchParams(location.search);
const lab = await startLivingLabHarness(
  el("viewer"),
  params.get("tiles") ?? "/data/tree/bench-tileset.json",
);
// Exposed in both static and dev entries for reproducible recordings.
Object.assign(window, { __lab: lab, LL: lab.lab, captured: useAsCaptured });
lab.widget.useBrowserRecommendedResolution = true;
lab.widget.resolutionScale = 1;
lab.viewBounds(35, -12, 15.4);
const store = lab.lab;
const bake = params.get("bake") ?? "/data/bake/articulation.json";
const delay = () => new Promise<void>((r) => setTimeout(r, 100));
async function ready() {
  const deadline = performance.now() + 180000;
  while (lab.status().loading || !lab.status().settled || lab.status().gaussians !== 687226) {
    if (performance.now() > deadline) throw new Error("Tree loading timed out");
    await delay();
  }
}
async function preset(arm: string) {
  const before = lab.status().frames;
  store.getState().set({
    motion: arm === "off" ? "off" : "articulated",
    strength: 1,
    windBearingDeg: 60,
    water: false,
    bench: true,
    articulationUrl: bake + (arm === "off" ? "" : `?breeze=${arm}`),
  });
  // Let the render hook observe the new URL before checking its load status.
  while (lab.status().frames <= before) await delay();
  if (arm !== "off" && !useAsCaptured.getState().on) {
    const deadline = performance.now() + 180000;
    while (!store.getState().articulationStatus.startsWith("Simulated")) {
      if (store.getState().articulationStatus.startsWith("Error") || performance.now() > deadline)
        throw new Error(store.getState().articulationStatus);
      await delay();
    }
  }
  await ready();
}
interface BenchWindow {
  __livingBench?: number[];
}
const benchWindow = window as unknown as BenchWindow;
const quantile = (v: number[], q: number) =>
  [...v].sort((a, b) => a - b)[Math.min(v.length - 1, Math.floor(v.length * q))] ?? 0;
let result: unknown;
el("captured").addEventListener("change", (e) =>
  useAsCaptured.getState().set((e.target as HTMLInputElement).checked),
);
(el("captured") as HTMLInputElement).checked = useAsCaptured.getState().on;
el("preset").addEventListener("change", (e) => {
  void preset((e.target as HTMLSelectElement).value);
});
run.onclick = () => {
  void (async () => {
    run.disabled = true;
    download.disabled = true;
    if (useAsCaptured.getState().on)
      throw new Error("Turn off As captured to measure simulated motion");
    const blocks: { round: number; arm: string; samples: number[] }[] = [];
    const rounds = Number(params.get("rounds") ?? 3),
      samples = Number(params.get("samples") ?? 32);
    if (
      !Number.isInteger(rounds) ||
      rounds < 1 ||
      rounds > 10 ||
      !Number.isInteger(samples) ||
      samples < 2 ||
      samples > 240
    )
      throw new Error("Invalid benchmark length");
    for (let round = 0; round < rounds; round++)
      for (const arm of ["off", "calm", "off", "breezy", "off", "windy"]) {
        el("status").textContent = `Round ${round + 1}/${rounds}: ${arm}`;
        await preset(arm);
        store.getState().set({ clock: null });
        benchWindow.__livingBench = [];
        while ((benchWindow.__livingBench?.length ?? 0) < samples + 4) await delay();
        blocks.push({
          round,
          arm,
          samples: (benchWindow.__livingBench ?? []).slice(4, samples + 4),
        });
      }
    const summary = ["off", "calm", "breezy", "windy"].map((arm) => {
      const selected = blocks.filter((b) => b.arm === arm).flatMap((b) => b.samples);
      const pairs = blocks.flatMap((b, i) =>
        b.arm === arm && i > 0
          ? [
              {
                added: quantile(b.samples, 0.5) - quantile(blocks[i - 1]?.samples ?? [], 0.5),
                ratio: quantile(b.samples, 0.5) / quantile(blocks[i - 1]?.samples ?? [], 0.5),
              },
            ]
          : [],
      );
      return {
        arm,
        medianMs: quantile(selected, 0.5),
        p95Ms: quantile(selected, 0.95),
        addedMs:
          arm === "off"
            ? 0
            : quantile(
                pairs.map((p) => p.added),
                0.5,
              ),
        ratio:
          arm === "off"
            ? 1
            : quantile(
                pairs.map((p) => p.ratio),
                0.5,
              ),
      };
    });
    const canvas = el("viewer").querySelectorAll("canvas");
    const gl = Array.from(canvas)
      .map((c) => c.getContext("webgl2"))
      .find(Boolean);
    const debug = gl?.getExtension("WEBGL_debug_renderer_info");
    if (Array.from(canvas).some((c) => c.width !== 1920 || c.height !== 1080))
      throw new Error("Drawing buffer changed: expected 1920 × 1080");
    result = {
      time: new Date().toISOString(),
      userAgent: navigator.userAgent,
      renderer: debug ? String(gl?.getParameter(debug.UNMASKED_RENDERER_WEBGL)) : "unavailable",
      buffers: Array.from(canvas).map((c) => [c.width, c.height]),
      status: lab.status(),
      method: "interleaved, four warmups, one-pixel readPixels fence, CPU+upload+draw",
      summary,
      blocks,
    };
    el("results").innerHTML = summary
      .map(
        (s) =>
          `<tr><td>${s.arm}</td><td>${s.medianMs.toFixed(3)}</td><td>${s.p95Ms.toFixed(3)}</td><td>${s.addedMs.toFixed(3)}</td><td>${s.ratio.toFixed(3)}</td></tr>`,
      )
      .join("");
    Object.assign(window, { __breezeResult: result });
    el("status").textContent = "Complete · Simulated. Download JSON to retain all samples.";
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
  a.download = "living-breeze-bench.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
try {
  await ready();
  const initial = params.get("preset") ?? "calm";
  if (!["off", "calm", "breezy", "windy"].includes(initial))
    throw new Error("Invalid preview preset");
  (el("preset") as HTMLSelectElement).value = initial;
  await preset(initial);
  run.disabled = false;
  el("status").textContent = useAsCaptured.getState().on
    ? "Ready · As captured (motion hidden)"
    : initial === "off"
      ? "Ready · Plain scan"
      : `Ready · Simulated ${initial} breeze`;
} catch (e) {
  el("status").textContent = String(e);
}
