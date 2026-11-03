import { prepare, type PerfOptions } from "./viewerPerf";

const canvas = document.querySelector<HTMLCanvasElement>("#viewer");
if (!canvas) throw new Error("Missing benchmark canvas");
// Only this opt-in HTML entry installs the hook. No app module imports it.
Object.assign(window, {
  viewerPerf: { prepare: (options: PerfOptions) => prepare(canvas, options) },
});
