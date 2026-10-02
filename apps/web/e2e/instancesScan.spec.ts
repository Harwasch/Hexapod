/**
 * The scene-instance hooks on a real segmented scan, for a manual smoke test: skipped unless
 * `INSTANCES_SCAN_DIR` names a tileset directory whose root declares `extras.instances`
 * (`segment_scene.py` writes one). Screenshots go to `INSTANCES_SHOTS_DIR` when set, else the
 * test's output directory:
 *
 *   INSTANCES_SCAN_DIR=/path/to/tiles npx playwright test e2e/instancesScan.spec.ts
 *
 * Baseline, the largest instances hidden, one highlighted (amber, the rest dimmed), and
 * everything hidden (what is left is what no instance claims). The committed check on the
 * synthetic yard is e2e/instances.spec.ts; this one only asserts what holds on any scan.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

import { expect, test } from "@playwright/test";

interface Measure {
  coverage: number;
  amber: number;
  luma: number;
  warmth: number;
}
interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}
/** `src/dev/instancesHarness.ts`, as the page exposes it. */
interface InstancesHarness {
  view(headingDeg: number, pitchDeg: number, rangeM: number): Promise<Measure>;
  instances(): { id: number; parent: number | null; level: number; splats: number }[];
  set(
    state: { hidden?: number[]; highlighted?: number[]; dimOthers?: boolean },
    rect?: Rect,
  ): Promise<Measure>;
  cones(mode: "off" | "below"): Promise<Measure>;
  rectOf(id: number): Rect | null;
  measure(rect?: Rect): Measure;
  hooks(): { visibility: string[]; color: boolean; table: boolean };
  tiles(): { uri: string; splats: number }[];
}

const SCAN = process.env.INSTANCES_SCAN_DIR;
const SHOTS = process.env.INSTANCES_SHOTS_DIR;
const HEADING = Number(process.env.INSTANCES_HEADING ?? 30);
const PITCH = Number(process.env.INSTANCES_PITCH ?? -35);
const RANGE = Number(process.env.INSTANCES_RANGE ?? 9);

const HTML = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Instances harness</title>
    <style>
      html, body { margin: 0; height: 100%; background: #10141a; }
      #viewer, #viewer .cesium-widget, #viewer canvas { width: 100vw; height: 100vh; display: block; }
    </style>
  </head>
  <body>
    <div id="viewer"></div>
    <script type="module">
      const harness = await import("/src/dev/instancesHarness.ts");
      window.__instances = await harness.startInstancesHarness({
        container: document.getElementById("viewer"),
        url: "/scan-tiles/tileset.json",
      });
    </script>
  </body>
</html>`;

test.skip(!SCAN || !existsSync(join(SCAN, "tileset.json")), "INSTANCES_SCAN_DIR is not set");

test("a segmented scan's objects hide and highlight", async ({ page }) => {
  test.setTimeout(1_800_000);
  const dir = resolve(SCAN ?? ".");
  const shot = (name: string): string => {
    if (!SHOTS) return test.info().outputPath(name);
    mkdirSync(SHOTS, { recursive: true });
    return join(SHOTS, name);
  };
  const errors: string[] = [];
  const logs: string[] = [];
  page.on("console", (message) => {
    logs.push(`${message.type()}: ${message.text()}`);
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/scan-tiles/**", (route) => {
    const relative = new URL(route.request().url()).pathname.replace(/^.*\/scan-tiles\//, "");
    if (relative.includes("..")) return route.abort();
    const type = relative.endsWith(".json")
      ? "application/json"
      : relative.endsWith(".glb")
        ? "model/gltf-binary"
        : "application/octet-stream";
    return route.fulfill({
      status: 200,
      contentType: type,
      body: readFileSync(join(dir, relative)),
    });
  });
  await page.route("**/instances-harness.html", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: HTML }),
  );
  await page.goto("/instances-harness.html");
  await page.waitForFunction(() => "__instances" in window, undefined, { timeout: 300_000 });
  const call = <K extends keyof InstancesHarness>(
    method: K,
    ...args: Parameters<InstancesHarness[K]>
  ): Promise<Awaited<ReturnType<InstancesHarness[K]>>> =>
    page.evaluate(
      ([m, a]) => {
        const harness = (window as unknown as { __instances: Record<string, unknown> }).__instances;
        return (harness[m] as (...x: unknown[]) => unknown)(...a);
      },
      [method, args] as [string, unknown[]],
    ) as Promise<Awaited<ReturnType<InstancesHarness[K]>>>;

  // Each step's result and time, written as it goes (a slow software-GL run can time out).
  const progress: Record<string, unknown> = {};
  const started = Date.now();
  const step = async <T>(name: string, run: () => Promise<T>): Promise<T> => {
    const value = await run();
    progress[name] = value;
    progress[`${name}AtS`] = (Date.now() - started) / 1000;
    writeFileSync(shot("measures.json"), JSON.stringify({ progress, logs }, null, 1));
    return value;
  };

  // Twice: the first view also waits for the level-of-detail tiles to finish refining.
  await step("first", () => call("view", HEADING, PITCH, RANGE));
  const baseline = await step("baseline", () => call("view", HEADING, PITCH, RANGE));
  await page.screenshot({ path: shot("1-baseline.png") });
  const hooks = await step("hooks", () => call("hooks"));
  const list = await call("instances");
  const largest = await step("largest", () =>
    Promise.resolve(list.filter((i) => i.parent === null).slice(0, 3)),
  );
  const hidden = await step("hidden", () => call("set", { hidden: largest.map((i) => i.id) }));
  await page.screenshot({ path: shot("2-hidden-largest.png") });
  const target = largest[0]?.id ?? 1;
  const rect = (await step("rect", () => call("rectOf", target))) ?? undefined;
  const before = await step("before", () => call("set", {}, rect));
  const lit = await step("lit", () => call("set", { highlighted: [target] }, rect));
  await step("litWhole", () => call("measure"));
  await page.screenshot({ path: shot("3-highlight.png") });
  const all = await step("all", () => call("set", { hidden: list.map((i) => i.id) }));
  await page.screenshot({ path: shot("4-hidden-all.png") });
  await step("restored", () => call("set", {}));

  test.info().annotations.push({ type: "measures", description: JSON.stringify(progress) });
  test.info().annotations.push({
    type: "logs",
    description: logs.filter((l) => /instance|shader|error|warn/i.test(l)).join("\n"),
  });
  expect(errors.filter((e) => /shader|compile|link|webgl/i.test(e))).toEqual([]);
  expect(hooks.table).toBe(true);
  expect(hooks.color).toBe(true);
  expect(hooks.visibility).toContain("splatInstanceVisibility");
  expect(hidden.coverage).toBeLessThan(baseline.coverage);
  expect(lit.amber).toBeGreaterThan(before.amber);
  expect(all.coverage).toBeLessThan(baseline.coverage * 0.5);
});
