/**
 * `scripts/check-bundle.mjs`, the CI check that the admin, phone and scan-viewer pages of a
 * production build load no CesiumJS. It is run as CI runs it -- `node <script> <dist>` --
 * against small hand-made builds, because the exit status is the whole contract: a check
 * that passed on the bug it exists for (the admin page modulepreloading the engine through a
 * shared `tslib`) or on an empty directory would be worse than none.
 */
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

const SCRIPT = resolve(__dirname, "../../scripts/check-bundle.mjs");

function page(entry: string, preloads: string[] = []): string {
  const links = preloads
    .map((href) => `<link rel="modulepreload" crossorigin href="/assets/${href}">`)
    .join("\n");
  return `<!doctype html><html><head>
<link rel="stylesheet" href="https://fonts.example.com/css">
<script type="module" crossorigin src="/assets/${entry}"></script>
${links}
</head><body></body></html>`;
}

/** A build in a temporary directory: `{ "admin.html": "...", "assets/a.js": "..." }`. */
function build(files: Record<string, string>): string {
  const root = mkdtempSync(join(tmpdir(), "check-bundle-"));
  roots.push(root);
  for (const [path, contents] of Object.entries(files)) {
    mkdirSync(dirname(join(root, path)), { recursive: true });
    writeFileSync(join(root, path), contents);
  }
  return root;
}

/** Every page clean: the globe imports the engine, the other three do not. */
function cleanBuild(): Record<string, string> {
  return {
    "index.html": page("index-1.js", ["cesium-1.js", "helpers-1.js"]),
    "admin.html": page("admin-1.js", ["helpers-1.js", "client-1.js"]),
    "upload.html": page("upload-1.js"),
    "view.html": page("view-1.js"),
    "assets/index-1.js": `import{a as e}from"./cesium-1.js";import"./helpers-1.js";e();`,
    "assets/cesium-1.js": `const s="uniform float czm_frameNumber;";export{s as a};`,
    "assets/helpers-1.js": `var o=Object.assign;export{o as a};`,
    "assets/admin-1.js": `import{a as t}from"./helpers-1.js";import{r}from"./client-1.js";t(r);`,
    "assets/client-1.js": `import{a}from"./helpers-1.js";export const r=a;`,
    "assets/upload-1.js": `export {};`,
    "assets/view-1.js": `const later=()=>import("./cesium-1.js");export{later};`,
  };
}

function run(root: string): { status: number | null; output: string } {
  const result = spawnSync(process.execPath, [SCRIPT, root], { encoding: "utf8" });
  return { status: result.status, output: `${result.stdout}${result.stderr}` };
}

const roots: string[] = [];
afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("check-bundle.mjs", () => {
  it("passes a build whose only Cesium page is the globe, and measures every page", () => {
    const { status, output } = run(build(cleanBuild()));
    expect(output).toContain("load no CesiumJS");
    expect(output).toMatch(/index\.html\s+JS\s+3 files/);
    expect(output).toMatch(/admin\.html\s+JS\s+3 files/);
    expect(status).toBe(0);
  });

  it("fails when a static import two hops from the page reaches the engine chunk", () => {
    // The bug it exists for: the admin page's shared chunk importing helpers from cesium-*.
    const files = cleanBuild();
    files["assets/client-1.js"] = `import{o}from"./cesium-1.js";export const r=o;`;
    const { status, output } = run(build(files));
    expect(output).toContain("admin.html loads CesiumJS: assets/cesium-1.js");
    expect(status).toBe(1);
  });

  it("catches the engine by its shader built-ins when the chunk has another name", () => {
    const files = cleanBuild();
    files["assets/upload-1.js"] = `import"./vendor-1.js";`;
    files["assets/vendor-1.js"] = `const s="czm_viewport";export{s};`;
    const { status, output } = run(build(files));
    expect(output).toContain("upload.html loads CesiumJS: assets/vendor-1.js");
    expect(status).toBe(1);
  });

  it("does not follow a dynamic import, which loads on demand rather than before the page", () => {
    // view-1.js in the clean build does `import("./cesium-1.js")`; that alone passes.
    const { status } = run(build(cleanBuild()));
    expect(status).toBe(0);
  });

  it("refuses a missing build or a page that names a file the build lacks", () => {
    expect(run(join(tmpdir(), "check-bundle-does-not-exist")).status).toBe(2);
    const files = cleanBuild();
    files["admin.html"] = page("admin-missing.js");
    const { status, output } = run(build(files));
    expect(output).toContain("admin.html names assets/admin-missing.js");
    expect(status).toBe(2);
  });
});
