#!/usr/bin/env node
// Proves, against a production build, that the pages which must not load CesiumJS do not.
//
//   node apps/web/scripts/check-bundle.mjs            # reads apps/web/dist
//   node apps/web/scripts/check-bundle.mjs /tmp/dist  # or any other build directory
//
// Three of the four entries have no use for a globe: `admin.html` is a table of runs,
// `upload.html` is what a phone opens over cellular to pick one file, `view.html` renders a
// scan with Spark. e2e watches the dev server for a cesium request, which proves the
// *source* imports none -- but the dev server serves unbundled modules, so it cannot see
// what the bundler does with them. That gap is real: a shared chunking rule once put `tslib`
// inside `cesium-*.js`, and the admin page modulepreloaded the whole 4.9 MB engine for three
// helpers its dialogs use (vite.config.ts says how). Only the built files can show that.
//
// So this reads each page the way a browser would before any code runs -- its entry
// scripts, modulepreloads and stylesheets -- and follows every *static* import of every
// script it finds (a dynamic `import()` is loaded on demand, not before the first frame, and
// is deliberately not followed). A page fails if anything it reaches is a Cesium chunk: one
// named `cesium-*` (the chunk vite.config.ts makes for the engine) or one that carries the
// engine's GLSL built-ins (`czm_`), which nothing but CesiumJS contains, so a renamed or
// merged chunk is caught too. It prints what each page loads, globe included, so the same
// run is the before/after measurement of the bundle.
//
// Exit status: 0 when every Cesium-free page is Cesium-free, 1 when one is not, 2 when the
// build directory or an entry is missing -- a check that silently passed on an empty
// directory would be worse than none.

import { existsSync, readFileSync } from "node:fs";
import { basename, dirname, join, posix, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { gzipSync } from "node:zlib";

/** Pages that must not reach CesiumJS, and the one that is measured but allowed to. */
export const CESIUM_FREE_PAGES = ["admin.html", "upload.html", "view.html"];
export const GLOBE_PAGE = "index.html";

/** Engine chunks by name, and by the GLSL built-in prefix only CesiumJS's shaders use. */
const CESIUM_CHUNK_NAME = /^cesium[-.]/i;
const CESIUM_CONTENT_MARKER = "czm_";

/** The attribute `name` of one HTML tag, or undefined. */
function attribute(tag, name) {
  const match = new RegExp(`\\s${name}\\s*=\\s*("([^"]*)"|'([^']*)'|([^\\s>]+))`, "i").exec(tag);
  return match ? (match[2] ?? match[3] ?? match[4]) : undefined;
}

/**
 * The same-origin files an HTML document loads before any script runs: module scripts,
 * modulepreloads and stylesheets, as paths relative to the build root. Cross-origin URLs
 * (a font CDN, say) are not part of the bundle and are left out.
 */
export function documentResources(html) {
  const found = [];
  for (const match of html.matchAll(/<(script|link)\b[^>]*>/gi)) {
    const tag = match[0];
    let url;
    if (match[1].toLowerCase() === "script") {
      url = attribute(tag, "src");
    } else {
      const rel = (attribute(tag, "rel") ?? "").toLowerCase().split(/\s+/);
      if (rel.includes("modulepreload") || rel.includes("stylesheet")) url = attribute(tag, "href");
    }
    if (!url || /^(?:[a-z]+:)?\/\//i.test(url) || url.startsWith("data:")) continue;
    found.push(url.split(/[?#]/)[0].replace(/^\//, ""));
  }
  return found;
}

/**
 * The relative specifiers a built ES module imports statically: `import x from "./a.js"`,
 * `import "./a.js"`, `export { x } from "./a.js"`, minified or not. `import("./a.js")` and
 * `import.meta` are not matched.
 */
export function staticImports(code) {
  const found = new Set();
  const pattern =
    /(?:^|[^\w$.])(?:import|export)(?!\s*\()(?!\.meta)\s*(?:[\w$*{}\s,]*?\bfrom\s*)?(["'])(\.{1,2}\/[^"']+)\1/g;
  for (const match of code.matchAll(pattern)) found.add(match[2]);
  return [...found];
}

/** Whether a built file is (or contains) the CesiumJS engine. */
export function isCesiumChunk(path, contents) {
  return CESIUM_CHUNK_NAME.test(basename(path)) || contents.includes(CESIUM_CONTENT_MARKER);
}

/**
 * Everything one page loads before its first frame, following static imports from every
 * script the document names. Paths are relative to `root`; `missing` lists what the page
 * names that the build does not contain.
 */
export function pageGraph(root, page) {
  const html = readFileSync(join(root, page), "utf8");
  const files = new Map();
  const missing = [];
  const queue = documentResources(html);
  while (queue.length > 0) {
    const path = queue.shift();
    if (files.has(path)) continue;
    const absolute = join(root, path);
    if (!existsSync(absolute)) {
      missing.push(path);
      continue;
    }
    const contents = readFileSync(absolute);
    files.set(path, contents);
    if (!/\.m?js$/.test(path)) continue;
    for (const specifier of staticImports(contents.toString("utf8"))) {
      queue.push(posix.normalize(posix.join(posix.dirname(path), specifier)));
    }
  }
  return { files, missing };
}

function kilobytes(bytes) {
  return `${(bytes / 1000).toLocaleString("en-US", { maximumFractionDigits: 0 })} kB`;
}

/** Raw and gzip byte counts of a page's scripts and stylesheets. */
export function measure(files) {
  const total = { js: { raw: 0, gzip: 0, count: 0 }, css: { raw: 0, gzip: 0, count: 0 } };
  for (const [path, contents] of files) {
    const kind = path.endsWith(".css") ? total.css : /\.m?js$/.test(path) ? total.js : undefined;
    if (!kind) continue;
    kind.raw += contents.length;
    kind.gzip += gzipSync(contents).length;
    kind.count += 1;
  }
  return total;
}

/** Runs the check over one build directory. Returns the exit status; prints a report. */
export function checkBundle(root, log = console) {
  if (!existsSync(root)) {
    log.error(`check-bundle: no build at ${root} (run \`pnpm --filter @twin/web build\`)`);
    return 2;
  }
  let status = 0;
  for (const page of [GLOBE_PAGE, ...CESIUM_FREE_PAGES]) {
    if (!existsSync(join(root, page))) {
      log.error(`check-bundle: ${page} is missing from ${root}`);
      status = 2;
      continue;
    }
    const { files, missing } = pageGraph(root, page);
    const { js, css } = measure(files);
    log.info(
      `${page.padEnd(12)} JS ${String(js.count).padStart(2)} files ${kilobytes(js.raw).padStart(9)} raw ` +
        `${kilobytes(js.gzip).padStart(9)} gz · CSS ${kilobytes(css.raw)} raw ${kilobytes(css.gzip)} gz`,
    );
    for (const path of missing) {
      log.error(`  ${page} names ${path}, which is not in the build`);
      status = 2;
    }
    if (!CESIUM_FREE_PAGES.includes(page)) continue;
    const scripts = [...files].filter(([path]) => /\.m?js$/.test(path));
    if (scripts.length === 0) {
      log.error(`  ${page} loads no script at all; is this the right build directory?`);
      status = 2;
    }
    for (const [path, contents] of scripts) {
      if (!isCesiumChunk(path, contents.toString("utf8"))) continue;
      log.error(`  ${page} loads CesiumJS: ${path} (${kilobytes(contents.length)})`);
      if (status === 0) status = 1;
    }
  }
  if (status === 0) log.info(`check-bundle: ${CESIUM_FREE_PAGES.join(", ")} load no CesiumJS`);
  return status;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const here = dirname(fileURLToPath(import.meta.url));
  const root = resolve(process.argv[2] ?? join(here, "..", "dist"));
  process.exitCode = checkBundle(root);
}
