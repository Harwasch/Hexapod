#!/usr/bin/env node
/** Local-only render check. No service, training process, or production build is used. */
import { createReadStream } from "node:fs";
import { mkdir, readFile, realpath, stat, writeFile } from "node:fs/promises";
import { dirname, extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { chromium } from "@playwright/test";
import { createServer } from "vite";

const { values } = parseArgs({
  options: {
    cameras: { type: "string" },
    "ref-dir": { type: "string" },
    source: { type: "string" },
    out: { type: "string" },
    budget: { type: "string" },
    "timeout-seconds": { type: "string", default: "900" },
    antialiased: { type: "boolean", default: false },
  },
});
for (const name of ["cameras", "ref-dir", "source", "out"]) {
  if (!values[name])
    throw new Error(`Missing --${name}. See apps/web/README.md (Console render check).`);
}
const budget =
  values.budget === "full"
    ? "full"
    : values.budget === undefined
      ? undefined
      : Number(values.budget);
if (budget !== undefined && budget !== "full" && !(Number.isSafeInteger(budget) && budget > 0)) {
  throw new Error("--budget must be full or a positive integer");
}
const timeoutMs = Number(values["timeout-seconds"]) * 1000;
if (!(Number.isFinite(timeoutMs) && timeoutMs > 0))
  throw new Error("--timeout-seconds must be positive");
const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = await realpath(values.source);
const tiles = (await stat(source)).isDirectory();
if (!tiles && extname(source).toLowerCase() !== ".spz")
  throw new Error("--source must be an SPZ or tileset directory");
const sourceRoot = tiles ? source : dirname(source);
const references = await realpath(values["ref-dir"]);
const output = resolve(values.out);
// Never write outputs over inputs (including the byte-identity-gated fixture).
for (const input of [sourceRoot, references]) {
  if (output === input || output.startsWith(input + sep))
    throw new Error("--out must be separate from inputs");
}
const document = JSON.parse(await readFile(values.cameras, "utf8"));
const cameras = Object.entries(document.cameras ?? document).filter(([name]) =>
  /^train-[\w-]+$/.test(name),
);
if (!cameras.length) throw new Error("No train-* cameras in camera JSON");
const port = Number(process.env.E2E_PORT ?? 0); // 0 reserves our own ephemeral loopback port.
const server = await createServer({
  root: web,
  configFile: resolve(web, "vite.config.ts"),
  server: { host: "127.0.0.1", port, strictPort: true, open: false },
  plugins: [
    {
      name: "console-check-local-inputs",
      apply: "serve",
      configureServer(server) {
        server.middlewares.use((request, response, next) => {
          const pathname = new URL(request.url ?? "/", "http://localhost").pathname;
          if (!pathname.startsWith("/__console-check/")) return next();
          void (async () => {
            let path;
            if (pathname === "/__console-check/source.spz" && !tiles) path = source;
            else if (pathname.startsWith("/__console-check/tiles/") && tiles) {
              const relative = decodeURIComponent(pathname.slice("/__console-check/tiles/".length));
              if (relative.split("/").some((p) => p === "..") || relative.includes("\\"))
                throw new Error("Unsafe path");
              path = await realpath(resolve(source, relative));
              if (!path.startsWith(source + sep)) throw new Error("Path outside source");
            } else {
              const name = pathname.slice("/__console-check/".length);
              if (!cameras.some(([camera]) => name === `trained-${camera}.png`))
                throw new Error("Unknown reference");
              path = await realpath(resolve(references, name));
              if (!path.startsWith(references + sep)) throw new Error("Path outside references");
            }
            response.setHeader(
              "Content-Type",
              { ".json": "application/json", ".png": "image/png" }[extname(path)] ??
                "application/octet-stream",
            );
            response.setHeader("Cache-Control", "no-store");
            const stream = createReadStream(path);
            stream.on("error", () => {
              response.statusCode = 404;
              response.end();
            });
            stream.pipe(response);
          })().catch(() => {
            response.statusCode = 404;
            response.end();
          });
        });
      },
    },
  ],
});
let browser;
try {
  await server.listen();
  const address = server.httpServer.address();
  if (!address || typeof address === "string") throw new Error("No local server address");
  const origin = `http://127.0.0.1:${address.port}`;
  browser = await chromium.launch({
    headless: true,
    args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
  });
  const page = await browser.newPage({ deviceScaleFactor: 1 });
  // The check only consumes local files. A tile accidentally naming a remote URI fails.
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    return url.origin === origin || url.protocol === "data:" || url.protocol === "blob:"
      ? route.continue()
      : route.abort();
  });
  const errors = [];
  page.on("console", (message) => {
    if (message.text().startsWith("console-check progress")) console.info(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`${origin}/console-check.html`);
  await page.waitForFunction(() => typeof window.consoleCheck === "function");
  await mkdir(output, { recursive: true });
  const report = {
    source,
    cameras: resolve(values.cameras),
    referenceDirectory: references,
    antialiased: values.antialiased,
    timeoutSeconds: timeoutMs / 1000,
    renderer: "SwiftShader",
    metric: "PSNR over all PNG RGB samples / 255; null = identical (+infinity)",
    renders: {},
  };
  for (const [name, camera] of cameras) {
    await page.setViewportSize({ width: camera.width, height: camera.height });
    let result;
    try {
      result = await page.evaluate((input) => window.consoleCheck(input), {
        camera,
        source: tiles ? "/__console-check/tiles/tileset.json" : "/__console-check/source.spz",
        tiles,
        reference: `/__console-check/trained-${name}.png`,
        budget,
        timeoutMs,
        antialiased: values.antialiased,
      });
      if (errors.length) throw new Error(errors.join("\n"));
    } catch (error) {
      const progress = await page
        .evaluate(() => window.consoleCheckProgress ?? null)
        .catch(() => null);
      report.renders[name] = {
        error: String(error),
        width: camera.width,
        height: camera.height,
        ...progress,
      };
      await writeFile(resolve(output, "scores.json"), JSON.stringify(report, null, 2) + "\n");
      throw error;
    }
    const { png, ...score } = result;
    const file = `console-${name}.png`;
    await writeFile(resolve(output, file), Buffer.from(png.split(",")[1], "base64"));
    report.renders[name] = { file, width: camera.width, height: camera.height, ...score };
    await writeFile(resolve(output, "scores.json"), JSON.stringify(report, null, 2) + "\n");
    console.info(`${name}: ${score.psnr ?? "infinity"} dB (${score.drawnSplats} splats)`);
  }
  await writeFile(resolve(output, "scores.json"), JSON.stringify(report, null, 2) + "\n");
} finally {
  await browser?.close();
  await server.close();
}
