import { cpSync, createReadStream, existsSync, readFileSync, statSync } from "node:fs";
import { extname, isAbsolute, join, normalize, resolve } from "node:path";
import { Readable } from "node:stream";
import type { ReadableStream as NodeReadableStream } from "node:stream/web";
import { fileURLToPath, pathToFileURL } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv, type Plugin } from "vite";

const cesiumSource = fileURLToPath(new URL("./node_modules/cesium/Build/Cesium", import.meta.url));
const cesiumVersion = (
  JSON.parse(
    readFileSync(
      fileURLToPath(new URL("./node_modules/cesium/package.json", import.meta.url)),
      "utf8",
    ),
  ) as { version: string }
).version;
/**
 * Where CesiumJS's static directories are served, and where its workers, WebAssembly and
 * textures are fetched from at runtime: a path named for the installed version
 * (`/cesium/1.145.0/`). Those files are not content-hashed, so at a fixed `/cesium/` an
 * upgrade changed what an old URL returned, and they could only be cached for a day (40-odd
 * revalidations on the first visit of every day). Named for the version, an upgrade changes
 * the URLs instead, and infra/pages/_headers marks the whole of `/cesium/*` immutable. The
 * version alone names the contents: `Build/Cesium` ships prebuilt in the npm package, and the
 * pnpm patch only touches `@cesium/engine`'s sources. Dev serves the same path.
 */
const CESIUM_BASE_URL = `/cesium/${cesiumVersion}/`;
/**
 * CesiumJS's static build directories the app loads at runtime through CESIUM_BASE_URL. Not
 * `Widgets`: only `@cesium/widgets` reads from it (InfoBox, picker icons), and the scene runs on
 * the engine's `CesiumWidget` without any of them (CesiumSceneManager).
 */
const CESIUM_STATIC_DIRS = ["Workers", "ThirdParty", "Assets"];

const MIME: Record<string, string> = {
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".xml": "application/xml",
  ".wasm": "application/wasm",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".svg": "image/svg+xml",
  ".ktx2": "image/ktx2",
  ".glsl": "text/plain",
  ".ttf": "font/ttf",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
};

/** Serves CesiumJS's static build assets (Workers, Assets, ThirdParty, Widgets) during `vite dev`. */
function cesiumDevAssets(): Plugin {
  return {
    name: "cesium-dev-assets",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = req.url?.split("?")[0] ?? "";
        if (!url.startsWith(CESIUM_BASE_URL)) return next();
        const relative = normalize(decodeURIComponent(url.slice(CESIUM_BASE_URL.length)));
        if (relative.startsWith("..")) return next();
        const file = join(cesiumSource, relative);
        if (!existsSync(file) || !statSync(file).isFile()) return next();
        res.setHeader(
          "Content-Type",
          MIME[extname(file).toLowerCase()] ?? "application/octet-stream",
        );
        res.setHeader("Cache-Control", "public, max-age=3600");
        createReadStream(file).pipe(res);
      });
    },
  };
}

/**
 * Copies CesiumJS's static build directories into the bundle. A plain recursive copy: a
 * glob-based copy needs forward slashes on Windows and, with this layout, lands nested
 * files under the package path instead of `cesium/<version>/<dir>/…`.
 */
function cesiumBuildAssets(): Plugin {
  let outDir = "dist";
  return {
    name: "cesium-build-assets",
    apply: "build",
    configResolved(config) {
      outDir = isAbsolute(config.build.outDir)
        ? config.build.outDir
        : resolve(config.root, config.build.outDir);
    },
    closeBundle() {
      for (const dir of CESIUM_STATIC_DIRS) {
        const from = join(cesiumSource, dir);
        if (!existsSync(from)) throw new Error(`CesiumJS build assets not found: ${from}`);
        cpSync(from, join(outDir, CESIUM_BASE_URL.slice(1), dir), {
          recursive: true,
          dereference: true,
        });
      }
    },
  };
}

/** `functions/r2/[[path]].js`: a Request in, a Response out, as Pages runs it. */
interface TileProxyFunction {
  onRequest: (context: {
    request: Request;
    env?: { TILE_PROXY_HOST?: string };
  }) => Promise<Response>;
}

/** Headers of one hop, which Node's server sets itself. */
const HOP_HEADERS = new Set(["connection", "keep-alive", "transfer-encoding"]);

/**
 * The public tile bucket on the dev server's own origin, as Pages serves it in production
 * (`functions/r2/`, `src/lib/tileProxy.ts`): the same function, run here, when
 * `TILE_PROXY_HOST` names a bucket (in the environment or `.env`). A tileset on that bucket
 * -- the hosted scans `python -m app.seed.scans` registers -- then loads from this origin, its
 * tiles, objects and object embeddings with it, so the bucket's CORS rule, which names the
 * deployment's origin, does not decide whether a dev server on any port can read them. Unset,
 * nothing changes: the probe 404s and every tileset loads from its own URL.
 *
 * The function's fetch is Node's: behind an HTTPS proxy, start the dev server with
 * `NODE_USE_ENV_PROXY=1`.
 */
function tileProxyDev(): Plugin {
  return {
    name: "tile-proxy-dev",
    apply: "serve",
    async configureServer(server) {
      const fromFiles = loadEnv(server.config.mode, server.config.envDir || server.config.root, "");
      const host = (process.env.TILE_PROXY_HOST ?? fromFiles.TILE_PROXY_HOST ?? "").trim();
      if (host === "") return;
      // Through a computed URL: the function is plain JS, and this project does not allowJs.
      const where = pathToFileURL(resolve(import.meta.dirname, "../../functions/r2/[[path]].js"));
      const { onRequest } = (await import(/* @vite-ignore */ where.href)) as TileProxyFunction;
      server.middlewares.use((req, res, next) => {
        const url = req.url ?? "";
        if (url !== "/r2" && !url.startsWith("/r2/")) return next();
        const headers = new Headers();
        for (const [name, value] of Object.entries(req.headers)) {
          if (typeof value === "string") headers.set(name, value);
        }
        const request = new Request(`http://${req.headers.host ?? "localhost"}${url}`, {
          method: req.method,
          headers,
        });
        onRequest({ request, env: { TILE_PROXY_HOST: host } })
          .then((response) => {
            res.statusCode = response.status;
            // Node's fetch hands back the body decoded: its encoding and length are not this
            // body's any more.
            const decoded = response.headers.has("content-encoding");
            response.headers.forEach((value, name) => {
              if (HOP_HEADERS.has(name)) return;
              if (decoded && (name === "content-encoding" || name === "content-length")) return;
              res.setHeader(name, value);
            });
            if (!response.body || req.method === "HEAD") {
              res.end();
              return;
            }
            Readable.fromWeb(response.body as unknown as NodeReadableStream).pipe(res);
          })
          .catch((error: unknown) => next(error));
      });
      server.config.logger.info(`  ➜  Tiles of ${host} served under /r2/ (TILE_PROXY_HOST)`);
    },
  };
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), cesiumDevAssets(), cesiumBuildAssets(), tileProxyDev()],
  define: {
    CESIUM_BASE_URL: JSON.stringify(CESIUM_BASE_URL),
  },
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
  // `pnpm preview` serves the production build against the local API: the real way to
  // judge performance (dev mode serves Cesium as thousands of unbundled modules and runs
  // React in development mode).
  preview: {
    port: 4173,
    strictPort: true,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
  build: {
    target: "es2022",
    sourcemap: true,
    chunkSizeWarningLimit: 5000,
    rolldownOptions: {
      // Three entries, and only the first one is a globe. `upload.html` is the page a
      // phone opens after scanning the handoff QR code — it should not pull a 3D globe
      // over cellular to pick one file. `admin.html` is the data console: full-page and
      // tabular, because a hundred runs and their parameters cannot be shown in a
      // floating panel over a 3D scene. Both import no CesiumJS. e2e watches the dev
      // server for a cesium request, which proves the *source* imports none; only the
      // build can prove the *chunks* do not, so `scripts/check-bundle.mjs` reads dist/
      // and fails CI when any of these pages reaches a Cesium chunk.
      input: {
        index: resolve(import.meta.dirname, "index.html"),
        upload: resolve(import.meta.dirname, "upload.html"),
        admin: resolve(import.meta.dirname, "admin.html"),
        // The scan viewer: one splat on its own, rendered with Spark (three.js), not the
        // globe. Also free of CesiumJS, which e2e asserts the same way.
        view: resolve(import.meta.dirname, "view.html"),
        // A separate neural-world product on the same host, with no globe dependency.
        worlds: resolve(import.meta.dirname, "worlds.html"),
      },
      // Keep shared chunks out of the entries. From rolldown 1.2.9 the common-chunk merge
      // (on by default) folded the globe page's lazy modules (demoRanchData, planFlow,
      // landPlanning, queries, ...) into index-*.js: 190 kB gz to 238 kB, past check-bundle's
      // INDEX_ENTRY_GZIP_BUDGET. https://rolldown.rs/reference/InputOptions.experimental
      experimental: { chunkOptimization: { mergeCommonChunks: false } },
      output: {
        codeSplitting: {
          groups: [
            // Shared runtime helpers, in a chunk of their own, claimed *before* the
            // engine's group below. A group takes its modules' dependencies with it
            // (`includeDependenciesRecursively`, on by default, and what keeps chunks free
            // of import cycles), and CesiumJS depends on `autolinker`, which depends on
            // `tslib`. So when the engine had the only group, `tslib` was bundled inside
            // `cesium-*.js`, and the admin page — whose dialogs use `react-remove-scroll`,
            // which also imports `tslib` — modulepreloaded all 4.9 MB of the engine for
            // three 200-byte helpers. Vite's dynamic-import preload helper landed there
            // the same way. Anything else the engine and another page come to share shows
            // up in check-bundle.mjs, which names the page and the chunk.
            {
              name: "helpers",
              test: /node_modules[\\/](tslib|@babel[\\/]runtime|@swc[\\/]helpers)[\\/]|vite[\\/]preload-helper/,
              priority: 2,
            },
            {
              name: "cesium",
              test: /node_modules[\\/](cesium|@cesium)[\\/]/,
              priority: 1,
            },
            // The cell views' shared code -- the look, the grid it reads, and what hands a
            // view to PlayCanvas or Spark -- is reached from five on-demand chunks (the view,
            // the per-cell rules, the key, both renderers), each a different set, so left alone
            // it split into four chunks of a few kB, and the globe's entry carries every
            // chunk's name for the preloader. One chunk for all of it.
            {
              name: "cellLook",
              test: /src[\\/](lib[\\/](cellLook|cellGrid|cellRamps)|cesium[\\/]scanView[\\/](cellModifier|scanCellView))\.ts$/,
            },
          ],
        },
      },
    },
  },
  optimizeDeps: {
    // The dev server's dependency scan follows imports from the HTML entries, dynamic imports
    // included, but not into module workers (`new Worker(new URL(...))`). A package only a
    // worker imports is found on the worker's first load instead, re-bundled, and the open
    // page reloaded: on a cold server, the first scan drawn under PlayCanvas reloaded the
    // app mid-test (e2e cameraRest.spec.ts in CI). Listed here, they are bundled at start.
    include: ["cesium", "@spz-loader/core"],
  },
});
