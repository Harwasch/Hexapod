import { cpSync, createReadStream, existsSync, readFileSync, statSync } from "node:fs";
import { extname, isAbsolute, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

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

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), cesiumDevAssets(), cesiumBuildAssets()],
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
      },
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
          ],
        },
      },
    },
  },
  optimizeDeps: {
    include: ["cesium"],
  },
});
