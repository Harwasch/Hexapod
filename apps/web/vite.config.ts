import { cpSync, createReadStream, existsSync, statSync } from "node:fs";
import { extname, isAbsolute, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

const cesiumSource = fileURLToPath(new URL("./node_modules/cesium/Build/Cesium", import.meta.url));
const CESIUM_BASE_URL = "/cesium/";
/** CesiumJS's static build directories the app loads at runtime through CESIUM_BASE_URL. */
const CESIUM_STATIC_DIRS = ["Workers", "ThirdParty", "Assets", "Widgets"];

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
 * files under the package path instead of `cesium/<dir>/…`.
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
        cpSync(from, join(outDir, "cesium", dir), { recursive: true, dereference: true });
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
    rollupOptions: {
      // Three entries, and only the first one is a globe. `upload.html` is the page a
      // phone opens after scanning the handoff QR code — it should not pull a 3D globe
      // over cellular to pick one file. `admin.html` is the data console: full-page and
      // tabular, because a hundred runs and their parameters cannot be shown in a
      // floating panel over a 3D scene. Both import no CesiumJS, which e2e asserts by
      // watching for a cesium script request rather than by trusting this comment.
      input: {
        index: resolve(import.meta.dirname, "index.html"),
        upload: resolve(import.meta.dirname, "upload.html"),
        admin: resolve(import.meta.dirname, "admin.html"),
        // The scan viewer: one splat on its own, rendered with Spark (three.js), not the
        // globe. Also free of CesiumJS, which e2e asserts the same way.
        view: resolve(import.meta.dirname, "view.html"),
      },
      output: {
        manualChunks: (id) =>
          id.includes("node_modules/cesium") || id.includes("node_modules/@cesium")
            ? "cesium"
            : undefined,
      },
    },
  },
  optimizeDeps: {
    include: ["cesium"],
  },
});
