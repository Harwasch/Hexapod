import { createReadStream, existsSync, statSync } from "node:fs";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";

const cesiumSource = fileURLToPath(new URL("./node_modules/cesium/Build/Cesium", import.meta.url));
const CESIUM_BASE_URL = "/cesium/";

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

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    cesiumDevAssets(),
    viteStaticCopy({
      targets: ["Workers", "ThirdParty", "Assets", "Widgets"].map((dir) => ({
        src: `${cesiumSource}/${dir}/*`,
        dest: `cesium/${dir}`,
      })),
    }),
  ],
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
  preview: {
    port: 4173,
    strictPort: true,
  },
  build: {
    target: "es2022",
    sourcemap: true,
    chunkSizeWarningLimit: 5000,
    rollupOptions: {
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
