import { cpSync, existsSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";
const root = new URL("../../../", import.meta.url);
const source = new URL("apps/web/dist-globe/", root);
const target = new URL("apps/ios/Modules/Globe/Sources/Globe/Resources/GlobeWeb/", root);
if (!existsSync(new URL("globe-embed.html", source))) throw Error("Build globe embed first");
// IIFE output avoids file-origin ES-module loading in WKWebView. All assets stay relative.
const html = readFileSync(new URL("globe-embed.html", source), "utf8");
const scripts = readdirSync(new URL("assets/", source)).filter((name) => name.endsWith(".js"));
const gzip = scripts.reduce(
  (n, name) => n + gzipSync(readFileSync(new URL(`assets/${name}`, source))).length,
  0,
);
if (gzip > 1_800_000 || !scripts.length)
  throw Error(`Embed scripts exceed 1.8 MB gzip budget: ${gzip}`);
for (const name of readdirSync(target))
  if (name !== "README.txt") rmSync(new URL(name, target), { recursive: true, force: true });
cpSync(fileURLToPath(source), fileURLToPath(target), { recursive: true });
cpSync(
  fileURLToPath(new URL("apps/web/public/fixtures/globe-splat/", root)),
  fileURLToPath(new URL("fixtures/globe-splat/", target)),
  { recursive: true },
);
writeFileSync(
  new URL("globe-embed.html", target),
  html.replace(/\s+type="module"/g, " defer").replace(/\s+crossorigin(?:="[^"]*")?/g, ""),
);
console.info(`Globe bundle copied; JS gzip ${gzip} bytes. Local IIFE entry and relative assets.`);
