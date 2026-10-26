// A long-lived headless browser on the dev app, driven over HTTP on 127.0.0.1:$DRIVER_PORT:
// POST /eval (body: JS expression, awaited) -> JSON; POST /shot?path=... -> PNG saved;
// POST /goto (body: URL path). For iterating on the living-lab shaders by eye.
import { createServer } from "node:http";

import { chromium } from "@playwright/test";

const base = process.env.LAB_URL ?? "http://localhost:5600";
const port = Number(process.env.DRIVER_PORT ?? 5601);
const gpu = process.env.LAB_GPU === "1";
const browser = await chromium.launch({
  args: gpu
    ? [
        "--no-sandbox",
        "--use-gl=angle",
        "--use-angle=vulkan",
        "--enable-gpu",
        "--ignore-gpu-blocklist",
      ]
    : ["--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
});
let page = await browser.newPage({
  viewport: { width: Number(process.env.LAB_W ?? 1280), height: Number(process.env.LAB_H ?? 720) },
});
const logs = [];
const watch = (p) => {
  p.on("console", (m) => {
    logs.push(`${m.type()}: ${m.text().slice(0, 400)}`);
    if (logs.length > 400) logs.shift();
  });
  p.on("pageerror", (e) => logs.push(`pageerror: ${String(e).slice(0, 400)}`));
};
watch(page);

const body = (req) =>
  new Promise((ok) => {
    let s = "";
    req.on("data", (d) => (s += d));
    req.on("end", () => ok(s));
  });
createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", "http://x");
  try {
    let out;
    if (url.pathname === "/fresh") {
      await page.close().catch(() => undefined);
      page = await browser.newPage({
        viewport: {
          width: Number(process.env.LAB_W ?? 1280),
          height: Number(process.env.LAB_H ?? 720),
        },
      });
      watch(page);
      out = "fresh";
    } else if (url.pathname === "/goto") {
      await page.goto(base + (await body(req)));
      out = "ok";
    } else if (url.pathname === "/eval") {
      out = await page.evaluate(`(async () => (${await body(req)}))()`);
    } else if (url.pathname === "/shot") {
      await page.screenshot({
        path: url.searchParams.get("path"),
        timeout: 240_000,
        animations: "allow",
        caret: "initial",
      });
      out = "saved";
    } else if (url.pathname === "/logs") {
      out = logs.splice(0);
    } else if (url.pathname === "/quit") {
      res.end("bye");
      await browser.close();
      process.exit(0);
    }
    res.end(JSON.stringify(out ?? null));
  } catch (e) {
    res.statusCode = 500;
    res.end(String(e));
  }
}).listen(port, "127.0.0.1", () => console.info(`driver on ${port}`));
