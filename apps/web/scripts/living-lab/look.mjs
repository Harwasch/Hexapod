// Opens the dev app at a site and saves a screenshot: `node look.mjs <siteId> <out.png> [renderer]`.
import { chromium } from "@playwright/test";

const [siteId, out, renderer = "playcanvas"] = process.argv.slice(2);
const base = process.env.LAB_URL ?? "http://localhost:5291";
const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
page.on("console", (m) => { if (m.type() === "error") console.log("console:", m.text().slice(0, 200)); });
await page.goto(`${base}/?renderer=${renderer}`);
await page.waitForFunction(() => "__twin" in window && Boolean(window.__twin), undefined, { timeout: 180_000 });
await page.getByTestId("onboarding-explore").click().catch(() => {});
await page.waitForFunction(() => window.__twin.sites.summaries.length > 0, undefined, { timeout: 120_000 });
await page.evaluate((id) => window.__twin.sites.flyTo(id), siteId);
await page.waitForFunction((id) => window.__twin.sites.flight?.state === "landed" && window.__twin.sites.activeSite?.id === id, siteId, { timeout: 300_000 });
await page.waitForTimeout(Number(process.env.SETTLE_MS ?? 20000));
await page.screenshot({ path: out });
await browser.close();
