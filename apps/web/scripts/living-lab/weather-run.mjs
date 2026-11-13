// One heavy job owns the static server and exactly one browser; no production traffic.
// heavy node apps/web/scripts/living-lab/weather-run.mjs smoke|record|bench|controls forest|crown|tree [hour]
import { spawn } from "node:child_process";
import { mkdir, readFile, writeFile, appendFile } from "node:fs/promises";
import { createWriteStream } from "node:fs";
import { resolve } from "node:path";
import { homedir } from "node:os";
import { randomUUID } from "node:crypto";
import { chromium } from "@playwright/test";
const [task = "smoke", scene = "tree", hour = "storm"] = process.argv.slice(2);
if (
  ![
    "smoke",
    "record",
    "record-hours",
    "bench",
    "controls",
    "device",
    "feed",
    "tier-a",
    "step-check",
  ].includes(task) ||
  !["forest", "forest-oblique", "forest-edge", "forest-overhead", "crown", "tree"].includes(scene)
)
  throw new Error("Invalid task/scene");
const isForest = scene.startsWith("forest");
const port = Number(process.env.LAB_PORT ?? 5720),
  readyToken = randomUUID();
if (!Number.isInteger(port) || port < 5720 || port > 5729) throw new Error("Ports 5720–5729 only");
const address = `http://127.0.0.1:${port}`;
const cache = resolve(homedir(), ".cache/hexapod-living-weather"),
  out = resolve(process.env.LAB_OUT ?? `${cache}/artifacts`);
await mkdir(out, { recursive: true });
const log = createWriteStream(`${cache}/serve-${task}-${scene}.log`);
const server = spawn(
  process.execPath,
  [
    "apps/web/scripts/living-lab/serve-bench.mjs",
    "--host",
    "127.0.0.1",
    "--port",
    String(port),
    "--ready-token",
    readyToken,
  ],
  { stdio: ["ignore", "pipe", "pipe"], env: process.env },
);
server.stdout.pipe(log);
server.stderr.pipe(log);
let browser, progress, closing;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const close = () =>
  (closing ??= (async () => {
    clearInterval(progress);
    await browser?.close();
    if (server.exitCode === null) {
      const exited = new Promise((r) => server.once("exit", r));
      server.kill("SIGTERM");
      await Promise.race([exited, sleep(3000)]);
      if (server.exitCode === null) server.kill("SIGKILL");
      await exited;
    }
    log.end();
  })());
process.on("SIGINT", () => {
  void close().then(() => process.exit(130));
});
process.on("SIGTERM", () => {
  void close().then(() => process.exit(143));
});
try {
  let ready = false;
  for (let i = 0; i < 180; i++) {
    if (server.exitCode !== null) throw new Error("Static server exited");
    try {
      const response = await fetch(`${address}/__living_ready`);
      if (response.ok && (await response.json()).token === readyToken) {
        ready = true;
        break;
      }
    } catch {
      /* The static build is still starting. */
    }
    await sleep(1000);
  }
  if (!ready) throw new Error("Static build/server timeout");
  browser = await chromium.launch({
    args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--disable-dev-shm-usage"],
    headless: true,
  });
  const deviceViewport = task === "device" || task === "feed";
  const page = await browser.newPage({
    viewport: { width: deviceViewport ? 1920 : 1024, height: deviceViewport ? 1600 : 640 },
    deviceScaleFactor: 1,
  });
  // Full-HD SwiftShader input dispatch can exceed Playwright's 30-second action default.
  page.setDefaultTimeout(120000);
  const logs = [];
  page.on("pageerror", (e) => logs.push(String(e)));
  page.on("console", (m) => {
    if (["error", "warning"].includes(m.type())) logs.push(m.text().slice(0, 500));
  });
  await page.route("**/*", (route) => {
    const u = new URL(route.request().url());
    return u.hostname === "127.0.0.1" ||
      u.hostname.endsWith("open-meteo.com") ||
      ["blob:", "data:"].includes(u.protocol)
      ? route.continue()
      : route.abort();
  });
  const path = `${address}/?preset=off${isForest ? `&living-forest=1&view=${scene.split("-")[1] ?? "oblique"}` : task === "tier-a" ? "&living-scan=spool" : ""}${task === "device" ? "&rounds=1&samples=2" : ""}`;
  progress = setInterval(() => {
    void page
      .evaluate(() => ({
        status: window.__lab?.status(),
        note: document.getElementById("status")?.textContent,
      }))
      .then((s) => console.info("Loading", JSON.stringify(s)))
      .catch(() => undefined);
  }, 15000);
  await page.goto(path);
  await page
    .waitForFunction(
      () => window.__weather && document.getElementById("run").disabled === false,
      undefined,
      { timeout: 240000 },
    )
    .catch(async (e) => {
      console.info(
        "Load failure",
        await page.evaluate(() => ({
          status: window.__lab?.status(),
          note: document.getElementById("status").textContent,
        })),
      );
      console.info(logs);
      throw e;
    });
  clearInterval(progress);
  console.info("Loaded", scene, await page.evaluate(() => window.__lab.status()));
  const ev = (s) => page.evaluate(s);
  const settle = async (patch) =>
    ev(
      `(async()=>{const before=__lab.status().frames;LL.getState().set(${JSON.stringify(patch)});if(window.__manualCapture)__lab.widget.render();while(__lab.status().frames<=before || __lab.status().loading || !__lab.status().settled)await new Promise(r=>setTimeout(r,10));__weather.overlay();return __lab.status();})()`,
    );
  const configure = async (arm) => {
    // A subsequent hour must be allowed to finish async sidecar loading even
    // after capture has stopped the automatic loops. No pulse runs while recording.
    const pulse = await ev(
      "window.__manualCapture ? setInterval(()=>__lab.widget.render(),100) : null",
    );
    try {
      await ev(`__weather.preset(${JSON.stringify(arm)})`);
      await settle({ clock: 0 });
    } finally {
      if (pulse !== null) await ev(`clearInterval(${pulse})`);
    }
  };
  const resize = async (w, h) => {
    await ev(
      `(()=>{document.querySelector('main').style.display='none';const v=document.getElementById('viewer');v.style.width='${w}px';v.style.height='${h}px';__lab.widget.resize();return true;})()`,
    );
    await settle({ clock: 0, bench: true });
  };
  const shot = async (file, decorated = true) => {
    const data =
      await ev(`(()=>{const viewer=document.getElementById('viewer');const c=document.createElement('canvas');c.width=viewer.clientWidth;c.height=viewer.clientHeight;const x=c.getContext('2d');x.fillStyle='#9fb3c8';x.fillRect(0,0,c.width,c.height);for(const source of viewer.querySelectorAll('canvas')){if(source.id==='field')continue;const gl=source.getContext('webgl2');if(gl?.getContextAttributes()?.preserveDrawingBuffer)x.drawImage(source,0,0,c.width,c.height);}
  ${decorated ? `if(__weather.forest && new URLSearchParams(location.search).get("view")==="overhead")x.drawImage(document.getElementById('field'),c.width-236,12,224,248);x.fillStyle='#142333ee';x.fillRect(0,c.height-62,c.width,62);x.fillStyle='white';x.font='14px sans-serif';const caption=document.getElementById('caption').textContent;const split=caption.indexOf(' · mean');x.fillText(split<0?caption:caption.slice(0,split),16,c.height-36);x.fillText(split<0?'':caption.slice(split+3),16,c.height-14);` : ""}
  return c.toDataURL('image/png').split(',')[1];})()`);
    await writeFile(file, Buffer.from(data, "base64"));
  };
  if (task === "device" || task === "feed") {
    if (task === "device") {
      await page.getByRole("button", { name: "Run interleaved benchmark" }).click();
      await page.waitForFunction(() => window.__breezeResult, undefined, { timeout: 1200000 });
      const result = await ev("__breezeResult");
      await writeFile(`${out}/device-${scene}.json`, JSON.stringify(result, null, 2));
      const downloadPromise = page.waitForEvent("download");
      await page.getByRole("button", { name: "Download JSON" }).click();
      const downloaded = await downloadPromise;
      await downloaded.saveAs(`${out}/device-${scene}-download.json`);
      const actual = JSON.parse(await readFile(`${out}/device-${scene}-download.json`, "utf8"));
      if (JSON.stringify(result) !== JSON.stringify(actual))
        throw new Error("Downloaded bench result differs");
    }
    // A feed-only run has not selected a benchmark arm to initialize the date.
    await page.locator("#date").fill("2024-12-04T21:00");
    await page.selectOption("#source", "live");
    await page
      .waitForFunction(
        () => document.getElementById("status").textContent.includes("Open-Meteo"),
        undefined,
        { timeout: 240000 },
      )
      .catch(async (error) => {
        console.info(
          "Feed state",
          await ev(
            "({note:document.getElementById('status').textContent,date:document.getElementById('date').value,wind:LL.getState().weather,motion:LL.getState().motion})",
          ),
        );
        throw error;
      });
    const live = await ev(
      "({note:document.getElementById('status').textContent,wind:LL.getState().weather,motion:LL.getState().motion})",
    );
    await page.route("https://archive-api.open-meteo.com/**", (route) =>
      route.fulfill({ status: 503, body: "unavailable" }),
    );
    await page.locator("#date").fill("2024-12-03T21:00");
    await page.locator("#date").dispatchEvent("change");
    await page.waitForFunction(
      () => document.getElementById("status").textContent.includes("Weather HTTP 503"),
      undefined,
      { timeout: 240000 },
    );
    const fallback = await ev(
      "({note:document.getElementById('status').textContent,wind:LL.getState().weather,motion:LL.getState().motion})",
    );
    if (fallback.wind !== null || fallback.motion !== "off")
      throw new Error("Weather failure left wind moving");
    const recorded = JSON.parse(
      await readFile(
        new URL("../../src/lib/weather/fixtures/minnetonka-hours.json", import.meta.url),
        "utf8",
      ),
    );
    const payload = (day, mean) =>
      JSON.stringify({
        ...recorded,
        hourly: {
          time: [`${day}T21:00`],
          wind_speed_10m: [mean],
          wind_gusts_10m: [mean + 4],
          wind_direction_10m: [90],
        },
      });
    let delayed;
    await page.route("https://archive-api.open-meteo.com/**", (route) => {
      const day = new URL(route.request().url()).searchParams.get("start_date");
      if (day === "2024-12-02") {
        delayed = route;
        return;
      }
      return route.fulfill({ status: 200, contentType: "application/json", body: payload(day, 4) });
    });
    // Observe before starting: a slow software-rendered scene can outlast the
    // feed's 12-second timeout before the later response is released.
    const oldNetworkDone = new Promise((resolve) => {
      const finish = (request, outcome) => {
        if (!request.url().includes("start_date=2024-12-02")) return;
        page.off("requestfinished", finished);
        page.off("requestfailed", failed);
        resolve({ outcome, failure: request.failure()?.errorText ?? null });
      };
      const finished = (request) => finish(request, "finished");
      const failed = (request) => finish(request, "failed");
      page.on("requestfinished", finished);
      page.on("requestfailed", failed);
    });
    const oldRequest = page.waitForRequest((r) => r.url().includes("start_date=2024-12-02"));
    await page.locator("#date").fill("2024-12-02T21:00");
    await page.locator("#date").dispatchEvent("change");
    await oldRequest;
    await page.locator("#date").fill("2024-12-01T21:00");
    await page.locator("#date").dispatchEvent("change");
    await page.waitForFunction(
      () => window.LL.getState().weather?.time === "2024-12-01T21:00Z",
      undefined,
      { timeout: 240000 },
    );
    while (!delayed) await sleep(10);
    await delayed.fulfill({
      status: 200,
      contentType: "application/json",
      body: payload("2024-12-02", 20),
    });
    const oldOutcome = await oldNetworkDone;
    const race = await ev(
      "new Promise(r=>setTimeout(()=>r({wind:LL.getState().weather,note:document.getElementById('status').textContent}),200))",
    );
    if (race.wind?.time !== "2024-12-01T21:00Z" || race.wind.meanMps !== 4)
      throw new Error("An old weather response replaced the selected hour");
    await writeFile(
      `${out}/feed-${scene}-browser.json`,
      JSON.stringify(
        {
          live,
          fallback,
          fallbackSource: "intercepted HTTP 503",
          race,
          oldOutcome,
          raceSource: "intercepted fixture with deliberately relabelled test dates",
        },
        null,
        2,
      ),
    );
  } else {
    await resize(task === "bench" ? 512 : 1024, task === "bench" ? 320 : 640);
    if (scene === "crown") {
      await ev("__lab.view([0,0,4.6],35,-12,7.8)");
      await settle({ clock: 0 });
    }
    if (task === "step-check") {
      await configure("storm");
      await settle({ clock: 3 });
      await shot(`${out}/step-${scene}-normal.png`, false);
      await ev(
        "(()=>{__lab.widget.useDefaultRenderLoop=false;window.requestAnimationFrame=()=>0;window.__manualCapture=true;return true;})()",
      );
      await settle({ clock: 3 });
      await shot(`${out}/step-${scene}-manual.png`, false);
    } else if (task === "smoke") {
      await configure(hour);
      await shot(`${out}/smoke-${scene}-${hour}.png`);
      await writeFile(
        `${out}/smoke-${scene}-${hour}.json`,
        JSON.stringify(
          await ev("({status:__lab.status(),lab:LL.getState(),renderer:navigator.userAgent})"),
          null,
          2,
        ),
      );
    } else if (task === "record" || task === "record-hours") {
      for (const selectedHour of task === "record-hours"
        ? (process.env.LAB_HOURS?.split(",") ?? ["calm", "light", "fresh", "storm"])
        : [hour]) {
        await configure(selectedHour);
        // A fixed-clock recording needs exactly one completed draw per source frame.
        // Stop repeated identical draws between PNG reads; the normal/fixed pair is pixel-tested.
        await ev(
          "(()=>{__lab.widget.useDefaultRenderLoop=false;window.requestAnimationFrame=()=>0;window.__manualCapture=true;return true;})()",
        );
        const frames = `${out}/${scene}-${selectedHour}-frames`;
        await mkdir(frames, { recursive: true });
        for (let i = 0; i < 192; i++) {
          const file = `${frames}/${String(i).padStart(3, "0")}.png`;
          await settle({ clock: i / 24 });
          await shot(file);
          if (i % 24 === 0) console.info(scene, selectedHour, `${i}/192`);
        }
        await writeFile(
          `${out}/${scene}-${selectedHour}-record.json`,
          JSON.stringify(
            {
              frames: 192,
              fps: 24,
              width: 1024,
              height: 640,
              clock: "i/24",
              interpolation: false,
              status: await ev("__lab.status()"),
              weather: await ev(
                `__weather.hours[__weather.names.indexOf(${JSON.stringify(selectedHour)})]`,
              ),
            },
            null,
            2,
          ),
        );
      }
    } else if (task === "bench") {
      for (let round = 0; round < 3; round++)
        for (const arm of ["off", "calm", "off", "storm"]) {
          await configure(arm);
          const samples = await ev(
            "(async()=>{LL.getState().set({clock:null,bench:true});window.__livingBench=[];while(__livingBench.length<12)await new Promise(r=>setTimeout(r,50));return __livingBench.slice(4,12);})()",
          );
          await appendFile(
            `${out}/bench-${scene}-raw.jsonl`,
            JSON.stringify({ round, arm, samples, status: await ev("__lab.status()") }) + "\n",
          );
          console.info("bench", scene, round, arm);
        }
    } else if (task === "controls" || task === "tier-a") {
      const controlScene = task === "tier-a" ? "tier-a" : scene;
      await configure("off");
      await settle({ clock: 0 });
      await shot(`${out}/control-${controlScene}-plain.png`, false);
      await configure("storm");
      await settle({ clock: 2 });
      await shot(`${out}/control-${controlScene}-moving.png`, false);
      await ev("captured.getState().set(true)");
      await settle({ clock: 3 });
      await shot(`${out}/control-${controlScene}-captured.png`, false);
      await ev("captured.getState().set(false)");
      await configure("off");
      await settle({ clock: 0 });
      await shot(`${out}/control-${controlScene}-toggle-off.png`, false);
      await configure("storm");
      await settle({ clock: 4, strength: 0 });
      await shot(`${out}/control-${controlScene}-zero.png`, false);
      await ev("captured.getState().set(true)");
      await page.reload();
      await page.waitForFunction(
        () => window.__weather && document.getElementById("run").disabled === false,
        undefined,
        { timeout: 240000 },
      );
      await resize(1024, 640);
      await settle({ clock: 0 });
      await shot(`${out}/control-${controlScene}-reload.png`, false);
      await writeFile(
        `${out}/control-${controlScene}-reload.json`,
        JSON.stringify(
          await ev("({captured:captured.getState().on,status:__lab.status()})"),
          null,
          2,
        ),
      );
    }
  }
  await writeFile(`${out}/${task}-${scene}-console.json`, JSON.stringify(logs, null, 2));
  if (logs.some((s) => /Shader|Compile|ReferenceError|TypeError/.test(s)))
    throw new Error(logs.join("\n"));
  console.info("Complete", task, scene);
} finally {
  await close();
}
