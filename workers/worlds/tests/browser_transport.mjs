/** Real Chromium/aiortc DTLS, encoded video and DataChannel integration; fixture output is synthetic. */
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { createConnection, createServer } from "node:net";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../../", import.meta.url));
const require = createRequire(resolve(root, "apps/web/package.json"));
const { chromium } = require("@playwright/test");
const ts = require("typescript");
const source = await readFile(resolve(root, "apps/web/src/worlds/player/transport.ts"), "utf8");
const script = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2023, module: ts.ModuleKind.ESNext },
}).outputText;
let turn;
let turnEnv = {};
if (process.env.WORLD_TEST_TURN_BINARY) {
  const port = await new Promise((resolve) => {
    const server = createServer();
    server.listen(0, "127.0.0.1", () => {
      const port = server.address().port;
      server.close(() => resolve(port));
    });
  });
  const secret = randomBytes(32).toString("hex");
  turn = spawn(
    process.env.WORLD_TEST_TURN_BINARY,
    [
      "--listening-ip=127.0.0.1",
      "--relay-ip=127.0.0.1",
      `--listening-port=${port}`,
      "--min-port=49160",
      "--max-port=49169",
      "--no-udp",
      "--no-tls",
      "--no-dtls",
      "--no-cli",
      "--realm=worlds-local-test",
      "--use-auth-secret",
      `--static-auth-secret=${secret}`,
      "--allow-loopback-peers",
      "--no-multicast-peers",
      "--relay-threads=1",
      "--total-quota=4",
      "--user-quota=4",
      "--log-file=/tmp/worlds-turn-test.log",
      "--pidfile=/tmp/worlds-turn-test.pid",
      "--no-stdout-log",
    ],
    { stdio: ["ignore", "ignore", "pipe"] },
  );
  turn.stderr.on("data", (data) => process.stderr.write(data));
  await new Promise((resolve, reject) => {
    const deadline = Date.now() + 10000;
    function check() {
      const socket = createConnection({ host: "127.0.0.1", port });
      socket.on("connect", () => {
        socket.destroy();
        resolve();
      });
      socket.on("error", () => {
        socket.destroy();
        if (Date.now() > deadline) reject(Error("Loopback TURN startup timed out"));
        else setTimeout(check, 100);
      });
    }
    check();
  });
  turnEnv = {
    WORLD_TURN_URLS: `turn:127.0.0.1:${port}?transport=tcp`,
    WORLD_TURN_SECRET: secret,
    WORLD_ICE_TRANSPORT_POLICY: "relay",
  };
}
const child = spawn(
  process.env.WORLD_TEST_PYTHON ?? "python",
  [resolve(root, "workers/worlds/tests/streaming_fixture.py")],
  { stdio: ["ignore", "pipe", "pipe"], env: { ...process.env, ...turnEnv } },
);
let browser;
try {
  const fixture = await new Promise((resolve, reject) => {
    let stdout = "";
    const timeout = setTimeout(
      () => reject(new Error("Synthetic fixture startup timed out")),
      15000,
    );
    child.once("exit", (code) => {
      clearTimeout(timeout);
      reject(new Error(`Fixture exited with ${code}`));
    });
    child.stdout.on("data", (data) => {
      stdout += data;
      if (stdout.includes("\n")) {
        clearTimeout(timeout);
        resolve(JSON.parse(stdout.split("\n")[0]));
      }
    });
    child.stderr.on("data", (data) => process.stderr.write(data));
  });
  const request = (path, options = {}) =>
    fetch(`${fixture.url}${path}`, {
      ...options,
      headers: { Authorization: `Bearer ${fixture.token}`, "Content-Type": "application/json" },
    });
  let httpActions = 0;
  browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox"],
    executablePath:
      process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH ??
      (existsSync("/usr/bin/chromium") ? "/usr/bin/chromium" : undefined),
  });
  const page = await browser.newPage({ viewport: { width: 960, height: 640 } });
  await page.addInitScript(() => {
    const Native = window.RTCPeerConnection;
    window.RTCPeerConnection = class extends Native {
      constructor(...args) {
        super(...args);
        window.testPeer = this;
        window.iceStates = [];
        this.addEventListener("iceconnectionstatechange", () =>
          window.iceStates.push(this.iceConnectionState),
        );
      }
    };
  });
  const errors = [];
  const signaling = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("https://world-transport.test/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/")
      return route.fulfill({
        contentType: "text/html",
        body: `<html><body style="margin:30px;background:#09121b;color:#cce3bf;font:16px system-ui"><h2>Encoded WebRTC transport verification</h2><p>Synthetic video fixture · no AI inference · software encoder</p><video id="video" muted autoplay playsinline style="width:100%"></video><pre id="status"></pre><script type="module">import {connectVideo} from '/transport.js';window.metrics={};window.fallbackFrames=0;const api={request:async(path,init)=>{const r=await fetch(path,init);if(!r.ok)throw Error('HTTP '+r.status);return r.json()},frame:async(path,signal)=>{window.fallbackFrames++;const r=await fetch(path,{signal});return r.status===204?null:r.blob()}};window.controller=new AbortController();window.connection=await connectVideo(api,${JSON.stringify(fixture.sessionId)},document.querySelector('video'),controller.signal,s=>document.querySelector('#status').textContent=s,image=>image.close(),m=>Object.assign(window.metrics,m),{audio:${process.env.WORLD_TEST_AUDIO === "1"}});window.ready=true;</script></body></html>`,
      });
    if (url.pathname === "/transport.js")
      return route.fulfill({ contentType: "application/javascript", body: script });
    if (url.pathname.endsWith("/actions")) httpActions++;
    const upstream = await request(url.pathname, {
      method: route.request().method(),
      body: route.request().postData() ?? undefined,
    });
    if (!url.pathname.endsWith("/frame"))
      signaling.push({
        path: url.pathname,
        status: upstream.status,
        ...(!upstream.ok ? { error: await upstream.clone().text() } : {}),
      });
    return route.fulfill({
      status: upstream.status,
      contentType: upstream.headers.get("content-type") ?? "application/json",
      body: Buffer.from(await upstream.arrayBuffer()),
    });
  });
  await page.goto("https://world-transport.test/");
  await page
    .waitForFunction(
      () => window.ready && window.metrics.framesDecoded > 3 && window.metrics.codec,
      undefined,
      { timeout: 30000 },
    )
    .catch(async (error) => {
      console.error(
        JSON.stringify({
          browser: await page.evaluate(() => ({
            ready: window.ready,
            metrics: window.metrics,
            status: document.querySelector("#status")?.textContent,
            fallback: window.fallbackFrames,
            ice: window.iceStates,
            candidates: window.testPeer?.localDescription?.sdp
              .split("\n")
              .filter((line) => line.startsWith("a=candidate")),
            remote: window.testPeer?.remoteDescription?.sdp
              .split("\n")
              .filter((line) => line.startsWith("a=candidate")),
          })),
          errors,
          signaling,
        }),
      );
      throw error;
    });
  const result = await page.evaluate(async () => {
    const ack = await window.connection.sendAction({
      type: "native",
      action: "forward",
      values: { pressed: true },
    });
    let rejected = false;
    try {
      await window.connection.sendAction({ type: "native", action: "invented-action" });
    } catch {
      rejected = true;
    }
    const reports = await window.testPeer.getStats();
    const audio = [...reports.values()].find(
      (report) => report.type === "inbound-rtp" && report.kind === "audio",
    );
    return {
      audioTracks: document.querySelector("video").srcObject.getAudioTracks().length,
      audioSamples: audio?.totalSamplesReceived ?? 0,
      ack,
      rejected,
      metrics: window.metrics,
      fallbackFrames: window.fallbackFrames,
      width: document.querySelector("video").videoWidth,
      height: document.querySelector("video").videoHeight,
    };
  });
  if (process.env.WORLD_TEST_AUDIO === "1") {
    assert.equal(result.audioTracks, 1);
    assert.ok(result.audioSamples > 0, "Browser must receive actual decoded audio samples");
  }
  assert.equal(result.ack.accepted, true);
  assert.equal(result.ack.appliesAt, "next-clip");
  assert.equal(result.rejected, true);
  assert.equal(result.fallbackFrames, 0);
  assert.equal(httpActions, 0);
  assert.deepEqual([result.width, result.height], [640, 360]);
  const expectedEncoder =
    process.env.WORLD_VIDEO_ENCODER === "h264-software" ? "software-h264" : "software";
  assert.equal(result.metrics.encoder, expectedEncoder);
  if (expectedEncoder === "software-h264") assert.equal(result.metrics.codec, "video/H264");
  assert.match(result.metrics.codec, /^video\/(VP8|H264)$/i);
  assert.deepEqual(errors, []);
  await page.screenshot({
    path: process.env.WORLD_TEST_SCREENSHOT ?? "/tmp/worlds-webrtc-browser.png",
  });
  await page.evaluate(() => {
    window.controller.abort();
    if (document.querySelector("video").srcObject !== null) throw Error("Stream was not released");
  });
  assert.equal((await request(`/sessions/${fixture.sessionId}`, { method: "DELETE" })).status, 204);
  assert.equal((await request(`/sessions/${fixture.sessionId}`)).status, 404);
  console.info(
    JSON.stringify(
      {
        test: "real-chromium-webrtc",
        fixture: "synthetic-no-inference",
        ice: turn ? "TURN-over-TCP" : "direct",
        ...result,
        cleanup: "verified",
      },
      null,
      2,
    ),
  );
} finally {
  await browser?.close();
  child.kill("SIGTERM");
  await new Promise((resolve) => {
    if (child.exitCode !== null) resolve();
    else child.once("exit", resolve);
  });
  turn?.kill("SIGTERM");
  if (turn)
    await new Promise((resolve) => {
      if (turn.exitCode !== null) resolve();
      else turn.once("exit", resolve);
    });
}
