import { Group, Scene, WebGLRenderer } from "three";
import type { SplatMesh } from "@sparkjsdev/spark";
import { deviceSplatBudget } from "@/lib/detail";
import {
  consoleSpark,
  consoleSplat,
  sparkSettled,
  sparkLodSettled,
  quiesceSpark,
  type ConsoleBudget,
} from "@/view/liveSpark";
import { ConsoleCheckSettle, type CheckProgress } from "./consoleCheckSettle";
import { openScanStream } from "@/view/sparkStream";
import { psnr, referenceCamera, rgbFloats, type ReferenceCamera } from "./consoleCheck";

interface CheckInput {
  camera: ReferenceCamera;
  source: string;
  tiles: boolean;
  reference: string;
  budget?: ConsoleBudget;
  antialiased: boolean;
  timeoutMs?: number;
}
interface CheckOutput {
  png: string;
  psnr: number | null;
  budget: ConsoleBudget;
  drawnSplats: number;
  activeSplats: number;
  tileGaussians: number;
  seconds: number;
  tiles: string[];
}

declare global {
  interface Window {
    consoleCheck: (input: CheckInput) => Promise<CheckOutput>;
    consoleCheckProgress?: CheckProgress & { seconds: number };
  }
}

async function pixels(url: string, width: number, height: number): Promise<Float64Array> {
  const image = new Image();
  image.src = url;
  await image.decode();
  if (image.naturalWidth !== width || image.naturalHeight !== height)
    throw new Error("Reference dimensions differ");
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("No 2D canvas");
  context.drawImage(image, 0, 0);
  return rgbFloats(context.getImageData(0, 0, width, height).data);
}

window.consoleCheck = async (input) => {
  delete window.consoleCheckProgress;
  const { width, height } = input.camera;
  const budget = input.budget ?? deviceSplatBudget();
  const timeoutMs = input.timeoutMs ?? 900_000;
  if (!(Number.isFinite(timeoutMs) && timeoutMs > 0)) throw new Error("Invalid timeout");
  const reference = await pixels(input.reference, width, height);
  const renderer = new WebGLRenderer({ antialias: false });
  renderer.setPixelRatio(1);
  renderer.setSize(width, height);
  renderer.setClearColor(0x000000, 1);
  document.body.replaceChildren(renderer.domElement);
  const camera = referenceCamera(input.camera);
  const scene = new Scene();
  const group = new Group(); // Identity solve frame: no up-axis or ENU rotation.
  const spark = consoleSpark(renderer, budget, input.antialiased);
  scene.add(spark, group);
  let mesh: SplatMesh | null = null;
  let stream: ReturnType<typeof openScanStream> | null = null;
  let tiles: string[] = [];
  let tileGaussians = 0;
  let loaded = false;
  let closed = false;
  let failure: Error | undefined;
  const started = performance.now();
  const sourceAbort = new AbortController();
  const releaseSources = (): void => {
    stream?.stop();
    mesh?.dispose();
  };
  const gate = new ConsoleCheckSettle();
  const progress = (): CheckProgress => ({
    stage: !loaded ? "loading-source" : stream && !stream.settled ? "loading-tiles" : "settling",
    tiles,
    tileGaussians,
    drawnSplats: spark.display.numSplats,
    activeSplats: spark.activeSplats,
    sortSettled: sparkSettled(spark),
    lodSettled: sparkLodSettled(spark),
    version: spark.current.version,
    mappingVersion: spark.current.mappingVersion,
  });
  try {
    // Root loading is inside the same deadline as subsequent tiles and GPU work.
    const load = async (): Promise<void> => {
      if (input.tiles) {
        stream = openScanStream(input.source, group, camera, renderer, spark, {
          fullResolution: budget === "full",
          projectionFromMatrix: true,
          report: (drawn) => {
            tiles = drawn.tiles.map((tile) => tile.uri);
            tileGaussians = drawn.gaussians;
          },
        });
        await stream.ready;
      } else {
        const response = await fetch(input.source, { signal: sourceAbort.signal });
        if (!response.ok) throw new Error(`Source answered ${response.status}`);
        mesh = consoleSplat(new Uint8Array(await response.arrayBuffer()), budget);
        await mesh.initialized;
        if (closed) {
          mesh.dispose();
          return;
        }
        group.add(mesh);
      }
      loaded = true;
    };
    void load().catch((error: unknown) => {
      failure = error instanceof Error ? error : new Error(String(error));
    });
    const png = await new Promise<string>((resolve, reject) => {
      let lastReport = started;
      renderer.setAnimationLoop(() => {
        try {
          if (failure !== undefined) throw failure;
          stream?.frame();
          const now = performance.now();
          const state = progress();
          window.consoleCheckProgress = { ...state, seconds: (now - started) / 1000 };
          if (now - lastReport >= 5000) {
            console.info(
              "console-check progress",
              JSON.stringify({ seconds: (now - started) / 1000, ...state }),
            );
            lastReport = now;
          }
          // Sample BEFORE onBeforeRender starts Spark's next exclusive LOD task. Require
          // a stable displayed generation as well as completed LOD, sort and tile work.
          const settled = gate.observe(now, state);
          document.body.dataset.settled = String(settled);
          if (settled) {
            spark.autoUpdate = false;
            spark.enableDriveLod = false;
            renderer.render(scene, camera); // Capture this exact display without scheduling more work.
            renderer.setAnimationLoop(null);
            resolve(renderer.domElement.toDataURL("image/png"));
          } else if (now - started > timeoutMs) {
            throw new Error(
              `Timed out waiting for tiles and Spark settled: ${JSON.stringify(state)}`,
            );
          } else if (state.stage === "settling") {
            // The reference camera is fixed. Decode the complete cut first so software GL
            // does not spend the deadline accumulating/sorting every intermediate ancestor.
            renderer.render(scene, camera);
          }
        } catch (error) {
          renderer.setAnimationLoop(null);
          reject(error instanceof Error ? error : new Error(String(error)));
        }
      });
    });
    const score = psnr(await pixels(png, width, height), reference);
    return {
      png,
      psnr: Number.isFinite(score) ? score : null,
      budget,
      drawnSplats: spark.display.numSplats,
      activeSplats: spark.activeSplats,
      tileGaussians,
      tiles,
      seconds: (performance.now() - started) / 1000,
    };
  } finally {
    closed = true;
    sourceAbort.abort();
    renderer.setAnimationLoop(null);
    await quiesceSpark(spark);
    releaseSources();
    spark.dispose();
    renderer.dispose();
    renderer.forceContextLoss();
  }
};
