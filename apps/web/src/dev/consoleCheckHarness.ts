import { Group, Scene, WebGLRenderer } from "three";
import type { SplatMesh } from "@sparkjsdev/spark";
import { deviceSplatBudget } from "@/lib/detail";
import { consoleSpark, consoleSplat, sparkSettled, type ConsoleBudget } from "@/view/liveSpark";
import { openScanStream } from "@/view/sparkStream";
import { psnr, referenceCamera, rgbFloats, type ReferenceCamera } from "./consoleCheck";

interface CheckInput {
  camera: ReferenceCamera;
  source: string;
  tiles: boolean;
  reference: string;
  budget?: ConsoleBudget;
  antialiased: boolean;
}
interface CheckOutput {
  png: string;
  psnr: number | null;
  budget: ConsoleBudget;
  drawnSplats: number;
  tiles: string[];
}

declare global {
  interface Window {
    consoleCheck: (input: CheckInput) => Promise<CheckOutput>;
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
  const { width, height } = input.camera;
  const budget = input.budget ?? deviceSplatBudget();
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
  try {
    if (input.tiles) {
      stream = openScanStream(input.source, group, camera, renderer, spark, {
        fullResolution: budget === "full",
        report: (drawn) => {
          tiles = drawn.tiles.map((tile) => tile.uri);
        },
      });
      await stream.ready;
    } else {
      const response = await fetch(input.source);
      if (!response.ok) throw new Error(`Source answered ${response.status}`);
      mesh = consoleSplat(new Uint8Array(await response.arrayBuffer()), budget);
      await mesh.initialized;
      group.add(mesh);
    }
    const png = await new Promise<string>((resolve, reject) => {
      const deadline = performance.now() + 180_000;
      let stableFrames = 0;
      renderer.setAnimationLoop(() => {
        try {
          stream?.frame();
          renderer.render(scene, camera);
          // Both the desired tile cut and Spark's asynchronous accumulator/sort must
          // finish. Two consecutive frames also reject the initial empty accumulator.
          const settled =
            (stream?.settled ?? true) && spark.display.numSplats > 0 && sparkSettled(spark);
          document.body.dataset.settled = String(settled);
          stableFrames = settled ? stableFrames + 1 : 0;
          if (stableFrames >= 2) {
            renderer.setAnimationLoop(null);
            resolve(renderer.domElement.toDataURL("image/png"));
          } else if (performance.now() > deadline) {
            throw new Error("Timed out waiting for tiles and Spark settled");
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
      tiles,
    };
  } finally {
    renderer.setAnimationLoop(null);
    stream?.stop();
    mesh?.dispose();
    spark.dispose();
    renderer.dispose();
    renderer.forceContextLoss();
  }
};
